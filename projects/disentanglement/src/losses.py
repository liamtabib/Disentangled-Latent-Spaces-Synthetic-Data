import torch
import torch.nn as nn
import sys
sys.path.append(".")
import torch.nn.functional as F
import cv2
import dlib
import numpy as np

class DiscriminatorLoss(nn.Module):
    """
    Implements the discriminator loss, which can optionally include only the second half of the discriminator predictions.
    This loss compares the predicted labels from discriminator against the true labels for ID matching.
    """

    def __init__(self, second):
        """
        Initializes the DiscriminatorLoss class.
        
        Args:
            second (bool): If True, only the second half of the discriminator predictions will be used.
        """
        super(DiscriminatorLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss for binary classification tasks.
        self.second = second # a variable that is TRUE if we are to only use the ID network for the second half

    def forward(self, discriminator_firsthalf_preds, discriminator_secondhalf_preds, targets, T_turn):
        """
        Computes the discriminator loss: based on a concatenation of the second half of two latent vectors, a network predicts if
        they belong to the same ID or not, and a minimax game is started where the NICE network wants to minimize the accuracy of this
        network. Optionally also play a MaxMax game with a network trained to predict same ID based on the first half.
        
        Args:
            discriminator_firsthalf_preds (torch.Tensor): Predictions from the discriminator of the first half.
            discriminator_secondhalf_preds (torch.Tensor): Predictions from the discriminator of the second half.
            targets (torch.Tensor): True labels for the samples.
            T_turn (bool): If True, invert the targets for loss calculation (used during adversarial training).
        
        Returns:
            torch.Tensor: The computed loss.
        """

        if not self.second:
            # Calculate loss for the first half if not using only the second half.
            loss_firsthalf = self.loss_fn(discriminator_firsthalf_preds, targets)
            
            if T_turn:
                # Invert targets if T_turn is True.
                targets = 1 - targets

            # Calculate loss for the second half.
            loss_secondhalf = self.loss_fn(discriminator_secondhalf_preds, targets)
                    
            return loss_firsthalf + loss_secondhalf  # Return the sum of both halves' losses.
        
        else:            
            if T_turn:
                targets = 1 - targets  # Invert targets if T_turn is True.

            # Calculate loss only for the second half.
            loss_secondhalf = self.loss_fn(discriminator_secondhalf_preds, targets)

            # Cleanup and free memory after computation.
            del targets
            torch.cuda.empty_cache()
            
            return loss_secondhalf
        


def cross_entropy(logits, target, size_average=True):
    """
    Calculates the cross-entropy loss between logits and target labels, optionally averaging the loss over the batch.
    
    Args:
        logits (torch.Tensor): The logits predictions from the model.
        target (torch.Tensor): The true labels.
        size_average (bool): If True, returns the mean loss per batch. Otherwise, returns the sum.
    
    Returns:
        torch.Tensor: The calculated cross-entropy loss.
    """
    if size_average:
        return torch.mean(torch.sum(-target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(-target * F.log_softmax(logits, -1), -1))


class SwitchLoss(nn.Module):
    """
    A binary classification loss that determines if two input images are of the same identity or not based on their embeddings.
    """
    def __init__(self, face_recognition):
        """
        Initializes the SwitchLoss class with a face recognition model to compute embeddings.

        Args:
            face_recognition (torch.nn.Module): The model used to compute embeddings from images.
        """
        super(SwitchLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.face_recognition = face_recognition

    def forward(self, first_image, second_image, same_ID: bool):
        """
        Computes the switch loss between two images based on the cosine similarity of their embeddings.
        
        Args:
            first_image (torch.Tensor): The first image tensor.
            second_image (torch.Tensor): The second image tensor.
            same_ID (bool): True if the images are from the same identity; False otherwise.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        # Compute embeddings for both images.
        embedding1 = self.face_recognition(first_image).squeeze_()
        embedding2 = self.face_recognition(second_image).squeeze_()

        # Compute cosine similarity between embeddings.
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=-1)

        # Create the target tensor based on whether the images are from the same identity.
        target_value = 1.0 if same_ID else 0.0
        target = torch.full_like(similarity, target_value)

        # Compute the loss between the similarity and the target.
        loss = self.loss_fn(similarity, target)
        
        # Return the computed loss.
        return loss


class ContrastiveLoss(nn.Module):
    """
    Implements a contrastive loss for either simple contrastive setting or triplet margin setting.
    This class supports computing either a multilabel N-pair loss, suitable for tasks involving metric learning,
    or a triplet margin loss which focuses on bringing positive samples closer together while pushing negative samples apart.

    Reference: Sohn, K. (2016). Improved deep metric learning with multi-class n-pair loss objective.
    Advances in neural information processing systems, 29.

    Attributes:
        triplet (bool): Determines whether to use triplet margin loss instead of N-pair loss. If True, triplet margin loss is used.
    """

    def __init__(self, triplet: bool):
        """
        Initializes the ContrastiveLoss class.

        Args:
            triplet (bool): If True, uses triplet margin loss instead of N-pair loss. This affects the forward calculation.
        """
        super(ContrastiveLoss, self).__init__()
        self.triplet = triplet

    def forward(self, anchor, positive, target=None, negative=None):
        """
        Computes the loss based on the mode set in the constructor.

        Args:
            anchor (torch.Tensor): The anchor embeddings.
            positive (torch.Tensor): The positive embeddings, i.e., embeddings that should be close to the anchor.
            target (torch.Tensor, optional): The target similarity matrix for N-pair loss. Required if not using triplet loss.
            negative (torch.Tensor, optional): The negative embeddings for triplet loss. Required if using triplet loss.

        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        half_size = anchor.size(1) // 2  # Assuming embeddings are split and only half is used.

        # Slice the embeddings to use only the relevant part.
        anchor = anchor[:, :half_size]
        positive = positive[:, :half_size]

        if not self.triplet:
            # Calculate contrastive loss using N-pair approach.
            batch_size = anchor.size(0)
            logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))  # Compute logits for a mini-batch.
            loss_ce = cross_entropy(logit, target)  # Cross-entropy loss from logits and target matrix.
            l2_loss = torch.sum(anchor**2) / batch_size + torch.sum(positive**2) / batch_size  # Regularization by L2 norm.

            loss = loss_ce + 0.01 * l2_loss  # Combine cross-entropy loss and L2 regularization.

            # Cleanup and free GPU memory.
            del logit, loss_ce, l2_loss
            torch.cuda.empty_cache()

            return loss / 1000  # Scale down the loss to prevent overflow or fast divergence during training.
        
        else:
            # Calculate triplet margin loss.
            negative = negative[:, :half_size]  # Slice the negative embeddings.

            # Calculate similarities.
            positive_similarity = F.cosine_similarity(anchor, positive)
            negative_similarity = F.cosine_similarity(anchor, negative)

            # Compute the margin loss.
            similarity_diff = positive_similarity - negative_similarity
            margin = 0.5  # Define the margin for triplet loss.
            losses = F.relu(margin - similarity_diff)  # Hinge loss: max(0, margin - similarity_diff).

            # Average the losses and add L2 regularization.
            triplet_loss = torch.mean(losses)
            l2_loss = (torch.mean(anchor**2) + torch.mean(positive**2) + torch.mean(negative**2)) / 3

            # Total loss: triplet margin loss plus regularization.
            loss = triplet_loss + 0.01 * l2_loss
            return loss


class LandmarkDetector(nn.Module):
    def __init__(self, predictor_path):
        super(LandmarkDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def forward(self, images):
        device = images.device
        batch_size = images.shape[0]
        images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        batch_bounding_boxes = []

        for i in range(batch_size):
            image_np = cv2.cvtColor(images_np[i], cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            image_bounding_boxes = []

            for face in faces:
                landmarks = self.predictor(gray, face)
                points = np.zeros((68, 2), dtype=int)
                for j in range(68):
                    points[j] = (landmarks.part(j).x, landmarks.part(j).y)

                regions = {
                    "mouth": list(range(48, 61)),
                    "right_eyebrow": list(range(17, 22)),
                    "left_eyebrow": list(range(22, 27)),
                    "right_eye": list(range(36, 42)),
                    "left_eye": list(range(42, 48)),
                    "nose": list(range(27, 36)),
                    "jaw": list(range(0, 17))
                }

                for indices in regions.values():
                    region_points = points[indices]
                    x, y, w, h = cv2.boundingRect(np.array([region_points]))
                    image_bounding_boxes.append([x, y, x + w, y + h])

            batch_bounding_boxes.append(image_bounding_boxes)

        return torch.tensor(batch_bounding_boxes, dtype=torch.int32, device=device)

class LandmarkLoss(nn.Module):
    def __init__(self, lm_network):
        super(LandmarkLoss, self).__init__()
        self.lm_network = lm_network

    def get_bounding_boxes(self, images):
        return self.lm_network(images)

    def forward(self, generator, model, w_star):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        w_plus = model.inverse_T(w_star.view(-1, 16, 512))
        reconstructed_images = generator(w_plus).mul_(0.5).add_(0.5).clamp_(0, 1)
        bounding_boxes = self.get_bounding_boxes(reconstructed_images)
        f
        
        loss = 0.0
        for k in range(bounding_boxes.shape[1]):
            perturbed_w_star = w_star.clone()
            perturbed_w_star[:, k] += torch.randn_like(w_star[:, k])
            
            perturbed_latents = T_inv(perturbed_w_star)
            perturbed_images = generator(perturbed_latents)
            
            pixel_diffs = (original_images - perturbed_images) ** 2
            
            masks = torch.zeros_like(original_images)
            for idx in range(bounding_boxes.shape[0]):
                x, y, xw, yh = bounding_boxes[idx, k]
                masks[idx, :, y:yh, x:xw] = 1
            
            inside_bb_loss = (masks * pixel_diffs).sum()
            outside_bb_loss = ((1 - masks) * pixel_diffs).sum()
            
            loss += inside_bb_loss - outside_bb_loss

        normalized_loss = loss / (bounding_boxes.shape[1] * torch.tensor(original_images.shape[2:]).prod() * original_images.shape[0])
        return normalized_loss
    



# Images generated through G are of dimension 3 x 1020 x 1020
# capital K denotes the total number of landmarks extracted, whilst k goes from 1 to K is an indice of each landmark k.

#For each latent w_star 
# Pass it through T^-1 and G to generate an image: reconstructed I #can be vectorized
# Pass it through landmark model to obtain bounding boxes for K landmark #maybe
# for each landmark k:
# perturb (from a standard normal) w_star in the k'th dimension, and pass it through  T^-1 and G to generate a perturbed image
# Enforce that the perturbation affects only the k'th landmark by encouraging that the pixels that are perturbed are in the
## bounding box of that landmark
# Hence, we compute the pixelwise difference between the reconstructed image and the perturbed image., and punish differences
# in pixels that do not lie in the bounding box of the landmark k and encourage differences that do lie in the bounding box of landmark k.

