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

        probabilities = (similarity + 1) / 2

        # Create the target tensor based on whether the images are from the same identity.
        target_value = 1.0 if same_ID else 0.0
        target = torch.full_like(similarity, target_value)

        # Compute the loss between the similarity and the target.
        loss = F.binary_cross_entropy(probabilities, target)
        
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
            logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))  # Compute logits for a mini-batch.
            loss_ce = cross_entropy(logit, target)  # Cross-entropy loss from logits and target matrix.

            return loss_ce
        
        else:
            # Calculate triplet margin loss.
            negative = negative[:, :half_size]  # Slice the negative embeddings.

            # Calculate similarities.
            positive_similarity = F.cosine_similarity(anchor, positive)
            negative_similarity = F.cosine_similarity(anchor, negative)

            # Compute the margin loss.
            similarity_diff = positive_similarity - negative_similarity
            margin = 0.3  # Define the margin for triplet loss.
            losses = F.relu(margin - similarity_diff)  # Hinge loss: max(0, margin - similarity_diff).

            triplet_loss = torch.mean(losses)
            return triplet_loss


class LandmarkDetector(nn.Module):
    def __init__(self, predictor_path):
        super(LandmarkDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def forward(self, images):
        device = images.device
        batch_size = images.shape[0]
        images_np = images.detach().cpu().numpy()  # Assuming images are already clamped to [0, 1]
        images_np = (images_np * 255).transpose(0, 2, 3, 1).astype(np.uint8)  # Scale and convert
        batch_bounding_boxes = []

        for i in range(batch_size):

            image_np = cv2.cvtColor(images_np[i], cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

            faces = self.detector(gray)
            image_bounding_boxes = []

            if len(faces) == 0:
                return None
            if len(faces) > 1:
                return None

            for face in faces:
                landmarks = self.predictor(gray, face)
                points = np.zeros((68, 2), dtype=int)
                for j in range(68):
                    points[j] = (landmarks.part(j).x, landmarks.part(j).y)

                regions = {
                    "mouth": list(range(48, 61)),
                    #"right_eyebrow": list(range(17, 22)),
                    #"left_eyebrow": list(range(22, 27)),
                    #"right_eye": list(range(36, 42)),
                    #"left_eye": list(range(42, 48)),
                    #"nose": list(range(27, 36)),
                    #"jaw": list(range(0, 17))
                }

                for indices in regions.values():
                    region_points = points[indices]
                    x, y, w, h = cv2.boundingRect(np.array([region_points]))
                    image_bounding_boxes.append([x, y, x + w, y + h])

            batch_bounding_boxes.append(image_bounding_boxes)

        return torch.tensor(batch_bounding_boxes, dtype=torch.int32, device=device)

class LandmarkLoss(nn.Module):
    def __init__(self, lm_network, weight_inside):
        super(LandmarkLoss, self).__init__()
        self.lm_network = lm_network
        self.weight_inside = weight_inside

    def get_bounding_boxes(self, images):
        return self.lm_network(images)

    def forward(self, generator, model, w_star,same_dim: bool,means,variances):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        with torch.no_grad():
            w_plus = model.inverse_T(w_star.view(-1, 16, 512))
            reconstructed_images = generator(w_plus).mul_(0.5).add_(0.5).clamp_(0, 1)
            
            bounding_boxes = self.get_bounding_boxes(reconstructed_images)
        if bounding_boxes is None:
            return torch.tensor(0.0, device=w_star.device, requires_grad=True)
            #print(bounding_boxes.shape)
            
        loss = 0.0
        for k in range(bounding_boxes.shape[1]):
            perturbed_w_star = w_star.clone()

            if same_dim:
                # Perturbing the k'th dimension
                perturbation = torch.normal(means.squeeze()[k], variances.squeeze()[k].sqrt(), size=(w_star.size(0),), device=w_star.device)
                perturbed_w_star[:, k] += perturbation
                perturbed_w_plus = model.inverse_T(perturbed_w_star.view(-1, 16, 512))
                perturbed_images = generator(perturbed_w_plus).mul_(0.5).add_(0.5).clamp_(0, 1)
                pixel_diffs = (reconstructed_images - perturbed_images).pow_(2)  # in-place power operation

                masks = torch.zeros_like(reconstructed_images)
                for idx in range(bounding_boxes.shape[0]):
                    x, y, xw, yh = bounding_boxes[idx, k]
                    masks[idx, :, y:yh, x:xw] = 1
                
                outside_bb_loss = ((1 - masks) * pixel_diffs).sum()
                #inside_bb_loss =  -(masks * pixel_diffs).sum()
                #print(f"outside loss: {outside_bb_loss}")
                #print(f"inside loss: {inside_bb_loss}")

                loss += outside_bb_loss
            else:

                perturbations_other_dims = torch.normal(means, variances.sqrt())
                perturbations_other_dims[:, k] = 0  # Zero out the perturbation for the k'th dimension
                perturbed_w_star_other_dims = w_star + perturbations_other_dims
                perturbed_w_plus_other_dims = model.inverse_T(perturbed_w_star_other_dims.view(-1, 16, 512))
                perturbed_images_other_dims = generator(perturbed_w_plus_other_dims).mul_(0.5).add_(0.5).clamp_(0, 1)

                pixel_diffs_other_dims = (reconstructed_images - perturbed_images_other_dims).pow(2)

                masks = torch.zeros_like(reconstructed_images)
                for idx in range(bounding_boxes.shape[0]):
                    x, y, xw, yh = bounding_boxes[idx, k]
                    masks[idx, :, y:yh, x:xw] = 1

                other_dimensions_loss =  (masks * pixel_diffs_other_dims).sum()
                #print(f"other dimensions loss: {other_dimensions_loss/100}")
                loss +=  other_dimensions_loss

        
        return loss