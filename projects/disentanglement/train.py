##############################################################################################
# HYPERPARAMETERS TO TUNE
#------------------------------------------#
#BS=2 (test=8), n=10.000
switch_loss_on = False  # Enable or disable switch loss.
lambda_switch = 1  # Weighting factor for switch loss.

#run1: before change:  lr = 0.000001. e=10. Problem: learning too fast: artifacts, and has more room for learning
#run2: same as run1 after changing to probability
#run3: same but half the learning rate 0.0000005
#next run: cut learning rate to 0.0000001 and atleast 30 epochs (3 days)
#------------------------------------------#
# BS= 4 (test=8)
# experiment with ratio_inside_outside
landmark_loss_on = False  # Enable or disable landmark loss.
lambda_landmark = 1  # Weighting factor for landmark loss.
ratio_inside_outside = 2
#------------------------------------------#
# BS = 16
#run1: triplet False, lr = 0.0001. training a bit too fast perhaps. decrease learning rate and train for longer time
#run2: same as run1 but with triplet loss. problem: strong collapse
#run3: decreased learning rate for triplet. convergance very slow and weird collapse.
# Conclusion: n_pairs much superior
# run4: try n_pairs 0.00001

#experiment  with  Triplet/n_pairs
contrastive_loss_on = True  # Enable or disable contrastive loss.
triplet = False  # Use triplet formulation for contrastive loss, otherwise n_pairs
lambda_contrastive = 1  # Weighting factor for contrastive loss.
#------------------------------------------#
# BS = 16
#experiment with pretrained, always using both networks

discriminator_loss_on = False  # Enable or disable discriminator loss.
only_second_half_ID_D = False  # Use only the ID Discriminator to drive out ID information from the second half of the latent space.
pretained_ID_D = False  # Toggle the use of a pretrained ID Discriminator.
lambda_discriminator = 1  # Weighting factor for discriminator loss.
#-----------------------------------------#

# General training hyperparameters.
train_batch_size = 16
test_batch_size = 16
lr = 0.003  # Learning rate for the NICE network. LR scales linearly with the number of training images?
epochs = 30  # Number of training epochs.
num_training_images = 100  # Choose either 30 000 for full training or a subset for experimentation.
num_encodings = 100  # Number of image to include in the metrics for FR distance, w_star distance and DCI.

##############################################################################################

# Standard Library Imports
import os
import time
from datetime import datetime

# Third Party Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from facenet_pytorch import MTCNN, InceptionResnetV1 #For switch loss and FR metric
from sklearn.model_selection import train_test_split

# Local Application/Library Specific Imports
import src.models as models
from src.data.dataset import NPairsCelebAHQDataset
import projects.disentanglement.src.utils as utils
from projects.disentanglement.src.losses import SwitchLoss, ContrastiveLoss, DiscriminatorLoss,LandmarkDetector,LandmarkLoss

# Import functions for dataset encoding, generating the grid, and calculating distances in FR space and latent space.
from projects.disentanglement.src.metrics.running_metrics import (
    encode_dataset, 
    mix_identity, 
    mix_landmarks,
    FR_latent_space_distance, 
    ratio_identity_part,
    DCI,
    pca_with_perturbation
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HYDRA_FULL_ERROR"] = "1"


def train(
    generator,
    model,
    data_loader,
    optimizer_disgan,
    optimizer_discriminators,
    contrastive_loss_fn,
    discriminator_loss_fn,
    switch_loss_fn,
    landmark_loss_fn,
    ID_discriminator_FirstHalf,
    ID_discriminator_SecondHalf,
    lambda_contrastive,
    lambda_discriminator,
    lambda_switch,
    lambda_landmark,
    device,
):
    """
    Trains a model using batches from a dataloader with specified loss functions and optimizers.

    This function handles the training of both a generator model and discriminator models
    using provided loss functions for contrastive, discriminator, and switch mechanisms. 
    It operates on batches provided by a dataloader, applying the necessary transformations,
    performing forward passes, computing losses, and updating model parameters.

    Parameters:
        generator (torch.nn.Module): The generator model for image synthesis.
        model (torch.nn.Module): The main model being trained.
        data_loader (torch.utils.data.DataLoader): The dataloader providing training batches.
        optimizer_disgan (torch.optim.Optimizer): Optimizer for the main model.
        optimizer_discriminators (torch.optim.Optimizer): Optimizer for the discriminator models.
        contrastive_loss_fn (function): Function to compute the contrastive loss.
        discriminator_loss_fn (function): Function to compute the discriminator loss.
        switch_loss_fn (function): Function to compute the switch loss.
        ID_discriminator_FirstHalf (torch.nn.Module): The discriminator for the first half of ID features.
        ID_discriminator_SecondHalf (torch.nn.Module): The discriminator for the second half of ID features.
        lambda_contrastive (float): Weight multiplier for the contrastive loss.
        lambda_discriminator (float): Weight multiplier for the discriminator loss.
        lambda_switch (float): Weight multiplier for the switch loss.
        device (torch.device): The device tensors will be sent to during training.

    Returns:
        dict: A dictionary containing average losses for each loss component.
        dict: A dictionary containing gradients for key operations for analysis.
    """
    model.train()  # Set the model to training mode.
    ID_discriminator_FirstHalf.train()  # Set the first half discriminator to training mode if used.
    ID_discriminator_SecondHalf.train()  # Set the second half discriminator to training mode.


    # Initialize variables to store losses and gradients.
    losses = {"final_loss":0,"contrastive_loss": 0,"discriminator_loss":0,"ID_D_loss":0,"switch_loss":0,"landmark_loss":0}
    grads = {"contrastive_loss": [],"discriminator_loss": [],"switch_loss": [],"landmark_loss":[]}

    # Begin a progress bar for training iterations.
    total_samples_processed = 0
    start_time = time.time()
    progress_bar = tqdm.tqdm(data_loader, desc="Training (SPS: 0.00)", leave=False, unit="batch", total=len(data_loader))

    for i, batch in enumerate(progress_bar):
        # Unpack the batch data.
        imgs, identities = batch
        img1, img2 = imgs[0].to(torch.float).to(device), imgs[1].to(torch.float).to(device)
        
        _, anchor = model(img1)  # Process the first image through the encoder and NICE model to get w_plus and w_star latents.
        _, positive = model(img2)  # Process the second image.

        batch_size = identities.size(0)
        # Create a negative sample by shifting the positive samples.
        negative = torch.cat((positive[-1].unsqueeze(0), positive[:-1]), dim=0)

        anchor = anchor.view(anchor.size(0), -1)
        positive = positive.view(positive.size(0), -1)
        negative = negative.view(negative.size(0), -1)

        half_latent_space_size = anchor.size(1) // 2 # Calculate the size of half the latent space.
        # Compute losses if their respective functions are enabled.
        if contrastive_loss_fn is not None:
            # Check if using triplet loss or not.
            if triplet:
                contrastive_loss_output = contrastive_loss_fn(anchor, positive, negative = negative)
            else:
                identities = identities.to(device)
                identities = identities.view(batch_size, 1)
                target = (identities == identities.transpose(0, 1)).float()
                contrastive_loss_output = contrastive_loss_fn(anchor, positive, target = target)


        else: contrastive_loss_output =  torch.tensor(0.0, device=device, requires_grad=True) # Set a default tensor if loss function is disabled.

        if discriminator_loss_fn is not None:
            # Compute the discriminator loss if enabled, handling either half or both halves of the latent space.

            if only_second_half_ID_D:
                preds_firsthalf = None  # If only using second half, set first half predictions to None.
            else:
                # Combine the first half of two latents for the ID-discriminator of the first half.
                positive_pair_first_half = torch.cat([anchor[:, :half_latent_space_size],positive[:, :half_latent_space_size]], dim=1)
                negative_pair_first_half = torch.cat([anchor[:, :half_latent_space_size],negative[:, :half_latent_space_size]], dim=1)
                combined_pair_first_half = torch.cat([positive_pair_first_half, negative_pair_first_half], dim=0)
                # Make prediction of wether they are same ID
                preds_firsthalf = ID_discriminator_FirstHalf(combined_pair_first_half)

            # Prepare input for the ID discriminator of the second half
            positive_pair_second_half = torch.cat([anchor[:, half_latent_space_size:],positive[:, half_latent_space_size:]], dim=1)
            negative_pair_second_half = torch.cat([anchor[:, half_latent_space_size:],negative[:, half_latent_space_size:]], dim=1)
            combined_pair_second_half = torch.cat([positive_pair_second_half, negative_pair_second_half], dim=0)
            # Get prediction of whether they are same ID
            preds_secondhalf = ID_discriminator_SecondHalf(combined_pair_second_half)

            # Prepare ground truth if they are of same ID or not
            positive_targets = torch.ones(batch_size, dtype=torch.float32, device=device).unsqueeze(1)
            negative_targets = torch.zeros(batch_size, dtype=torch.float32, device=device).unsqueeze(1)
            combined_targets = torch.cat([positive_targets, negative_targets], dim=0)

            # reset gradients for discriminator optimizers.
            optimizer_discriminators.zero_grad()
            # Compute the ID -D loss for the current batch.
            ID_D_loss = discriminator_loss_fn(preds_firsthalf, preds_secondhalf, targets=combined_targets, T_turn=False)
            ID_D_loss.backward(retain_graph=True) # Backpropagate the loss while keeping the computational graph
            optimizer_discriminators.step()# Update discriminator weights.

            # Re-compute discriminator predictions for the NICE network update.
            if not only_second_half_ID_D:
                preds_firsthalf_2 = ID_discriminator_FirstHalf(combined_pair_first_half)
            else: preds_firsthalf_2 = None
            preds_secondhalf_2 = ID_discriminator_SecondHalf(combined_pair_second_half)

            discriminator_loss_output = discriminator_loss_fn(preds_firsthalf_2, preds_secondhalf_2, combined_targets, T_turn=True)

        else: 
            ID_D_loss = torch.tensor(0.0, device=device, requires_grad=True)
            discriminator_loss_output = torch.tensor(0.0, device=device, requires_grad=True)
 

        if switch_loss_fn is not None:
            # Handle DataParallel wrapper by accessing the underlying module.
            if isinstance(model, torch.nn.DataParallel):
                model = model.module
            # Adjust batch processing based on size to ensure pairs are correctly formed.
            if batch_size % 2 == 0:
                w_star_i = anchor[:2]
                w_star_j = negative[:2]
            else:
                w_star_i = anchor[:1]
                w_star_j = negative[:1]

            # Randomly determine if the mixed image is positive or negative to the anchor.
            random_num = torch.bernoulli(torch.tensor([0.5]))
            same_ID = True if random_num.item() == 1 else False
            # Depending on the ID match status, concatenate latents appropriately.
            if same_ID:
                second_latent = torch.cat([
                    w_star_i[:, :half_latent_space_size],
                    w_star_j[:, half_latent_space_size:]
                ], dim=1).view(-1, 16, 512)

            else:
                second_latent = torch.cat([
                    w_star_j[:, :half_latent_space_size],
                    w_star_i[:, half_latent_space_size:]
                ], dim=1).view(-1, 16, 512)

            # Process the first latent through the inverse NICE function of the model.
            first_latent = model.inverse_T(w_star_i.view(-1, 16, 512))
            first_image = generator(first_latent).mul_(0.5).add_(0.5).clamp_(0, 1)
            first_image = F.interpolate(first_image, size=(160, 160), mode='bilinear', align_corners=False) # The FR is trained on this resolution
                
            # Process the second latent similarly.
            second_latent = model.inverse_T(second_latent)
            second_image = generator(second_latent).mul_(0.5).add_(0.5).clamp_(0, 1)
            second_image = F.interpolate(second_image, size=(160, 160), mode='bilinear', align_corners=False)
            # Compute the switch loss for the pair of images.
            switch_loss_output = switch_loss_fn(first_image, second_image, same_ID)
            # Clear any cached data to free up GPU memory.
            torch.cuda.empty_cache()


        else: switch_loss_output = torch.tensor(0.0, device=device, requires_grad=True)

        if landmark_loss_fn is not None:
            landmark_loss_output = landmark_loss_fn(generator, model, anchor)

        else: landmark_loss_output = torch.tensor(0.0, device=device, requires_grad=True)


        # Calculate the final loss for the batch by combining all active loss components with their respective weights.
        final_loss = (
                  lambda_contrastive * contrastive_loss_output + 
                  lambda_discriminator * discriminator_loss_output +
                  lambda_switch * switch_loss_output +
                  lambda_landmark * landmark_loss_output
        )
        # Compute gradients for all trainable parameters in the model.
        grads_batch = utils.compute_norm_gradients(
            [contrastive_loss_output, discriminator_loss_output, switch_loss_output, landmark_loss_output], model
        )
        # Zero out gradients for the main model optimizer to prevent accumulation from previous iterations.
        optimizer_disgan.zero_grad(set_to_none=True)
        # Backpropagate the final loss on the NICE network.
        final_loss.backward(retain_graph=True)
        # Update weights for the NICE model.
        optimizer_disgan.step()

        # Store the losses for this batch for later aggregation.
        losses["final_loss"] += final_loss.item()
        losses["contrastive_loss"] += contrastive_loss_output.item()
        losses["discriminator_loss"] += discriminator_loss_output.item()
        losses["ID_D_loss"] += ID_D_loss.item()
        losses["switch_loss"] += switch_loss_output.item()
        losses["landmark_loss"] += landmark_loss_output.item()
        # Store the gradients from this batch for later analysis.
        grads["contrastive_loss"].append(grads_batch[0])
        grads["discriminator_loss"].append(grads_batch[1])
        grads["switch_loss"].append(grads_batch[2])
        grads["landmark_loss"].append(grads_batch[3])

        # Clean up by deleting heavy variables.
        del final_loss
        # Free up memory by clearing cache on the GPU.
        torch.cuda.empty_cache()

        total_samples_processed += batch_size
        elapsed_time = time.time() - start_time
        sps = total_samples_processed / elapsed_time if elapsed_time > 0 else 0

        progress_bar.set_description(f"Training (SPS: {sps:.2f})")

    # Calculate average losses across all batches.
    average_losses = {key: losses[key] / (i + 1) for key in losses}

    return average_losses, grads


################################

# Define a validation function to evaluate the model on a separate subset.
def validate(
    generator,
    model,
    data_loader,
    contrastive_loss_fn,
    discriminator_loss_fn,
    switch_loss_fn,
    landmark_loss_fn,
    ID_discriminator_FirstHalf,
    ID_discriminator_SecondHalf,
    lambda_contrastive,
    lambda_discriminator,
    lambda_switch,
    lambda_landmark,
    device,
):
    """
    Validates the model using the provided validation dataloader and calculates loss for each batch.

    This function evaluates the model without making any updates to the model's parameters. It computes
    various losses to assess the performance of the model under the current training state. The function 
    handles different loss computations conditionally based on the provided loss functions and supports 
    both single and multiple discriminators.
    """
    model.eval()  # Set the model to training mode.
    ID_discriminator_FirstHalf.eval()  # Set the first half discriminator to training mode if used.
    ID_discriminator_SecondHalf.eval()  # Set the second half discriminator to training mode.

    # Initialize variables to store losses and gradients.
    losses = {"final_loss":0,"contrastive_loss": 0,"discriminator_loss":0,"ID_D_loss":0,"switch_loss":0,"landmark_loss":0}

    # Begin a progress bar for training iterations.
    total_samples_processed = 0
    start_time = time.time()
    progress_bar = tqdm.tqdm(data_loader, desc="Training (SPS: 0.00)", leave=False, unit="batch", total=len(data_loader))

    for i, batch in enumerate(progress_bar):
        with torch.no_grad():
            # Unpack the batch data.
            imgs, identities = batch
            img1, img2 = imgs[0].to(torch.float).to(device), imgs[1].to(torch.float).to(device)
            
            _, anchor = model(img1)  # Process the first image through the encoder and NICE model to get w_plus and w_star latents.
            _, positive = model(img2)  # Process the second image.

            batch_size = identities.size(0)
            # Create a negative sample by shifting the positive samples.
            negative = torch.cat((positive[-1].unsqueeze(0), positive[:-1]), dim=0)

            anchor = anchor.view(anchor.size(0), -1)
            positive = positive.view(positive.size(0), -1)
            negative = negative.view(negative.size(0), -1)

            half_latent_space_size = anchor.size(1) // 2 # Calculate the size of half the latent space.
            # Compute losses if their respective functions are enabled.
            if contrastive_loss_fn is not None:
                # Check if using triplet loss or not.
                if triplet:
                    contrastive_loss_output = contrastive_loss_fn(anchor, positive, negative = negative)
                else:
                    identities = identities.to(device)
                    identities = identities.view(batch_size, 1)
                    target = (identities == identities.transpose(0, 1)).float()
                    contrastive_loss_output = contrastive_loss_fn(anchor, positive, target = target)


            else: contrastive_loss_output =  torch.tensor(0.0, device=device, requires_grad=True) # Set a default tensor if loss function is disabled.

            if discriminator_loss_fn is not None:
                # Compute the discriminator loss if enabled, handling either half or both halves of the latent space.

                if only_second_half_ID_D:
                    preds_firsthalf = None  # If only using second half, set first half predictions to None.
                else:
                    # Combine the first half of two latents for the ID-discriminator of the first half.
                    positive_pair_first_half = torch.cat([anchor[:, :half_latent_space_size],positive[:, :half_latent_space_size]], dim=1)
                    negative_pair_first_half = torch.cat([anchor[:, :half_latent_space_size],negative[:, :half_latent_space_size]], dim=1)
                    combined_pair_first_half = torch.cat([positive_pair_first_half, negative_pair_first_half], dim=0)
                    # Make prediction of wether they are same ID
                    preds_firsthalf = ID_discriminator_FirstHalf(combined_pair_first_half)

                # Prepare input for the ID discriminator of the second half
                positive_pair_second_half = torch.cat([anchor[:, half_latent_space_size:],positive[:, half_latent_space_size:]], dim=1)
                negative_pair_second_half = torch.cat([anchor[:, half_latent_space_size:],negative[:, half_latent_space_size:]], dim=1)
                combined_pair_second_half = torch.cat([positive_pair_second_half, negative_pair_second_half], dim=0)
                # Get prediction of whether they are same ID
                preds_secondhalf = ID_discriminator_SecondHalf(combined_pair_second_half)

                # Prepare ground truth if they are of same ID or not
                positive_targets = torch.ones(batch_size, dtype=torch.float32, device=device).unsqueeze(1)
                negative_targets = torch.zeros(batch_size, dtype=torch.float32, device=device).unsqueeze(1)
                combined_targets = torch.cat([positive_targets, negative_targets], dim=0)

                # Compute the ID -D loss for the current batch.
                ID_D_loss = discriminator_loss_fn(preds_firsthalf, preds_secondhalf, targets=combined_targets, T_turn=False)

                # Re-compute discriminator predictions for the NICE network update.
                if not only_second_half_ID_D:
                    preds_firsthalf_2 = ID_discriminator_FirstHalf(combined_pair_first_half)
                else: preds_firsthalf_2 = None
                preds_secondhalf_2 = ID_discriminator_SecondHalf(combined_pair_second_half)

                discriminator_loss_output = discriminator_loss_fn(preds_firsthalf_2, preds_secondhalf_2, combined_targets, T_turn=True)

            else: 
                ID_D_loss = torch.tensor(0.0, device=device, requires_grad=True)
                discriminator_loss_output = torch.tensor(0.0, device=device, requires_grad=True)
    

            if switch_loss_fn is not None:
                # Handle DataParallel wrapper by accessing the underlying module.
                if isinstance(model, torch.nn.DataParallel):
                    model = model.module
                # Adjust batch processing based on size to ensure pairs are correctly formed.
                if batch_size % 2 == 0:
                    w_star_i = anchor[:2]
                    w_star_j = negative[:2]
                else:
                    w_star_i = anchor[:1]
                    w_star_j = negative[:1]

                # Randomly determine if the mixed image is positive or negative to the anchor.
                random_num = torch.bernoulli(torch.tensor([0.5]))
                same_ID = True if random_num.item() == 1 else False
                # Depending on the ID match status, concatenate latents appropriately.
                if same_ID:
                    second_latent = torch.cat([
                        w_star_i[:, :half_latent_space_size],
                        w_star_j[:, half_latent_space_size:]
                    ], dim=1).view(-1, 16, 512)

                else:
                    second_latent = torch.cat([
                        w_star_j[:, :half_latent_space_size],
                        w_star_i[:, half_latent_space_size:]
                    ], dim=1).view(-1, 16, 512)

                # Process the first latent through the inverse NICE function of the model.
                first_latent = model.inverse_T(w_star_i.view(-1, 16, 512))
                first_image = generator(first_latent).mul_(0.5).add_(0.5).clamp_(0, 1)
                first_image = F.interpolate(first_image, size=(160, 160), mode='bilinear', align_corners=False) # The FR is trained on this resolution
                    
                # Process the second latent similarly.
                second_latent = model.inverse_T(second_latent)
                second_image = generator(second_latent).mul_(0.5).add_(0.5).clamp_(0, 1)
                second_image = F.interpolate(second_image, size=(160, 160), mode='bilinear', align_corners=False)
                # Compute the switch loss for the pair of images.
                switch_loss_output = switch_loss_fn(first_image, second_image, same_ID)


            else: switch_loss_output = torch.tensor(0.0, device=device, requires_grad=True)

            if landmark_loss_fn is not None:
                landmark_loss_output = landmark_loss_fn(generator, model, anchor)

            else: landmark_loss_output = torch.tensor(0.0, device=device, requires_grad=True)


            # Calculate the final loss for the batch by combining all active loss components with their respective weights.
            final_loss = (
                    lambda_contrastive * contrastive_loss_output + 
                    lambda_discriminator * discriminator_loss_output +
                    lambda_switch * switch_loss_output +
                    lambda_landmark * landmark_loss_output
            )

            # Store the losses for this batch for later aggregation.
            losses["final_loss"] += final_loss.item()
            losses["contrastive_loss"] += contrastive_loss_output.item()
            losses["discriminator_loss"] += discriminator_loss_output.item()
            losses["ID_D_loss"] += ID_D_loss.item()
            losses["switch_loss"] += switch_loss_output.item()
            losses["landmark_loss"] += landmark_loss_output.item()

            total_samples_processed += batch_size
            elapsed_time = time.time() - start_time
            sps = total_samples_processed / elapsed_time if elapsed_time > 0 else 0

            progress_bar.set_description(f"Testing (SPS: {sps:.2f})")

    # Calculate average losses across all batches.
    average_losses = {key: losses[key] / (i + 1) for key in losses}

    return average_losses








@hydra.main(version_base="1.1", config_path="../config", config_name="config_disent")
def main(cfg):

    #Start a timer for the complete execution
    start = time.time()

    # Split into train, test and valid, where train and test are run in every epoch
    train_split_idx, test_split_idx = train_test_split(range(num_training_images))
    test_split_idx, val_split_idx = train_test_split(test_split_idx)

    # Define transformation to preprocess images before passing them to the dataloader
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Add this if your images are not PIL Images
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Instantiate the pre-trained StyleGAN generator and the NICE network
    generator = models.StyleGANSynthesis(cfg.generator_pretrained_dir)
    model = instantiate(cfg.model, _convert_="partial")
    model = model.to(device)
    generator = generator.to(device)

    # Freeze the params of StyleGAN
    for param in generator.parameters():
        param.requires_grad = False 

    # Create directory for all related outputs
    output_dir = os.path.join(cfg.work_dir, "output", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    # Create file with the parameters used
    arguments_file_path = os.path.join(output_dir, 'config.txt')
    with open(arguments_file_path, 'w') as file:
        file.write("Training Configuration Parameters:\n")
        file.write("-------------------------------------\n")
        file.write(f"Learning Rate (lr): {lr}\n")
        file.write(f"Train Batch size: {train_batch_size}\n")
        file.write(f"Test Batch size: {test_batch_size}\n")
        file.write(f"Total Training Epochs: {epochs}\n")
        file.write(f"Number of Training Images: {num_training_images}\n")
        file.write(f"Number of Encodings: {num_encodings}\n\n")

        file.write("Loss Functions and Hyperparameters:\n")
        file.write("-------------------------------------\n")
        file.write(f"Contrastive Loss Enabled: {contrastive_loss_on}\n")
        file.write(f"  - Lambda for Contrastive Loss: {lambda_contrastive}\n")
        file.write(f"  - Triplet Mode for Contrastive Loss: {triplet}\n")
        file.write(f"Discriminator Loss Enabled: {discriminator_loss_on}\n")
        file.write(f"  - Lambda for Discriminator Loss: {lambda_discriminator}\n")
        file.write(f"  - Only Second Half ID Discriminator Used: {only_second_half_ID_D}\n")
        file.write(f"  - Pretrained ID Discriminator: {pretained_ID_D}\n")
        file.write(f"Switch Loss Enabled: {switch_loss_on}\n")
        file.write(f"  - Lambda for Switch Loss: {lambda_switch}\n")
        file.write(f"Landmark Loss Enabled: {landmark_loss_on}\n")
        file.write(f"  - Lambda for Landmark Loss: {lambda_landmark}\n")
        file.write(f"  - ratio inside outside: {ratio_inside_outside}\n")

    # Initialize face detection and recognition tools for further training and evaluation.
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # init class to calculate distances in the face recognition latent space to measure distances.
    FR_distance = FR_latent_space_distance(
            generator=generator,
            face_detection=mtcnn, 
            face_recognition=resnet, 
            save_path= os.path.join(output_dir, "FR_distance.txt"), 
        )
    ##################################################################################
    # Initialize loss functions based on the configuration.
    if contrastive_loss_on:
        contrastive_loss_fn = ContrastiveLoss(triplet)
    else:
        contrastive_loss_fn = None

    if switch_loss_on:
        switch_loss_fn = SwitchLoss(face_recognition=resnet)
    else:
        switch_loss_fn = None

    if discriminator_loss_on:
        ID_discriminator_SecondHalf = models.ID_Discriminator_firsthalf(input_dim=8192).to(device)
        discriminatorLoss_fn = DiscriminatorLoss(second=only_second_half_ID_D)

        if only_second_half_ID_D:
            ID_discriminator_FirstHalf = None
            optimizer_discriminators = optim.Adam(ID_discriminator_SecondHalf.parameters(), lr=lr)
        else:
            ID_discriminator_FirstHalf = models.ID_Discriminator_firsthalf(input_dim=8192).to(device)
            optimizer_discriminators = optim.Adam(list(ID_discriminator_FirstHalf.parameters()) + list(ID_discriminator_SecondHalf.parameters()), lr=lr)
    else:
        discriminatorLoss_fn = None
        ID_discriminator_FirstHalf=  None
        ID_discriminator_SecondHalf = None
        optimizer_discriminators = None
    
    if landmark_loss_on:
        predictor_path = "projects/disentanglement/pretrained_models/shape_predictor_68_face_landmarks.dat"
        landmark_model = LandmarkDetector(predictor_path)
        landmark_loss_fn = LandmarkLoss(landmark_model, ratio_inside_outside)
    else: landmark_loss_fn = None
        
    ##################################################################################

    # Utilize multiple GPUs if available.
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        generator = nn.DataParallel(generator)
        ID_discriminator_FirstHalf = nn.DataParallel(ID_discriminator_FirstHalf)
        ID_discriminator_SecondHalf = nn.DataParallel(ID_discriminator_SecondHalf)


    # Load pre-trained weights for ID discriminators if configured to do so.
    if discriminator_loss_on and pretained_ID_D:
        if not only_second_half_ID_D:
            first_half_discriminator_path = 'projects/disentanglement/pretrained_models/discriminators/ID_discriminator_FirstHalf.pt'
            ID_discriminator_FirstHalf.load_state_dict(torch.load(first_half_discriminator_path))
    
        second_half_discriminator_path = 'projects/disentanglement/pretrained_models/discriminators/ID_discriminator_SecondHalf.pt'
        ID_discriminator_SecondHalf.load_state_dict(torch.load(second_half_discriminator_path))

    # Configure the optimizer for the main model's parameters.
    disgan_optimizer = optim.Adam(model.module.nice.parameters(), lr=lr)

    ##################################################################################
    # Initialize datasets for training, testing, and validation phases.
    dataset_fn = NPairsCelebAHQDataset
    train_dataset = dataset_fn(
       work_dir= cfg.work_dir,split_idx= train_split_idx, sample_pairs=cfg.training.sample_pairs, transform=transform
    )
    test_dataset = dataset_fn(
       work_dir= cfg.work_dir, split_idx=test_split_idx, sample_pairs=cfg.training.sample_pairs, transform=transform
    )
    val_dataset = dataset_fn(
        work_dir=cfg.work_dir, split_idx=val_split_idx, sample_pairs=cfg.training.sample_pairs, transform=transform
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=cfg.training.data_loaders.train.shuffle
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=cfg.training.data_loaders.val.shuffle
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=cfg.training.data_loaders.val.shuffle
    )
    print(f"batch size: ")
    print(f"Number of training batches in each epoch: {len(train_dataloader)} with batch size: {train_batch_size}")
    print(f"Number of test batches in each epoch: {len(test_dataloader)} with batch size: {test_batch_size}")
    print(f"Number of validation batches in each epoch: {len(val_dataloader)} with batch size: {test_batch_size}")
    ##################################################################################
    
    # Initialize structures to record losses and gradients for analysis and adjustment.
    train_losses = {"contrastive_loss": [], "discriminator_loss": [], "ID_D_loss": [], "switch_loss": [], "landmark_loss":[]}
    test_losses = {"contrastive_loss": [], "discriminator_loss": [], "ID_D_loss": [], "switch_loss": [], "landmark_loss": []}
    all_grads = {"contrastive_loss": [], "discriminator_loss": [], "switch_loss": [], "landmark_loss":[], "landmark_loss":[]}


    # Main training loop processing each epoch.
    for epoch in range(epochs):

        #METRICS
        ##################################################################################
        # Generate and save a grid of combined images to monitor visual progress.
        saved_grid_filepath = os.path.join(output_dir, f"mixed_identity_epoch_{epoch}.jpg")    
        mix_identity(generator,model,saved_grid_filepath)

        if landmark_loss_on:
            saved_landmark_filepath = os.path.join(output_dir, f"mixed_landmark_epoch_{epoch}.jpg")    
            mix_landmarks(generator,model,saved_landmark_filepath)

        # Encode datasets to analyze latent space distances.
        encoded_images_tensor,identity_ids_tensor = encode_dataset(model,num_encodings)

        saved_pca_filepath = os.path.join(output_dir, f"pca_epoch_{epoch}.jpg")    
        pca_with_perturbation(generator,model,encoded_images_tensor,saved_pca_filepath)

        # distances in latent space
        average_positive_distances_first_half , average_negative_distances_first_half, ratio_distances = ratio_identity_part(encoded_images_tensor,identity_ids_tensor)
        latent_distances_filename = os.path.join(output_dir, "latent_distances_FH.txt")
        # Log latent space distance analysis to file.
        with open(latent_distances_filename, 'a') as file:
            file.write(
                f"Epoch {epoch}: F-H latent distance positives: {average_positive_distances_first_half:.3f}, "
                f"F-H latent distance negatives: {average_negative_distances_first_half:.3f} "
                f"Ratio : {ratio_distances:.3f}\n"
    )
        # Calculate distances in face recognition space to assess quantitatively the quality of the mixing.
        FR_distance(encoded_images_tensor,identity_ids_tensor,model,epoch)
        ##################################################################################
        print(f"[{datetime.now()}] training epoch {epoch}/{epochs}...")
        # Execute one training epoch and retrieve losses and gradient norms.
        train_batch_losses, train_batch_grad = train(
            generator,
            model,
            train_dataloader,
            disgan_optimizer,
            optimizer_discriminators,
            contrastive_loss_fn,
            discriminatorLoss_fn,
            switch_loss_fn,
            landmark_loss_fn,
            ID_discriminator_FirstHalf,
            ID_discriminator_SecondHalf,
            lambda_contrastive,
            lambda_discriminator,
            lambda_switch,
            lambda_landmark,
            device,
        )

        print(f"[{datetime.now()}] testing epoch {epoch}/{epochs}...")
        # Validate the model performance on the test set and retrieve test losses.
        test_batch_losses = validate(
            generator,
            model,
            test_dataloader,
            contrastive_loss_fn,
            discriminatorLoss_fn,
            switch_loss_fn,
            landmark_loss_fn,
            ID_discriminator_FirstHalf,
            ID_discriminator_SecondHalf,
            lambda_contrastive,
            lambda_discriminator,
            lambda_switch,
            lambda_landmark,
            device,
        )

        # Calculate and log the mean loss values for the current epoch.
        contrastive_loss_value = np.mean(train_batch_losses["contrastive_loss"])
        discriminator_loss_value = np.mean(train_batch_losses["discriminator_loss"])
        switch_loss_value = np.mean(train_batch_losses["switch_loss"])
        landmark_loss_value = np.mean(train_batch_losses["landmark_loss"])

        # Calculate and log the ratios of loss components for analysis.
        contrastive_to_discriminator_ratio = contrastive_loss_value / discriminator_loss_value if discriminator_loss_value != 0 else 0
        contrastive_to_switch_ratio = contrastive_loss_value / switch_loss_value if switch_loss_value != 0 else 0
        contrastive_to_landmark_ratio = contrastive_loss_value / landmark_loss_value if landmark_loss_value != 0 else 0

        with open(os.path.join(output_dir, "losses.txt"), 'a') as file:
            file.write(
                f"contrastive_loss_value: {contrastive_loss_value:.3f}, \n "
                f"discriminator_loss_value: {discriminator_loss_value:.3f},\n "
                f"switch_loss_value: {switch_loss_value:.3f}, \n"
                f"landmark_loss_value: {landmark_loss_value:.3f}, \n "
                f"Contrastive to Discriminator Ratio: {contrastive_to_discriminator_ratio:.3f},\n "
                f"Contrastive to Switch Ratio: {contrastive_to_switch_ratio:.3f}\n"
                f"Contrastive to landmark Ratio: {contrastive_to_landmark_ratio:.3f}\n \n"
            )

        # Store gradients and losses for later analysis and adjustment.
        all_grads["contrastive_loss"].append(train_batch_grad["contrastive_loss"])
        train_losses["contrastive_loss"].append(np.mean(train_batch_losses["contrastive_loss"]))
        test_losses["contrastive_loss"].append(np.mean(test_batch_losses["contrastive_loss"]))

        all_grads["discriminator_loss"].append(train_batch_grad["discriminator_loss"])
        train_losses["discriminator_loss"].append(np.mean(train_batch_losses["discriminator_loss"]))
        test_losses["discriminator_loss"].append(np.mean(test_batch_losses["discriminator_loss"]))

        train_losses["ID_D_loss"].append(np.mean(train_batch_losses["ID_D_loss"]))
        test_losses["ID_D_loss"].append(np.mean(test_batch_losses["ID_D_loss"]))

        all_grads["switch_loss"].append(train_batch_grad["switch_loss"])
        train_losses["switch_loss"].append(np.mean(train_batch_losses["switch_loss"]))
        test_losses["switch_loss"].append(np.mean(test_batch_losses["switch_loss"]))

        all_grads["landmark_loss"].append(train_batch_grad["landmark_loss"])
        train_losses["landmark_loss"].append(np.mean(train_batch_losses["landmark_loss"]))
        test_losses["landmark_loss"].append(np.mean(test_batch_losses["landmark_loss"]))

        # Plot and save metrics for visual analysis.

        metric_names = [
        "contrastive_loss" if contrastive_loss_on else "",
        "discriminator_loss" if discriminator_loss_on else "",
        "ID_D_loss" if discriminator_loss_on else "",
        "switch_loss" if switch_loss_on else "",
        "landmark_loss" if landmark_loss_on else ""
        ]
        metric_names = [name for name in metric_names if name]


        for metric_name in metric_names:
            utils.plot_metrics(
                train_losses[metric_name],
                test_losses[metric_name],
                metric_name,
                os.path.join(output_dir, f"{metric_name}.png"),
            )
        utils.plot_grads(all_grads, os.path.join(output_dir, f"grads.png"))
        # Save model weights at the end of the final epoch.
        if epoch == epochs-1:
            utils.save_weights(model, os.path.join(output_dir, f"model_T_{epoch}"))
        # Close all matplotlib plots to free memory.
        plt.close('all')
    # Validate model performance on the validation dataset after training.
    val_batch_loss = validate(
            generator,
            model,
            val_dataloader,
            contrastive_loss_fn,
            discriminatorLoss_fn,
            switch_loss_fn,
            landmark_loss_fn,
            ID_discriminator_FirstHalf,
            ID_discriminator_SecondHalf,
            lambda_contrastive,
            lambda_discriminator,
            lambda_switch,
            lambda_landmark,
            device,
        )
    # Log final validation losses.
    # Define the file path for the log file
    validation_log_path = os.path.join(output_dir, "validation_losses.txt")

    # Log final validation losses to a file with clear headers and descriptions.
    with open(validation_log_path, 'w') as log_file:
        log_file.write("Final Validation Losses\n")
        log_file.write("----------------------\n")
        log_file.write("This section provides the final calculated losses for the validation set at the end of the training process.\n")
        log_file.write("\n")
        log_file.write("Contrastive Loss: {:.3f}\n".format(np.mean(val_batch_loss["contrastive_loss"])))
        log_file.write("Discriminator Loss: {:.3f}\n".format(np.mean(val_batch_loss["discriminator_loss"])))
        log_file.write("Switch Loss: {:.3f}\n".format(np.mean(val_batch_loss["switch_loss"])))
        log_file.write("Landmark Loss: {:.3f}\n".format(np.mean(val_batch_loss["landmark_loss"])))

    # Calculate DCI 

    results_path = os.path.join(output_dir, "result_dci.txt")

    dci = DCI(encoded_images_tensor)
    importance_matrix, train_loss, test_loss = dci.evaluate()
    scores = {
        "informativeness_train": train_loss,
        "informativeness_test": test_loss,
        "disentanglement": DCI.disentanglement(importance_matrix),
        "completeness": DCI.completeness(importance_matrix)
    }

    # Save the scores to a file
    with open(results_path, 'w') as f:
        for key, value in scores.items():
            f.write(f"{key}: {value}\n")

    # Calculate total execution time and log it.
    end = time.time()
    elapsed_time_in_seconds = end - start
    elapsed_time_in_minutes = elapsed_time_in_seconds / 60

    time_info_path = os.path.join(output_dir, "training_times.txt")
    with open(time_info_path, 'w') as file:
        file.write(f"total training time: {str(int(elapsed_time_in_minutes))} min") 


if __name__ == "__main__":
    main()