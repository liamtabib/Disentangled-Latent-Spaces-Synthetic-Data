import os
from datetime import datetime
import time
import tqdm
import hydra
from hydra.utils import instantiate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import src.models as models
from src.data.dataset import NPairsCelebAHQDataset
import projects.disentanglement.src.utils as utils
import torch.optim as optim
from torchvision import transforms


class DiscriminatorLoss(nn.Module):

    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, discriminator_firsthalf_preds, discriminator_secondhalf_preds, targets, T_turn):

        loss_firsthalf = self.loss_fn(discriminator_firsthalf_preds, targets)
        
        if T_turn:
            targets = 1 - targets

        loss_secondhalf = self.loss_fn(discriminator_secondhalf_preds, targets)
                
        return loss_firsthalf + loss_secondhalf

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HYDRA_FULL_ERROR"] = "1"

def train(
    model,
    data_loader,
    optimizer_discriminators,
    perturbation_loss_fn,
    ID_discriminator_FirstHalf,
    ID_discriminator_SecondHalf,
    device
):
    utils.freeze_model(model)
    # Initialize SPS calculation variables
    total_loss = 0  # Initialize total loss for the epoch

    ID_discriminator_FirstHalf.train()
    ID_discriminator_SecondHalf.train()

    total_samples_processed = 0
    start_time = time.time()
    progress_bar = tqdm.tqdm(data_loader, desc="Training (SPS: 0.00)", leave=False, unit="batch", total=len(data_loader))


    for batch in progress_bar:
    
        imgs, identities = batch
        img1, img2 = imgs[0].to(torch.float).to(device), imgs[1].to(torch.float).to(device)
        identities = identities.to(device)
        
        _, anchor = model(img1)  # Assuming model returns latents and reconstruction
        _, positive = model(img2)

        batch_size = identities.size(0)

        negative = torch.cat((positive[-1].unsqueeze(0), positive[:-1]), dim=0)

        anchor = anchor.view(anchor.size(0), -1)
        positive = positive.view(positive.size(0), -1)
        negative = negative.view(negative.size(0), -1)


        half_latent_space_size = anchor.size(1) // 2

        # Ground truth is given by 
        positive_pair_first_half = torch.cat([anchor[:, :half_latent_space_size],positive[:, :half_latent_space_size]], dim=1)
        negative_pair_first_half = torch.cat([anchor[:, :half_latent_space_size],negative[:, :half_latent_space_size]], dim=1)
        combined_pair_first_half = torch.cat([positive_pair_first_half, negative_pair_first_half], dim=0)

        positive_pair_second_half = torch.cat([anchor[:, half_latent_space_size:],positive[:, half_latent_space_size:]], dim=1)
        negative_pair_second_half = torch.cat([anchor[:, half_latent_space_size:],negative[:, half_latent_space_size:]], dim=1)
        combined_pair_second_half = torch.cat([positive_pair_second_half, negative_pair_second_half], dim=0)


        positive_targets = torch.ones(batch_size, dtype=torch.float32, device=device).unsqueeze(1)
        negative_targets = torch.zeros(batch_size, dtype=torch.float32, device=device).unsqueeze(1)
        combined_targets = torch.cat([positive_targets, negative_targets], dim=0)


        preds_firsthalf = ID_discriminator_FirstHalf(combined_pair_first_half)
        preds_secondhalf = ID_discriminator_SecondHalf(combined_pair_second_half)


        optimizer_discriminators.zero_grad()

        discriminator_losses = perturbation_loss_fn(preds_firsthalf, preds_secondhalf, combined_targets, T_turn=False)
        
        discriminator_losses.backward()
        optimizer_discriminators.step()

        total_samples_processed += batch_size
        elapsed_time = time.time() - start_time
        sps = total_samples_processed / elapsed_time if elapsed_time > 0 else 0

        # Dynamically update tqdm progress bar description
        progress_bar.set_description(f"Training (SPS: {sps:.2f})")
        total_loss += discriminator_losses.item()  # Aggregate loss

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch Loss: {avg_loss:.4f}")  # Print average loss after each epoch
    return 



@hydra.main(version_base="1.1", config_path="../config", config_name="config_disent")
def main(cfg):

    transform = transforms.Compose([
    transforms.ToPILImage(),  # Add this if your images are not PIL Images
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
    
    model = instantiate(cfg.model, _convert_="partial")
    
    model = model.to(device)


    output_dir = os.path.join(cfg.work_dir, "output", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)


    ID_discriminator_FirstHalf = models.ID_Discriminator_firsthalf(input_dim=8192).to(device)
    ID_discriminator_SecondHalf = models.ID_Discriminator_firsthalf(input_dim=8192).to(device)

    PerturbationLoss_fn = DiscriminatorLoss()

    optimizer_discriminators = optim.Adam(list(ID_discriminator_FirstHalf.parameters()) + list(ID_discriminator_SecondHalf.parameters()), lr=0.001)


    if torch.cuda.device_count() > 1:
        # this will use all GPUs available
        ID_discriminator_FirstHalf = nn.DataParallel(ID_discriminator_FirstHalf)
        ID_discriminator_SecondHalf = nn.DataParallel(ID_discriminator_SecondHalf)
        
    dataset_fn = NPairsCelebAHQDataset
    full_range_idx = list(range(30000))

    train_dataset = dataset_fn(
       work_dir= cfg.work_dir,split_idx= full_range_idx, sample_pairs=cfg.training.sample_pairs, transform=transform
    )


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.data_loaders.train.batch_size,
        num_workers=cfg.training.data_loaders.train.num_workers,
        pin_memory=True,
        shuffle=cfg.training.data_loaders.train.shuffle
    )


    for epoch in range(1):

        train(
            model,
            train_dataloader,
            optimizer_discriminators,
            PerturbationLoss_fn,
            ID_discriminator_FirstHalf,
            ID_discriminator_SecondHalf,
            device,
        )
    
    
    # Saving state dictionaries of discriminators
    torch.save(ID_discriminator_FirstHalf.state_dict(), os.path.join(output_dir, f"ID_discriminator_FirstHalf.pt"))
    torch.save(ID_discriminator_SecondHalf.state_dict(), os.path.join(output_dir, f"ID_discriminator_SecondHalf.pt"))



if __name__ == "__main__":
    main()
