import argparse
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from src.models import StyleGANEncoder
from src.data.dataset import CelebAHQDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_indices(x, indices_to_exclude):
    mask = np.ones(len(x), dtype=bool)
    mask[indices_to_exclude] = False
    print(mask.sum())

    return x[mask]


def plot_metrics(train, test, plot_name, output_dir):
    plt.figure()
    plt.plot(train, c="red")
    plt.plot(test, c="green")
    plt.legend(["train", "test"])
    plt.xlabel("epoch")
    plt.ylabel(plot_name)

    plt.savefig(output_dir)


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            # nn.LeakyReLU(),
            # nn.Linear(64, 32),
            # nn.LeakyReLU(),
            # nn.Dropout(),
            # nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(64, out_features),
        )
        self.single_mlp = nn.Linear(in_features, out_features)

    def forward(self, w_plus):
        return self.single_mlp(w_plus)


def get_data(images_path, attribute_path, encoder_pretrained_dir, p_threshold=2, batch_size=4):
    with open(os.path.join(os.getcwd(), "w_plus_space.npy"), "rb") as f:
        input_data = pickle.load(f)
    x, attributes = np.squeeze(np.array(input_data[0])), input_data[1]

    attributes = np.array([i.numpy() for i in attributes])
    x = x.reshape(30000, x.shape[1] * x.shape[2])
    # scaler = MinMaxScaler()
    # x = scaler.fit_transform(x)
    return x, attributes


def get_latent_space(data_loader, encoder_dir):
    print("Extracting latent spaces for each image...")
    latent_spaces = []
    attributes = []
    encoder = StyleGANEncoder(encoder_dir)

    for batch in tqdm.tqdm(data_loader):
        images, attribute = batch
        images = images.to(device).to(torch.float)
        output = encoder(images)
        latent_spaces.extend(output.values())
        attributes.extend(attribute)

        with open(os.path.join(os.getcwd(), "w_plus_space1.npy"), "wb") as f:
            pickle.dump((latent_spaces, attributes), f)

    return np.array(latent_spaces)


def train(model, data_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = []
    acc_all = []

    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        preds = torch.round(torch.sigmoid(output))

        acc = accuracy_score(y.cpu(), preds.detach().cpu(), normalize=True)
        acc_all.append(acc)

    return train_loss, np.mean(acc_all)


def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    test_loss = []
    acc_all = []

    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = loss_fn(output, y)
        test_loss.append(loss.item())
        preds = torch.round(torch.sigmoid(output))

        acc = accuracy_score(y.cpu(), preds.detach().cpu(), normalize=True)
        acc_all.append(acc)

    return test_loss, np.mean(acc_all)


def select_random_ind(y, val_size=0.2):
    min_number_samples = np.min(np.unique(y, return_counts=True)[1])
    val_size = int(val_size * min_number_samples)
    val_set_ind = np.concatenate(
        (
            np.random.choice(np.where(y == 1)[0], size=val_size, replace=False),
            np.random.choice(np.where(y == 0)[0], size=val_size, replace=False),
        )
    )

    return val_set_ind


def train_loop(args):
    np.random.seed(args.seed)
    models = []
    input_data, attributes = get_data(
        args.images_dir, args.attribute_dir, args.encoder_pretrained_dir
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(attributes.shape[1])
    for i in range(attributes.shape[1]):
        print(f"Tranining the classifier {i}...")

        x = input_data
        y = attributes[:, i]
        y[y > 0] = 1
        y[y < 0] = 0

        val_set_ind = select_random_ind(y)
        print(len(val_set_ind))

        x_val = x[val_set_ind]
        y_val = y[val_set_ind]

        x = x[~np.isin(np.arange(len(x)), val_set_ind)]
        y = y[~np.isin(np.arange(len(y)), val_set_ind)]
        if args.balanced:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=args.seed)
            train_ind, test_ind = list(sss.split(x, y))[0]
            x_train, y_train, x_test, y_test = x[train_ind], y[train_ind], x[test_ind], y[test_ind]
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=args.seed)

        print("Train class distirbution:", np.unique(y_train, return_counts=True))
        print("Test class distirbution:", np.unique(y_test, return_counts=True))

        if args.classifier_type == "GradientBoosting":
            print("Using GradientBoosting classifier...")

            model = GradientBoostingClassifier(verbose=1)
            model.fit(x_train, y_train)

            train_acc = np.mean(model.predict(x_train) == y_train)
            test_acc = np.mean(model.predict(x_test) == y_test)

            print(
                "Attribute {}/{}, Train acc: {:.2f} Test acc {:.2f}".format(
                    i, attributes.shape[1], train_acc, test_acc
                )
            )

        elif args.classifier_type == "SVM":
            print("Using SVM classifier...")
            model = SVC(gamma="auto", verbose=1)
            model.fit(x_train, y_train)

            train_acc = np.mean(model.predict(x_train) == y_train)
            test_acc = np.mean(model.predict(x_test) == y_test)

            print(
                "Attribute {}/{}, Train acc: {:.2f} Test acc {:.2f}".format(
                    i, attributes.shape[1], train_acc, test_acc
                )
            )

        elif args.classifier_type == "MLP":
            print("Using MLP...")

            model = MLP(x_train.shape[1], 1)
            model = model.to(device)
            train_dataset = TensorDataset(
                torch.tensor(x_train), torch.tensor(y_train[:, None], dtype=torch.float)
            )
            test_dataset = TensorDataset(
                torch.tensor(x_test), torch.tensor(y_test[:, None], dtype=torch.float)
            )
            val_dataset = TensorDataset(
                torch.tensor(x_val), torch.tensor(y_val[:, None], dtype=torch.float)
            )

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            loss_fn = nn.BCEWithLogitsLoss()

            train_loss_epoch, test_loss_epoch = [], []
            train_acc_epoch, test_acc_epoch = [], []

            for epoch in range(args.num_epochs):
                train_loss, train_acc = train(model, train_dataloader, optimizer, loss_fn, device)
                test_loss, test_acc = evaluate(model, test_dataloader, loss_fn, device)
                val_loss, val_acc = evaluate(model, val_dataloader, loss_fn, device)

                train_loss_epoch.append(np.mean(train_loss))
                test_loss_epoch.append(np.mean(test_loss))

                train_acc_epoch.append(train_acc)
                test_acc_epoch.append(test_acc)

                plot_metrics(
                    train_loss_epoch,
                    test_loss_epoch,
                    "loss",
                    os.path.join(args.output_dir, "loss.png"),
                )
                plot_metrics(
                    train_acc_epoch,
                    test_acc_epoch,
                    "loss",
                    os.path.join(args.output_dir, "acc.png"),
                )

                train_loss = np.mean(train_loss)
                test_loss = np.mean(test_loss)
                val_loss = np.mean(val_loss)

                print(
                    "Epoch {}/{} --- Train Loss: {:.3f}, Acc: {:.3f} ".format(
                        epoch, args.num_epochs, train_loss, train_acc
                    )
                )
                print(
                    "Epoch {}/{} --- Test Loss: {:.3f}, Acc: {:.3f} ".format(
                        epoch, args.num_epochs, test_loss, test_acc
                    )
                )

                print(
                    "Epoch {}/{} --- Val Loss: {:.3f}, Val: {:.3f} ".format(
                        epoch, args.num_epochs, val_loss, val_acc
                    )
                )
        models.append(model.state_dict())

    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate the DCI of given latent codes")
    parser.add_argument(
        "--images_dir",
        type=str,
        default=r"C:\Users\Kateryna\Documents\RISE\DIFFUSE\datasets\CelebAMask-HQ\images",
        help="path to latent codes",
    )
    parser.add_argument(
        "--attribute_dir",
        default=r"C:\Users\Kateryna\Documents\RISE\DIFFUSE\disentanglement\datasets\CelebAMask-HQ\CelebAMask-HQ-attribute-anno.txt",
        type=str,
        help="path to attribute",
    )
    parser.add_argument(
        "--encoder_pretrained_dir",
        default="pretrained_models/restyle_pSp_ffhq.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "output", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
    )
    parser.add_argument("--num_epochs", default=100)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--balanced", default=False)
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        "--classifier_type", default="MLP", type=str, choices=["MLP", "GradientBoosting", "SVM"]
    )

    args = parser.parse_args()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # dataset = CelebAHQDataset(range(30000), sample_pairs=False, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    # get_latent_space(dataloader, args.encoder_pretrained_dir)
    models = train_loop(args)

    with open(os.path.join(os.getcwd(), "models.pkl"), "wb") as f:
        pickle.dump(models, f)
