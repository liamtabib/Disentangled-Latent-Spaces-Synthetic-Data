import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    """Code is from https://github.com/deepinsight/insightface"""

    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        """
        Initializes a PolyScheduler instance.

        Args:
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.
            base_lr (float): The base learning rate.
            max_steps (int): The total number of steps in the training process.
            warmup_steps (int): The number of warm-up steps to gradually increase the learning rate.
            last_epoch (int, optional): The index of the last epoch. Default is -1.

        """

        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        self.current_lr = self.base_lr * alpha
        return [self.current_lr for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            self.current_lr = self.warmup_lr_init
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            self.current_lr = self.base_lr * alpha
            return [self.current_lr for _ in self.optimizer.param_groups]


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class EarlyStop:
    """
    Class for implementing early stopping during training based on a validation loss threshold.

    Args:
        tot_epochs (int): Total number of epochs for training.
        early_stop_cfg (object): Configuration object for early stopping parameters.

    Attributes:
        counter (int): Counter to keep track of the number of epochs with no improvement in validation loss.
        tot_epochs (int): Total number of epochs for training.
        patience (int): Number of epochs to wait for improvement in validation loss before stopping.
        enabled (bool): Flag indicating whether early stopping is enabled or not.
        min_epoch (int): Minimum epoch number to start checking for early stopping.

    Methods:
        check_early_stop_condition(val_loss, epoch):
            Checks if the early stopping condition is met based on the validation loss and current epoch number.

    """

    def __init__(self, tot_epochs, early_stop_cfg):
        """
        Initializes the EarlyStop object.

        Args:
            tot_epochs (int): Total number of epochs for training.
            early_stop_cfg (object): Configuration object for early stopping parameters.

        """
        self.counter = 0
        self.tot_epochs = tot_epochs
        self.patience = early_stop_cfg.patience
        self.enabled = early_stop_cfg.enabled
        self.min_epoch = early_stop_cfg.min_epoch

    def check_early_stop_condition(self, val_loss, epoch):
        """
        Checks if the early stopping condition is met based on the validation loss and current epoch number.

        Args:
            val_loss (float): Validation loss value.
            epoch (int): Current epoch number.

        Returns:
            bool: True if the early stopping condition is met, False otherwise.

        """
        if self.enabled and (epoch > self.min_epoch):
            self.counter += 1
            if self.counter >= self.patience or val_loss is None:
                print(f"Early stopping at epoch {epoch} with val_loss {val_loss:.4f}")
                return True
        return False


class BatchSizeScheduler:
    def __init__(self, tot_epochs, cfg):
        """
        A class for dynamically adjusting the batch size during training based on a predefined schedule.

        Args:
            tot_epochs (int): Total number of epochs.
            cfg (object): Configuration object containing parameters for batch size scheduling.

        Attributes:
            counter (int): Counter for tracking the number of epochs.
            tot_epochs (int): Total number of epochs.
            patience (int): Number of epochs to wait before reducing the batch size.
            enabled (bool): Flag indicating whether batch size scheduling is enabled.
            min_epoch (int): Minimum epoch from which batch size scheduling starts.
            final_batch_size (int): Final batch size to be used after reducing the batch size.
        """
        self.counter = 0
        self.tot_epochs = tot_epochs
        self.patience = cfg.patience
        self.enabled = cfg.enabled
        self.min_epoch = cfg.min_epoch
        self.final_batch_size = cfg.final_batch_size

    def update_batch_size(self, epoch, training_cfg, data_loader_func, train_aug, train_loader):
        """
        Update the batch size based on the current epoch and configuration.

        Args:
            epoch (int): Current epoch number.
            training_cfg (object): Training configuration object.
            data_loader_func (function): Data loader function.
            train_aug (object): Training data augmentation object.
            train_loader (object): Training data loader object.

        Returns:
            object: Updated training data loader object.
        """
        if epoch > self.min_epoch and self.enabled:
            self.counter += 1
            if self.counter >= self.patience:
                assert training_cfg.batch_size > self.final_batch_size
                training_cfg.batch_size = self.final_batch_size
                print(f"Update of the batch size to {training_cfg.batch_size}")

                train_loader, _ = data_loader_func(
                    dl_cfg=training_cfg,
                    split_type="train",
                    augmentation=train_aug,
                )
                self.counter = 0
        return train_loader
