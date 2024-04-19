import os
import sys

from pathlib import Path
import pickle
from argparse import Namespace
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

import sys
sys.path.append(".")

from ext.stylegan3_editing.utils.inference_utils import (
    load_encoder,
    get_average_image,
    run_on_batch,
)
from ext.stylegan3_editing.notebooks.notebook_utils import run_alignment
from ext.stylegan3.dnnlib.util import open_url
import ext.stylegan3.training.networks_stylegan3 as networks_stylegan3
from ext.stylegan3 import legacy


class StyleGANSynthesis(nn.Module):
    """
    Load the pre-trained StyleGAN3 and use only the synthesis network
    """

    def __init__(self, pretrained_model_dir):
        super(StyleGANSynthesis, self).__init__()
        with open_url(pretrained_model_dir) as f:
            G_orig = legacy.load_network_pkl(f)["G_ema"]  # type: ignore
            

        self.G = networks_stylegan3.Generator(**G_orig.init_kwargs)
        self.G.load_state_dict(G_orig.state_dict())
        #with open(pretrained_model_dir, "rb") as f:
         #   self.G = pickle.load(f)["G_ema"]

        for param in self.G.parameters():
            param.requires_grad = False

    def forward(self, w):
        output = self.G.synthesis(w, noise_mode="const", force_fp32=True)
        return output

    def get_s(self, w):
        return self.G.affine(w)


class StyleGANEncoder(nn.Module):
    """
    Load the pre-trained StyleGAN3 Encoder.
    """

    def __init__(self, pretrained_model_dir, n_iter_batch=1):
        super(StyleGANEncoder, self).__init__()
        self.net, self.opts = load_encoder(checkpoint_path=pretrained_model_dir)
        self.avg_image = get_average_image(self.net)
        self.opts.n_iters_per_batch = n_iter_batch

        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, img):
        _, result_latents = run_on_batch(
            inputs=img,
            net=self.net,
            opts=self.opts,
            avg_image=self.avg_image,
            landmarks_transform=None,
        )
        result_latents = torch.tensor(
            list(result_latents.values()), device="cuda" if torch.cuda.is_available() else "cpu"
        )

        if len(result_latents.shape) > 3:
            return torch.squeeze(result_latents,1)
        else:
            return result_latents


class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """
        in_out_dim: input/output dim.
        mid_dim: number of units in a hidden layer.
        hidden: number of hidden layers.
        mask_config: 1 if transform odd units, 0 if transform even units
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config
        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim // 2, mid_dim), 
            nn.ReLU()
            )

        self.mid_block = nn.ModuleList(
            [nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU()) for _ in range(hidden - 1)]
        )

        self.out_block = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x, reverse=False):
        """
        :param x: input tensor
        :param reverse: True in inference mode, False in sampling mode
        :return:
        """
        B, D1, D2 = x.size()
        x = x.reshape((B, (D1 * D2) // 2, 2))

        if self.mask_config:
            on, off = x[:, :, 0].clone(), x[:, :, 1].clone()
        else:
            off, on = x[:, :, 0].clone(), x[:, :, 1].clone()

        off_changed = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_changed = self.mid_block[i](off_changed)

        shift = self.out_block(off_changed)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, D1, D2))


class Scaling(nn.Module):
    def __init__(self, dim):
        """
        Initialize a (log-)scaling layer.
        :param dim: input and output dim.
        """
        super(Scaling, self).__init__()
        if isinstance(dim, list):
            self.scale = nn.Parameter(torch.zeros((1, dim[0], dim[1]), requires_grad=True))
        else:
            self.scale = nn.Parameter(torch.zeros((1, dim), requires_grad=True))

    def forward(self, x, reverse=False):
        """

        :param x: input tensor
        :param reverse:False in sampling mode, True in inference mode.
        :return:
        """
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)

        return x, log_det_J


class NICE(nn.Module):
    def __init__(self, w_plus_dim, coupling=4, mid_dim=1000, hidden=5, mask_config=1):
        """
        :param coupling: number of coupling layers.
        :param z_dim: input and output dim.
        :param mid_dim: number of units in a hidden layers
        :param hidden: number of hidden layers
        :param mask_config:
        """
        super(NICE, self).__init__()

        self.coupling = nn.ModuleList(
            [
                Coupling(w_plus_dim[0] * w_plus_dim[1], mid_dim, hidden, (mask_config + i) % 2)
                for i in range(coupling)
            ]
        )
        self.scaling = Scaling(w_plus_dim)

    def forward(self, w_plus):
        """
        g: Transformation from W+ to W* (inverse of f). This part is trained.
        """
        x, _ = self.scaling(w_plus, reverse=True)  # x shape is [B, D1, D2]
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)  # x shape is [B, D1, D2]

        return x

    def f(self, w_hat):
        """
        Transformation from W* to W+. This is used during the sampling mode. It is not used in training in the LIA paper.
        In the original NICE, it is optimized during the encoder training with the detached coupling and scaling.
        :return:
        """
        for i in range(len(self.coupling)):
            w_hat = self.coupling[i](w_hat)

        return self.scaling(w_hat)


class DisGAN(nn.Module):
    def __init__(
        self, w_plus_dim, pretrained_encoder_dir, coupling=4, mid_dim=1000, hidden=5, mask_config=1
    ):
        super(DisGAN, self).__init__()
        self.encoder = StyleGANEncoder(pretrained_encoder_dir)

        # Freeze the encoder and train only the bijective transformation
        for param in self.encoder.parameters():
            param.requires_grad = False

        if not isinstance(w_plus_dim, list):
            w_plus_dim = list(w_plus_dim)

        self.nice = NICE(w_plus_dim, coupling, mid_dim, hidden, mask_config)

    def forward(self, x):
        w_plus = self.encoder(x)
        w_hat = self.nice(w_plus)

        #if len(w_plus.shape) > 3:
         #   w_plus = torch.squeeze(w_plus, dim=1)
        #if len(w_hat.shape) > 3:
         #   w_hat = torch.squeeze(w_hat, dim=1)
        return w_plus, w_hat

    def inverse_T(self, w_hat):
        return self.nice.f(w_hat)[0]



class ID_Discriminator_firsthalf(nn.Module):
    def __init__(self, input_dim): # a tensor of dim [1, 4056+4056] where 4056 is the first half of eaach tensor
        super(ID_Discriminator_firsthalf, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


class ID_Discriminator_secondhalf(nn.Module):
    def __init__(self, input_dim): # a tensor of dim [1, 4056+4056] where 4056 is the second half of eaach tensor
        super(ID_Discriminator_secondhalf, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)
    