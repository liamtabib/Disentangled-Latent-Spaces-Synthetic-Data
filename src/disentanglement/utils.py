import os
import subprocess

import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from graphviz import Digraph



def download_encoder(file_id, file_name, code_dir="pixel2style2pixel"):
    """
    Get wget download command for downloading the desired model and save to directory ../pretrained_models.
    Code is from here: https://github.com/eladrich/pixel2style2pixel
    """
    current_directory = os.getcwd()
    save_path = os.path.join(os.path.dirname(current_directory), code_dir, "pretrained_models")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
        FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path
    )
    subprocess.run(url, shell=True)


def get_fid(generator, dataloader, batch_real_imgs, device):
    fid = FrechetInceptionDistance(feature=64).to(device)

    for i, batch in tqdm.tqdm(enumerate(dataloader)):
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print(
            f"Run generator loop Current Memory: {current_memory / 1e6} MB, Peak Memory: {peak_memory / 1e6} MB"
        )

        batch = batch.to(device)
        batch_fake_imgs = generator(batch)

        fid.update(batch_fake_imgs.to(torch.uint8).to(device), real=False)
        fid.update(batch_real_imgs[i].to(torch.uint8).to(device), real=True)

    return fid.compute().cpu(), (batch_fake_imgs, batch_real_imgs[i])


def save_weights(model, output_dir):
    torch.save(model, output_dir)


def print_gpu_memory_usage(tag=""):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{tag} - Memory Allocated: {allocated / 1024**3:.3f} GB")
    print(f"{tag} - Memory Reserved: {reserved / 1024**3:.3f} GB")


def compute_norm(params):
    total_norm = 0
    for p in params:
        # Check if the gradient is None
        if p is not None:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm


def compute_norm_gradients(losses, model):
    grads = []
    params_to_differentiate = [param for param in model.parameters() if param.requires_grad]

    for loss in losses:
        try:
            # This call might fail with a RuntimeError
            grad = torch.autograd.grad(loss, params_to_differentiate, retain_graph=True, allow_unused=True)
            grads.append(grad)
        except RuntimeError as e:
            print(f"Skipping gradient computation due to error: {e}")
            grads.append(None)  # Append None or handle differently if you wish

    # Compute norm for each valid gradient, skip None entries
    norm_grads = [compute_norm(grad) if grad is not None else 0 for grad in grads]
    return norm_grads



def plot_metrics(train, test,plot_type, save_dir):
    plt.figure()
    plt.plot(train, c="b")
    plt.plot(test, c="r")

    plt.legend([f"Train {plot_type}", f"Test {plot_type}"])
    plt.xlabel("Epoch")
    plt.ylabel(f"{plot_type}")
    plt.savefig(save_dir)

def plot_grads(grads, save_dir):
    grads = {key: np.hstack(grads[key]) for key in grads}
    num_rows = len(grads.keys())
    fig, axes = plt.subplots(num_rows, figsize=(15, 10), squeeze=False)  # ensure axes is always 2D
    axes = axes.flatten()  # Flatten the array for easy iteration

    keys = list(grads.keys())
    num_iters = len(grads[keys[0]])

    for i, key in enumerate(keys):
        axes[i].plot(range(num_iters), grads[key])
        axes[i].set_title(f"Normalized gradients of {key} with respect to model parameters")
        axes[i].set_xlabel("Iterations")
        axes[i].set_ylabel("Normalized gradients over all model parameters")
    
    fig.tight_layout()
    plt.savefig(save_dir)
    plt.close(fig)  # Close the figure to free memory

def make_dot(var, params, graph_name):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(graph_name, node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot