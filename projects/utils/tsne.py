import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import wandb

# from
# https://learnopencv.com/t-sne-for-feature-visualization/
# also https://learnopencv.com/t-sne-t-distributed-stochastic-neighbor-embedding-explained/


def tsne_analysis(model, data_loader, ids_unique, device, plot_size=1000, max_image_size=100):
    """
    Perform T-SNE analysis on the given model's output features and visualize the results.

    Args:
        model: The trained model.
        data_loader: Data loader providing the input data.
        ids_unique: Unique IDs of the data samples.
        device: Device to run the model on.
        plot_size (int, optional): Size of the T-SNE plot. Default is 1000.
        max_image_size (int, optional): Maximum size of the images in the T-SNE plot. Default is 100.

    Returns:
        None
    """
    model.eval()
    features, labels, imgs = None, [], []
    pca = PCA(n_components=50)
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, total=len(data_loader), desc="TSNE analysis"):
            labels += batch["id"]
            imgs += batch["path"]
            images = batch["img"].to(device)

            output = model.forward(images)
            current_features = output.detach().cpu().numpy()
            features = (
                current_features
                if features is None
                else np.concatenate((features, current_features))
            )

    # Perform PCA on the features
    current_features = pca.fit_transform(features)

    # Perform T-SNE on the features
    tsne = TSNE(n_components=2, n_iter=3000).fit_transform(features)

    # Extract x and y coordinates from T-SNE output
    tx, ty = tsne[:, 0], tsne[:, 1]

    # Scale and move the coordinates to fit the [0, 1] range
    tx, ty = scale_to_01_range(tx), scale_to_01_range(ty)

    # Visualize the T-SNE plot as colored points
    visualize_tsne_points(tx, ty, labels, ids_unique)

    # Visualize the T-SNE plot with samples as images
    visualize_tsne_images(tx, ty, imgs, labels, plot_size, max_image_size)


def scale_to_01_range(x):
    """
    Scale the given array to the [0, 1] range.

    Args:
        x: Input array.

    Returns:
        Scaled array.
    """
    # compute the distribution range
    value_range = np.max(x) - np.min(x)
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    """
    Scale the given image while maintaining aspect ratio.

    Args:
        image: Input image.
        max_image_size: Maximum size for the scaled image.

    Returns:
        Scaled image.
    """
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label, labels_color):
    """
    Draw a rectangle around the image with a color corresponding to the class label.

    Args:
        image: Input image.
        label: Class label.
        labels_color: Dictionary to store class labels and their corresponding colors.

    Returns:
        Image with a rectangle drawn around it.
    """
    image_height, image_width, _ = image.shape
    if label not in labels_color:
        labels_color[label] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    image = cv2.rectangle(
        image,
        (0, 0),
        (image_width - 1, image_height - 1),
        color=labels_color[label],
        thickness=5,
    )
    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    """
    Compute the plot coordinates for placing the image based on its T-SNE coordinates.

    Args:
        image: Input image.
        x: T-SNE x coordinate.
        y: T-SNE y coordinate.
        image_centers_area_size: Size of the area where image centers are placed.
        offset: Offset for placing the images.

    Returns:
        Top-left and bottom-right coordinates for placing the image on the plot.
    """
    image_height, image_width, _ = image.shape
    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset
    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)
    br_x = tl_x + image_width
    br_y = tl_y + image_height
    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    """
    Visualize the T-SNE plot with samples as images.

    Args:
        tx: T-SNE x coordinates.
        ty: T-SNE y coordinates.
        images: List of image paths.
        labels: List of labels.
        plot_size: Size of the plot.
        max_image_size: Maximum size for the scaled images.

    Returns:
        None
    """
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    labels_color = {}
    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image, label, x, y in tqdm(
        zip(images, labels, tx, ty), desc="Building the T-SNE plot", total=len(images)
    ):
        image = cv2.imread(image)
        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)
        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label, labels_color)
        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(
            image, x, y, image_centers_area_size, offset
        )
        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    # add a legend outside the image
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.imshow(tsne_plot[:, :, ::-1])
    plt.savefig("tsne_images.png")
    wandb.log({"tsne_images": wandb.Image("tsne_images.png")})


def visualize_tsne_points(tx, ty, labels, classes):
    """
    Visualize the T-SNE plot with samples as colored points.

    Args:
        tx: T-SNE x coordinates.
        ty: T-SNE y coordinates.
        labels: List of labels.
        classes: List of unique classes.

    Returns:
        None
    """
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in classes:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc="best")
    # finally, show the plot
    plt.savefig("tsne_points.png")
    wandb.log({"tsne_points": wandb.Image("tsne_points.png")})


def visualize_tsne(tsne, images, labels, plot_size=10000, max_image_size=1000):
    """
    Visualize the T-SNE plot with both colored points and samples as images.

    Args:
        tsne: T-SNE coordinates.
        images: List of image paths.
        labels: List of labels.
        plot_size: Size of the plot.
        max_image_size: Maximum size for the scaled images.

    Returns:
        None
    """
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    visualize_tsne_images(
        tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size
    )
