import matplotlib.pyplot as plt
import numpy as np
import torch
import math


def plot_combined_grid(images, contrastive_images, labels, rows=4, cols=4, figsize=(20, 20), normalize=True):
    """
    Plots a grid of image groups, each containing the original image and two contrastive images side by side,
    with titles from the labels tensor.

    Args:
        images (torch.Tensor): Tensor of original images with shape [N, 3, H, W].
        contrastive_images (torch.Tensor): Tensor of contrastive images with shape [N, 2, 3, H, W].
        labels (torch.Tensor or list): Tensor or list of labels with shape [N].
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        figsize (tuple): Size of the entire figure.
        normalize (bool): Whether to normalize images to [0,1] for display.
    """
    N, C, H, W = images.shape
    N_contrastive, num_contrastive, C_contrastive, H_contrastive, W_contrastive = contrastive_images.shape

    assert N == N_contrastive, "Number of samples in images and contrastive_images must match."
    assert C == C_contrastive and H == H_contrastive and W == W_contrastive, "Image dimensions must match."

    # Ensure labels is a list or a 1D tensor
    if isinstance(labels, torch.Tensor):
        if labels.dim() == 1:
            labels = labels.tolist()
        else:
            raise ValueError("labels tensor must be 1-dimensional.")
    elif isinstance(labels, list):
        if len(labels) != N:
            raise ValueError("Length of labels list must match the number of images.")
    else:
        raise TypeError("labels must be a torch.Tensor or a list.")

    total_slots = rows * cols
    if N > total_slots:
        print(f"Warning: You have {N} samples but the grid can only accommodate {total_slots} slots.")
        print("Some samples will be ignored.")
        N = total_slots
    elif N < total_slots:
        print(f"Warning: You have {N} samples but the grid has {total_slots} slots.")
        print("Some slots will be empty.")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i in range(total_slots):
        ax = axes[i]
        ax.axis('off')  # Hide axes by default

        if i < N:
            # Original image
            img_orig = images[i].permute(1, 2, 0).cpu().numpy()
            # Contrastive images
            img_contrast1 = contrastive_images[i, 0].permute(1, 2, 0).cpu().numpy()
            img_contrast2 = contrastive_images[i, 1].permute(1, 2, 0).cpu().numpy()

            if normalize:
                # Normalize each image to [0,1] for display
                def normalize_image(img):
                    img_min = img.min()
                    img_max = img.max()
                    if img_max > img_min:
                        return (img - img_min) / (img_max - img_min)
                    else:
                        return img

                img_orig = normalize_image(img_orig)
                img_contrast1 = normalize_image(img_contrast1)
                img_contrast2 = normalize_image(img_contrast2)

            # Concatenate images horizontally
            concatenated = np.concatenate((img_orig, img_contrast1, img_contrast2), axis=1)

            ax.imshow(concatenated)
            ax.set_title(f"{labels[i]}", fontsize=10)
            ax.axis('off')  # Hide axes for this subplot

    plt.tight_layout()
    plt.show()

def plot_contrastive_image_pairs(data, unnormalize=None):
    """
    Plots pairs of contrastive images side by side with labels as titles.

    Parameters:
    - data: list containing [images, labels]
        - images: Tensor of shape [batch_size, 2, 3, 256, 256]
        - labels: Tensor of shape [batch_size]
    - unnormalize: function to unnormalize images if they were normalized
    """
    images, labels = data  # Unpack the data list

    batch_size = images.size(0)
    num_contrastive = images.size(1)  # Should be 2

    # Determine grid size (e.g., 4x4 for batch_size=16)
    grid_size = math.ceil(math.sqrt(batch_size))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(4 * grid_size, 4 * grid_size))
    axes = axes.flatten()  # Flatten in case of a grid larger than batch_size

    for idx in range(batch_size):
        ax = axes[idx]
        pair = images[idx]  # Shape: [2, 3, 256, 256]
        label = labels[idx].item() if isinstance(labels[idx], torch.Tensor) else labels[idx]

        imgs = []
        for i in range(num_contrastive):
            img = pair[i]  # Shape: [3, 256, 256]
            img = img.cpu().numpy()  # Convert to NumPy array
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC

            if unnormalize:
                img = unnormalize(img)

            # Ensure the image is in the range [0, 1] for display
            img = np.clip(img, 0, 1)

            imgs.append(img)

        # Concatenate images horizontally
        combined_img = np.hstack(imgs)
        ax.imshow(combined_img)
        ax.axis('off')
        ax.set_title(f'Label: {label}', fontsize=14)

    # Remove any unused subplots
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()