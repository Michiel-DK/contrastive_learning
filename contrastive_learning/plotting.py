import matplotlib.pyplot as plt
import numpy as np
import torch

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
