import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from contrastive_learning.dataloader import *
import torchvision.transforms.functional as TF


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
    

def visualize_image(image_tensor):
    """
    Visualizes an image tensor.

    Args:
        image_tensor (Tensor): Image tensor of shape (3, H, W).
    """
    image_pil = TF.to_pil_image(image_tensor)
    plt.imshow(image_pil)
    plt.axis('off')
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np

def plot_train_val_batch_side_by_side(train_loader, val_loader, batch_size=8, unnormalize=None):
    """
    Plots a batch of images from both the train and validation loaders side by side.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        batch_size (int): Number of samples to display from each loader.
        unnormalize (function, optional): Function to unnormalize images if necessary.
    """
    # Get one batch from both train and validation loaders
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    train_images, train_labels = train_batch
    val_images, val_labels = val_batch

    # Ensure the batch size does not exceed the available samples
    batch_size = min(batch_size, len(train_images), len(val_images))

    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    if batch_size == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one sample

    for i in range(batch_size):
        # Train image
        train_img = train_images[i].permute(1, 2, 0).cpu().numpy()
        val_img = val_images[i].permute(1, 2, 0).cpu().numpy()

        # Unnormalize if necessary
        if unnormalize:
            train_img = unnormalize(train_img)
            val_img = unnormalize(val_img)

        # Clip images to [0, 1] range for display
        train_img = np.clip(train_img, 0, 1)
        val_img = np.clip(val_img, 0, 1)

        # Plot train image
        axes[i][0].imshow(train_img)
        axes[i][0].set_title(f"Train Label: {train_labels[i].item()}")
        axes[i][0].axis('off')

        # Plot validation image
        axes[i][1].imshow(val_img)
        axes[i][1].set_title(f"Validation Label: {val_labels[i].item()}")
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        # Initialize STL10 DataLoaders with a subset
        stl_unlabeled_dataloader, stl_train_contrast_dataloader = get_datasets(
            batch_size=16, 
            dataset_type='imagemaskdataset', 
            DATASET_PATH='data/',  # Specify where to download/load STL10 data
            labeled_split=0.3  # Retain 30% of the train_data_contrast
        )

        # Get a single batch from the train_contrast DataLoader
        for batch in stl_train_contrast_dataloader:
            contrastive_images, _ = batch
            # Select the first image in the batch (first view)
            first_image = contrastive_images[0][0]  # Assuming n_views=2
            visualize_image(first_image)
            break  # Visualize only one image

    except Exception as e:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
