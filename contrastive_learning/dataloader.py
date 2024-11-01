import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import glob
import random
    
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class ContrastiveTransformations:
    """
    Creates multiple augmented views of an image for contrastive learning.

    Args:
        base_transform (callable): A torchvision.transforms transformation to apply.
        n_views (int): Number of augmented views to generate.
    """
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for _ in range(self.n_views)]
    

class ImageMaskDataset(VisionDataset):
    """
    A custom dataset class for image and mask pairs with optional contrastive transformations.

    Attributes:
        root (str): Root directory of the dataset.
        transform (callable, optional): Transformation to apply to images.
        target_transform (callable, optional): Transformation to apply to masks.
        apply_mask (bool): Whether to apply the mask to the image.
        contrastive_transform (callable, optional): Transformation for contrastive learning.
        is_unlabeled (bool): If True, all labels are set to -1.
        classes (list): List of class names.
        class_to_idx (dict): Mapping from class names to indices.
        image_files (list): List of image file paths.
        mask_files (list): List of corresponding mask file paths.
        labels (list): List of labels for each image.
    """
    def __init__(
        self, 
        root, 
        transform=None, 
        target_transform=None, 
        apply_mask=False, 
        contrastive_transform=None, 
        is_unlabeled=False
    ):
        """
        Initializes the ImageMaskDataset.

        Args:
            root (str): Root directory of the dataset where images and masks are stored.
            transform (callable, optional): A function/transform to apply to the images.
            target_transform (callable, optional): A function/transform to apply to the masks.
            apply_mask (bool, optional): Whether to apply masks to images.
            contrastive_transform (callable, optional): Additional transformations for contrastive learning.
            is_unlabeled (bool, optional): If True, all labels are set to -1.
        """
        super(ImageMaskDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.transform = transform
        self.target_transform = target_transform
        self.apply_mask = apply_mask
        self.contrastive_transform = contrastive_transform
        self.is_unlabeled = is_unlabeled

        self.image_files = []
        self.mask_files = []
        self.labels = []  # To store class labels

        # 1. Extract class names from subdirectories
        self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        print(f"Class to Index Mapping: {self.class_to_idx}")

        # 2. Collect image and mask file paths, assign class labels
        for class_name in self.classes:
            
            class_dir = os.path.join(root, class_name, class_name)
            mask_dir = os.path.join(root, class_name, f"{class_name} GT")

            # Check if mask directory exists
            if not os.path.isdir(mask_dir):
                print(f"Mask directory not found for class '{class_name}'. Skipping this class.")
                continue

            # List all image files in the class directory
            imgs = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                   glob.glob(os.path.join(class_dir, '*.jpeg')) + \
                   glob.glob(os.path.join(class_dir, '*.png'))
            print(f"\nProcessing Class: {class_name}")
            print(f"Number of Images Found: {len(imgs)}")

            # List all mask files in the mask directory
            masks = glob.glob(os.path.join(mask_dir, '*.jpg')) + \
                    glob.glob(os.path.join(mask_dir, '*.jpeg')) + \
                    glob.glob(os.path.join(mask_dir, '*.png'))
            print(f"Number of Masks Found: {len(masks)}")

            # Create a mapping from image stem to mask path
            mask_dict = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}

            # Iterate over each image and find its corresponding mask
            for img_path in imgs:
                img_stem = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = mask_dict.get(img_stem, None)
                if mask_path:
                    self.image_files.append(img_path)
                    self.mask_files.append(mask_path)
                    if self.is_unlabeled:
                        self.labels.append(-1)  # Set label to -1 for unlabeled data
                    else:
                        self.labels.append(self.class_to_idx[class_name])  # Assign class index
                else:
                    # Image without corresponding mask is skipped
                    print(f"Skipping Image (No Mask Found): {img_path}")
                    continue

        print(f"\nTotal Images with Masks: {len(self.image_files)}")
        print(f"Total Masks: {len(self.mask_files)}")
        print(f"Total Labels: {len(self.labels)}")

        assert len(self.image_files) == len(self.mask_files) == len(self.labels), \
            f"Number of images ({len(self.image_files)}), masks ({len(self.mask_files)}), and labels ({len(self.labels)}) must match."

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (contrastive_image, label) if contrastive_transform is applied,
                   else (image, label).
        """
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        label = self.labels[idx]

        # Load image and mask
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")
        
        try:
            mask = Image.open(mask_path).convert('L')  # Grayscale
        except Exception as e:
            raise IOError(f"Error loading mask {mask_path}: {e}")

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Apply transforms to mask if present
        if self.target_transform:
            mask = self.target_transform(mask)

        # Apply the mask to the image if required and if mask is present
        if self.apply_mask and mask is not None:
            image = self.apply_mask_to_image(image, mask)

        # **Apply Contrastive Transform if provided**
        if self.contrastive_transform:
            # Convert tensor back to PIL Image for contrastive transformations
            image_pil = transforms.ToPILImage()(image)
            contrastive_images = self.contrastive_transform(image_pil)  # List of tensors
            contrastive_image = torch.stack(contrastive_images)  # Shape: (n_views, C, H, W)
        else:
            contrastive_image = None

        # **Prepare the return tuple**
        if contrastive_image is not None:
            # Return contrastive_image and label
            return contrastive_image, torch.tensor(label, dtype=torch.long)
        else:
            # Return image and label
            return image, torch.tensor(label, dtype=torch.long)

    def apply_mask_to_image(self, image, mask):
        """
        Applies a binary mask to the image.

        Args:
            image (Tensor): Image tensor of shape (3, H, W).
            mask (Tensor): Mask tensor of shape (1, H, W).

        Returns:
            Tensor: Masked image.
        """
        # Ensure the mask is binary
        mask = (mask > 0).float()

        # Expand mask to have the same number of channels as image
        if mask.shape[0] == 1 and image.shape[0] == 3:
            mask = mask.expand_as(image)

        # Apply the mask
        masked_image = image * mask

        return masked_image

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os
import random

def get_datasets(batch_size=16):
    """
    Initializes and returns DataLoaders for labeled and unlabeled datasets.

    Args:
        batch_size (int, optional): Number of samples per batch.

    Returns:
        tuple: (labeled_dataloader, unlabeled_dataloader)
    """
    # Define root directory
    root_dir = 'data/'

    # Define image and mask directories
    image_dirs = [os.path.abspath(os.path.join(root_dir, x, x)) 
                 for x in os.listdir(root_dir) 
                 if not x.endswith('.txt') and not x.endswith('.m')]

    # Exclude 'NA_Fish_Dataset' directories
    image_dirs = [path for path in image_dirs if 'NA_Fish_Dataset' not in path]

    # Define mask directories
    mask_dirs = [os.path.abspath(os.path.join(root_dir, x, f"{x} GT")) 
                for x in os.listdir(root_dir) 
                if not x.endswith('.txt') and not x.endswith('.m')]

    mask_dirs = [path for path in mask_dirs if 'NA_Fish_Dataset' not in path]

    # Example transformations (ensure they convert images/masks to tensors correctly)
    image_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()  # Convert mask to tensor directly
    ])

    contrast_transforms = ContrastiveTransformations(  # Using the implemented class
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=256),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5,
                                       contrast=0.5,
                                       saturation=0.5,
                                       hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
           # transforms.Resize((96, 96)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ]),
        n_views=2  # Number of contrastive views
    )

    # Initialize the full labeled dataset
    full_labeled_dataset = ImageMaskDataset(
        root=root_dir,
        transform=image_transform,
        target_transform=mask_transform,
        apply_mask=True,
        contrastive_transform=contrast_transforms,  # For labeled data
        is_unlabeled=False  # Labels are based on class indices
    )

    # Determine labeled and unlabeled indices based on label_fraction
    label_fraction = 0.3 # 70% labeled, 30% unlabeled
    total_size = len(full_labeled_dataset)
    labeled_size = int(total_size * label_fraction)
    unlabeled_size = total_size - labeled_size

    # Create a shuffled list of indices
    indices = list(range(total_size))
    random.seed(123)  # Ensure reproducibility
    random.shuffle(indices)

    # Split indices
    labeled_indices = indices[:labeled_size]
    unlabeled_indices = indices[labeled_size:]

    # Create Subsets
    labeled_subset = Subset(full_labeled_dataset, labeled_indices)

    # Initialize the unlabeled dataset with labels set to -1 and contrastive_transform
    full_unlabeled_dataset = ImageMaskDataset(
        root=root_dir,
        transform=image_transform,
        target_transform=mask_transform,
        apply_mask=True,
        contrastive_transform=contrast_transforms,  # Include contrastive transforms for consistency
        is_unlabeled=True  # All labels set to -1
    )

    # Create Subset for unlabeled data
    unlabeled_subset = Subset(full_unlabeled_dataset, unlabeled_indices)

    # Create DataLoaders
    NUM_WORKERS = 4  # Adjust based on your system

    # DataLoader for labeled data
    labeled_dataloader = DataLoader(
        labeled_subset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle for training
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # Keeps workers alive between epochs
    )

    # DataLoader for unlabeled data
    unlabeled_dataloader = DataLoader(
        unlabeled_subset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # Keeps workers alive between epochs
    )

    return labeled_dataloader, unlabeled_dataloader

    

if __name__ == '__main__':
    
    try:

        labeled_dataloader, unlabeled_dataloader = get_datasets()
        
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
