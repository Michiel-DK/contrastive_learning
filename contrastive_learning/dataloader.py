import random
    
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

from torchvision.datasets import STL10  # **Added Import for STL10**

import matplotlib.pyplot as plt

from itertools import chain

import io

import torch

from google.cloud import storage

def contrastive_collate_fn(batch):
    """
    Custom collate function to batch images and labels for contrastive learning.
    
    Args:
        batch: List of tuples (contrastive_images, label), where contrastive_images is a list of tensors.
        
    Returns:
        Tuple[Tensor, Tensor]:
            - Contrastive images tensor of shape (batch_size, n_views, C, H, W)
            - Labels tensor of shape (batch_size,)
    """
    # Separate data and labels from the batch
    data, labels = zip(*batch)

    # Ensure `data` is a list of tensors of shape (n_views, C, H, W)
    # Stack to form the output shape: (batch_size, n_views, C, H, W)
    contrastive_images = torch.stack([torch.stack(views) for views in data])  # Shape: (batch_size, n_views, C, H, W)

    # Stack labels into a single tensor with shape (batch_size,)
    labels_tensor = torch.tensor(labels, dtype=torch.long)  # Shape: (batch_size,)

    # Debugging to confirm the structure
    print("Contrastive images shape:", contrastive_images.shape)  # Expected: (batch_size, n_views, C, H, W)
    print("Labels tensor shape:", labels_tensor.shape)            # Expected: (batch_size,)
    
    import ipdb;ipdb.set_trace()

    return contrastive_images, labels_tensor

def set_label_to_minus_one(label):
    return -1

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

class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Original label is ignored
        return data, -1  # Label is set to -1 for unlabeled data

def fetch_images_masks_from_gcs(bucket_name, prefix='data/', is_unlabeled=False):
    """
    Fetch image and mask paths from a Google Cloud Storage bucket based on the dataset structure.

    Args:
        bucket_name (str): The GCS bucket name.
        prefix (str): The GCS path prefix (e.g., 'data/' to start from the data folder).
        is_unlabeled (bool): If True, assigns -1 as labels for all images.

    Returns:
        tuple: Lists of image files, mask files (if applicable), labels, classes, and class_to_idx.
    """
    # Initialize GCS client and bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Collect data
    image_files = []
    mask_files = []
    labels = []
    classes = []
    class_to_idx = {}

    # List objects within the specified prefix in the bucket
    blobs = bucket.list_blobs(prefix=prefix)

    # Dictionary to map each class to its images and masks
    class_images = {}
    class_masks = {}

    for blob in blobs:
        path_parts = blob.name.split('/')

        # Skip if blob does not follow expected structure
        if len(path_parts) < 3:
            continue

        class_name = path_parts[1]  # Extract class name from folder structure
        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(class_to_idx)
            classes.append(class_name)
            class_images[class_name] = []
            class_masks[class_name] = []

        # Check if it's an image or mask file and assign paths accordingly
        if path_parts[-1].endswith(('.jpg', '.jpeg', '.png')):
            if " GT" in path_parts[-2]:  # Assume masks are in "class_name GT" folder
                class_masks[class_name].append(blob.name)
            else:
                class_images[class_name].append(blob.name)

    # Match images to masks by filename and build final lists
    for class_name, images in class_images.items():
        masks = {os.path.splitext(os.path.basename(m))[0]: m for m in class_masks.get(class_name, [])}
        for img_path in images:
            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = masks.get(img_stem)
            if mask_path:
                image_files.append(f"gs://{bucket_name}/{img_path}")
                mask_files.append(f"gs://{bucket_name}/{mask_path}")
                labels.append(class_to_idx[class_name] if not is_unlabeled else -1)

    return image_files, mask_files, labels, classes, class_to_idx


class ImageMaskDataset(VisionDataset):
    def __init__(
        self, 
        root, 
        transform=None, 
        target_transform=None, 
        apply_mask=False, 
        contrastive_transform=None, 
        is_unlabeled=False,
        gcs_bucket_name=None,
        gcs_prefix='data/'
    ):
        super(ImageMaskDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.transform = transform
        self.target_transform = target_transform
        self.apply_mask = apply_mask
        self.contrastive_transform = contrastive_transform
        self.is_unlabeled = is_unlabeled

        # Check if a GCS bucket name is provided
        if gcs_bucket_name:
            # Use GCS paths
            self.image_files, self.mask_files, self.labels, self.classes, self.class_to_idx = \
                fetch_images_masks_from_gcs(gcs_bucket_name, prefix=gcs_prefix, is_unlabeled=is_unlabeled)
                
        else:
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

        # If using GCS, load images and masks directly from GCS
        if img_path.startswith("gs://"):
            client = storage.Client()
            bucket_name, img_blob_name = img_path[5:].split('/', 1)
            mask_bucket_name, mask_blob_name = mask_path[5:].split('/', 1)

            # Download image from GCS
            bucket = client.bucket(bucket_name)
            img_blob = bucket.blob(img_blob_name)
            img_bytes = img_blob.download_as_bytes()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Download mask from GCS
            mask_bucket = client.bucket(mask_bucket_name)
            mask_blob = mask_bucket.blob(mask_blob_name)
            mask_bytes = mask_blob.download_as_bytes()
            mask = Image.open(io.BytesIO(mask_bytes)).convert('L')  # Grayscale
        else:
            # Load image and mask from the local file system
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

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

    def apply_mask_to_image(self, image, mask, padding=10, resize_size=(96, 96)):
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
        
        # Find the bounding box of the mask
        # Convert mask to boolean for easier operations
        mask_bool = mask[0].bool()  # Shape: (H, W)

        # Check if the mask has any non-zero pixels
        if not mask_bool.any():
            print("Warning: Mask is empty. Returning the original masked image without cropping.")
            return masked_image

        # Get the coordinates of non-zero pixels
        coords = mask_bool.nonzero(as_tuple=False)  # Shape: (N, 2), where N is number of non-zero pixels
        y_min = torch.min(coords[:, 0]).item()
        y_max = torch.max(coords[:, 0]).item()
        x_min = torch.min(coords[:, 1]).item()
        x_max = torch.max(coords[:, 1]).item()

        # Add padding while ensuring the coordinates are within image boundaries
        H, W = mask.shape[1], mask.shape[2]
        y_min_padded = max(y_min - padding, 0)
        y_max_padded = min(y_max + padding, H)
        x_min_padded = max(x_min - padding, 0)
        x_max_padded = min(x_max + padding, W)

        # Crop the masked image
        cropped_masked_image = masked_image[:, y_min_padded:y_max_padded, x_min_padded:x_max_padded]
            
        # Convert the cropped tensor to PIL Image for resizing
        cropped_pil = transforms.ToPILImage()(cropped_masked_image)

        # Resize the cropped image to the desired size
        resized_pil = transforms.Resize(resize_size)(cropped_pil)

        # Convert back to tensor
        resized_image_tensor = transforms.ToTensor()(resized_pil)

        return resized_image_tensor
    

class SubsetWithModifiedLabels(Dataset):
    """
    A wrapper for a subset of a dataset that modifies the labels.
    
    Args:
        subset (Subset): The subset of the dataset.
        label_transform (callable): A function to transform the labels.
    """
    def __init__(self, subset, label_transform):
        self.subset = subset
        self.label_transform = label_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data, label = self.subset[idx]
        label = self.label_transform(label)
        return data, label

def get_datasets(batch_size=16, labeled_split=0.3, dataset_type='ImageMaskDataset', DATASET_PATH='path_to_dataset/', gcs_bucket_name=None, gcs_prefix='data/'):
    """
    Initializes and returns DataLoaders for the selected dataset type.

    Args:
        batch_size (int, optional): Number of samples per batch.
        labeled_split (float, optional): Fraction of labeled data (only for ImageMaskDataset).
        dataset_type (str, optional): Type of dataset to use ('ImageMaskDataset' or 'STL10').
        DATASET_PATH (str, optional): Path for STL10 dataset storage.

    Returns:
        tuple: Depending on dataset_type:
            - If 'ImageMaskDataset': (labeled_dataloader, unlabeled_dataloader)
            - If 'STL10': (stl_unlabeled_dataloader, stl_train_contrast_dataloader)
    """
    dataset_type = dataset_type.lower()
    supported_types = ['imagemaskdataset', 'stl10']
    if dataset_type not in supported_types:
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'. Supported types: {supported_types}")

    # Common parameters
    NUM_WORKERS = 2  # Adjust based on your system
    
            # Define transformations
    image_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    mask_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()  # Convert mask to tensor directly
        ])

    contrast_transforms = ContrastiveTransformations(
            transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=96),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.5,
                                           contrast=0.5,
                                           saturation=0.5,
                                           hue=0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=9),
                transforms.ToTensor(),
                # Optional: Uncomment if normalization is desired
                transforms.Normalize((0.5,), (0.5,))
            ]),
            n_views=2  # Number of contrastive views
        )

    if dataset_type == 'imagemaskdataset':
        # Define root directory for custom dataset
        root_dir = 'data/' if gcs_bucket_name is None else None

        # Initialize the full dataset once
        full_dataset = ImageMaskDataset(
            root=root_dir,
            transform=image_transform,
            target_transform=mask_transform,
            apply_mask=True,
            contrastive_transform=contrast_transforms,  # For labeled data
            is_unlabeled=False,  # Initially, labels are based on class indices
            gcs_bucket_name=gcs_bucket_name,
            gcs_prefix=gcs_prefix
        )
        
        # Determine labeled and unlabeled indices based on label_fraction
        label_fraction = labeled_split  # e.g., 0.3 for 30% labeled
        total_size = len(full_dataset)
        labeled_size = int(total_size * label_fraction)
        unlabeled_size = total_size - labeled_size

        # Create a shuffled list of indices
        indices = list(range(total_size))
        random.seed(123)  # Ensure reproducibility
        random.shuffle(indices)

        # Split indices
        labeled_indices = indices[:labeled_size]
        unlabeled_indices = indices[labeled_size:]

        print(f'Len labeled dataset: {len(labeled_indices)} / Len unlabeled dataset: {len(unlabeled_indices)}')

        # Create Subsets
        labeled_subset = Subset(full_dataset, labeled_indices)
        unlabeled_subset_original = Subset(full_dataset, unlabeled_indices)

        # Wrap the unlabeled subset to modify labels
        unlabeled_subset = SubsetWithModifiedLabels(
            unlabeled_subset_original,
            label_transform=set_label_to_minus_one  # Use the top-level function
        # Set all labels to -1
            )
        
        import ipdb;ipdb.set_trace()
        
        # Create DataLoaders for ImageMaskDataset
        labeled_dataloader = DataLoader(
            labeled_subset,
            batch_size=batch_size,
            shuffle=False,  # Shuffle for training
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=contrastive_collate_fn #lambda batch: contrastive_collate_fn(batch) # Use custom collate function
        )

        unlabeled_dataloader = DataLoader(
            unlabeled_subset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True ,
            collate_fn=lambda batch: contrastive_collate_fn(batch)   # Use custom collate function
        )

        print("ImageMaskDataset DataLoaders Created Successfully.")

        return labeled_subset, unlabeled_subset

    elif dataset_type == 'stl10':

        print(labeled_split)
        # Load the STL10 train split
        full_train_dataset = STL10(
        root=DATASET_PATH,
        split='train',
        download=True,
        transform=contrast_transforms
        )
        total_train = len(full_train_dataset)
        print(f"Total training samples: {total_train}")
    
        # Calculate subset sizes
        labeled_size = int(total_train * labeled_split)
        unlabeled_size = total_train - labeled_size
        print(f"Labeled samples: {labeled_size} ({labeled_split*100}%)")
        print(f"Unlabeled samples: {unlabeled_size} ({(1 - labeled_split)*100}%)")

        # Ensure reproducibility
        generator = torch.Generator().manual_seed(123)

        # Split the dataset
        labeled_subset, unlabeled_subset = random_split(
            full_train_dataset,
            [labeled_size, unlabeled_size],
            generator=generator
        )

        print(f"Filtered labeled_subset size: {len(labeled_subset)}")
        print(f"Filtered unlabeled_subset size: {len(unlabeled_subset)}")

        # Wrap the unlabeled subset to ignore labels
        unlabeled_subset = UnlabeledDataset(unlabeled_subset)

        # Create DataLoaders
        labeled_dataloader = DataLoader(
            labeled_subset,
            batch_size=batch_size,
            shuffle=False,  # Shuffle labeled data
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        collate_fn=collate_fn  # Use custom collate function
        )

        unlabeled_dataloader = DataLoader(
            unlabeled_subset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle unlabeled data
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        collate_fn=collate_fn  # Use custom collate function
        )
        
        print("STL10 Labeled and Unlabeled DataLoaders Created Successfully.")

        return unlabeled_dataloader, labeled_dataloader
    

if __name__ == '__main__':
    
    try:

        labeled_dataloader, unlabeled_dataloader = get_datasets(batch_size=256, labeled_split=0.1, dataset_type='ImageMaskDataset', DATASET_PATH='data/', gcs_bucket_name='fish-dataset-cl', gcs_prefix='data/')
                        
        labeled_batch = next(iter(labeled_dataloader))
        unlabeled_batch = next(iter(unlabeled_dataloader))

        # Check labeled batch structure
        print(
            "Labeled batch structure:",
            type(labeled_batch),
            labeled_batch[0].shape,  # Expected: torch.Size([256, 2, 3, 96, 96])
            labeled_batch[1].shape,  # Expected: torch.Size([256])
            labeled_batch[1]  # Should display real labels for labeled data
        )

        # Check unlabeled batch structure
        print(
            "Unlabeled batch structure:",
            type(unlabeled_batch),
            unlabeled_batch[0].shape,  # Expected: torch.Size([256, 2, 3, 96, 96])
            unlabeled_batch[1].shape,  # Expected: torch.Size([256])
            unlabeled_batch[1]  # Should display all -1 for unlabeled data
        )
        import ipdb;ipdb.set_trace()
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
