import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import glob
import random

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import os
import glob
import random
from torchvision.datasets import VisionDataset


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VisionDataset
from torchvision import transforms
from PIL import Image
import os
import glob
import random

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

class ImageMaskDataset(VisionDataset):
    def __init__(
        self, 
        root, 
        split='train',  # 'train', 'test', 'unlabeled', 'train+unlabeled'
        transform=None, 
        target_transform=None, 
        apply_mask=False, 
        contrastive_transform=None, 
        label_fraction=1.0,
        download=False,
        seed=42
    ):
        """
        Args:
            root (string): Root directory of the dataset where images and masks are stored.
            split (string, optional): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
                                      Determines which subset of the dataset to use.
            transform (callable, optional): A function/transform to apply to the images.
            target_transform (callable, optional): A function/transform to apply to the masks.
            apply_mask (bool, optional): Whether to apply masks to images.
            contrastive_transform (callable, optional): Additional transformations for contrastive learning.
            label_fraction (float, optional): Fraction of data to be labeled (between 0 and 1).
            download (bool, optional): If True, downloads the dataset (not implemented).
            seed (int, optional): Random seed for reproducibility.
        """
        super(ImageMaskDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.apply_mask = apply_mask
        self.contrastive_transform = contrastive_transform
        self.label_fraction = label_fraction
        self.seed = seed

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

        # Shuffle and split the dataset based on split parameter
        random.seed(self.seed)
        self.indices = list(range(len(self.image_files)))
        random.shuffle(self.indices)

        if self.split == 'train':
            labeled_count = int(len(self.image_files) * self.label_fraction)
            self.labeled_set = set(self.indices[:labeled_count])
        elif self.split == 'test':
            # Define test split as the remaining data after training
            labeled_count = int(len(self.image_files) * self.label_fraction)
            self.labeled_set = set(self.indices[labeled_count:])
        elif self.split == 'unlabeled':
            # All data is considered unlabeled
            self.labeled_set = set()
        elif self.split == 'train+unlabeled':
            # All data is used for training (both labeled and unlabeled)
            self.labeled_set = set(self.indices[:int(len(self.image_files) * self.label_fraction)])
        else:
            raise ValueError(f"Invalid split '{self.split}'. Expected one of ['train', 'test', 'unlabeled', 'train+unlabeled'].")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.image_files[actual_idx]
        mask_path = self.mask_files[actual_idx]
        label = self.labels[actual_idx]

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Apply transforms to mask if present
        if self.target_transform:
            mask = self.target_transform(mask)

        # Apply the mask to the image if required and if mask is present
        if self.apply_mask and mask is not None:
            image = self.apply_mask_to_image(image, mask)

        # Determine if the sample is labeled
        is_labeled = actual_idx in self.labeled_set

        # Initialize contrastive_image as None
        contrastive_image = None

        # Apply contrastive transform if provided
        if self.contrastive_transform:
            # Convert tensor back to PIL Image for contrastive transformations
            image_pil = transforms.ToPILImage()(image)
            contrastive_images = self.contrastive_transform(image_pil)  # List of tensors
            contrastive_image = torch.stack(contrastive_images)  # Shape: (n_views, C, H, W)

        # Prepare the return dictionary
        sample = {
            'image': image,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long),  # Multiclass label
            'is_labeled': torch.tensor(is_labeled, dtype=torch.float32)  # 1.0 for labeled, 0.0 for unlabeled
        }

        if self.contrastive_transform:
            sample['contrastive_image'] = contrastive_image

        return sample

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

    def get_labeled_dataset(self):
        """
        Returns a Subset of the dataset containing only labeled data.
        """
        labeled_indices = [idx for idx in range(len(self.image_files)) 
                           if idx in self.labeled_set]
        return Subset(self, labeled_indices)

    def get_unlabeled_dataset(self):
        """
        Returns a Subset of the dataset containing only unlabeled data.
        """
        unlabeled_indices = [idx for idx in range(len(self.image_files)) 
                             if idx not in self.labeled_set]
        return Subset(self, unlabeled_indices)

    
def get_datasets():
    
        # Define image and mask directories
        image_dirs = [os.path.abspath('data'+'/'+x+'/'+x) for x in os.listdir('data/') if not x.endswith('.txt') and not x.endswith('.m')]
        
        # correct for NA directory
        image_dirs = [path for path in image_dirs if 'NA_Fish_Dataset' not in path]

        mask_dirs = [os.path.abspath('data'+'/'+x+'/'+x +' GT') for x in os.listdir('data/') if not x.endswith('.txt') and not x.endswith('.m')]
        
        mask_dirs = [path for path in mask_dirs if 'NA_Fish_Dataset' not in path]

        # Example transformations (ensure they convert images/masks to tensors correctly)
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()  # Convert mask to tensor directly
        ])
        
        contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomResizedCrop(size=256),
                                            transforms.RandomApply([
                                                transforms.ColorJitter(brightness=0.5,
                                                                        contrast=0.5,
                                                                        saturation=0.5,
                                                                        hue=0.1)
                                            ], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.GaussianBlur(kernel_size=9),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))
                                            ])
        root_dir = 'data/'
        # Initialize the dataset
        dataset = ImageMaskDataset(
                root=root_dir,
                split='train',  # Change as needed: 'train', 'test', 'unlabeled', 'train+unlabeled'
                transform=image_transform,
                target_transform=mask_transform,
                apply_mask=True,
                contrastive_transform=ContrastiveTransformations(contrast_transforms, n_views=2),
                label_fraction=0.7,  # 70% labeled, 30% unlabeled
                download=False,  # Implement download functionality if needed
                seed=123
            )

            
            # Create DataLoaders
        batch_size = 16
        NUM_WORKERS = 4

    # DataLoader for labeled data
        labeled_dataset = dataset.get_labeled_dataset()
        labeled_dataloader = DataLoader(
            labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        # DataLoader for unlabeled data
        unlabeled_dataset = dataset.get_unlabeled_dataset()
        unlabeled_dataloader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        
        print(type(labeled_dataset), type(unlabeled_dataset))
        
        return labeled_dataloader, unlabeled_dataloader
    

if __name__ == '__main__':
    
    try:

        labeled_dataloader, unlabeled_dataloader = get_datasets()
        
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
