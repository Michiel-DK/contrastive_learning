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

class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class ImageMaskDataset(Dataset):
    def __init__(
        self, 
        image_dirs, 
        mask_dirs=None, 
        transform=None, 
        mask_transform=None, 
        apply_mask=False, 
        contrastive_transform=None, 
        label_fraction=1.0,
        seed=42
    ):
        """
        Args:
            image_dirs (list): List of directories containing images.
            mask_dirs (list, optional): List of directories containing masks. 
                                        Should correspond to image_dirs if provided.
                                        If None, all images are considered unlabeled.
            transform (callable, optional): Transformations to apply to images.
            mask_transform (callable, optional): Transformations to apply to masks.
            apply_mask (bool, optional): Whether to apply masks to images.
            contrastive_transform (callable, optional): Additional transformations for contrastive learning.
            label_fraction (float, optional): Fraction of data to be labeled (between 0 and 1).
            seed (int, optional): Random seed for reproducibility.
        """
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs if mask_dirs is not None else [None] * len(image_dirs)
        assert len(self.mask_dirs) == len(self.image_dirs), "mask_dirs must be the same length as image_dirs or None."

        self.transform = transform
        self.mask_transform = mask_transform
        self.apply_mask = apply_mask
        self.contrastive_transform = contrastive_transform
        self.label_fraction = label_fraction
        self.seed = seed

        self.image_files = []
        self.mask_files = []

        # Collect image and mask file paths, include only images with masks
        for img_dir, msk_dir in zip(self.image_dirs, self.mask_dirs):
            # List all image files in the image directory
            imgs = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                   glob.glob(os.path.join(img_dir, '*.jpeg')) + \
                   glob.glob(os.path.join(img_dir, '*.png'))
            print(f"Processing Image Directory: {img_dir}")
            print(f"Number of Images Found: {len(imgs)}")

            if msk_dir is not None and os.path.exists(msk_dir):
                # List all mask files in the mask directory
                masks = glob.glob(os.path.join(msk_dir, '*.jpg')) + \
                        glob.glob(os.path.join(msk_dir, '*.jpeg')) + \
                        glob.glob(os.path.join(msk_dir, '*.png'))
                print(f"Processing Mask Directory: {msk_dir}")
                print(f"Number of Masks Found: {len(masks)}")

                # Create a mapping from image stem to mask path
                mask_dict = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}

                # Iterate over each image and find its corresponding mask
                for img_path in imgs:
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path = mask_dict.get(img_name, None)
                    if mask_path:
                        self.image_files.append(img_path)
                        self.mask_files.append(mask_path)
                    else:
                        # Image without corresponding mask is skipped
                        print(f"Skipping Image (No Mask Found): {img_path}")
                        continue
            else:
                # If mask directory doesn't exist, skip adding images without masks
                print(f"Mask Directory Not Found or Not Provided for Image Directory: {img_dir}")
                print("Skipping all images in this directory.")
                continue

        print(f"Total Images with Masks: {len(self.image_files)}")
        print(f"Total Masks: {len(self.mask_files)}")

        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) must match."

        # Set random seed for reproducibility
        random.seed(self.seed)
        # Create indices and shuffle them
        self.indices = list(range(len(self.image_files)))
        random.shuffle(self.indices)

        # Determine number of labeled samples
        labeled_count = int(len(self.image_files) * self.label_fraction)
        self.labeled_set = set(self.indices[:labeled_count])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = self.image_files[actual_idx]
        mask_path = self.mask_files[actual_idx]

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Apply transforms to mask if present
        if self.mask_transform:
            mask = self.mask_transform(mask)

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
            'is_labeled': torch.tensor(is_labeled, dtype=torch.float32)  # 1.0 for labeled, 0.0 for unlabeled
        }

        if self.contrastive_transform:
            sample['contrastive_image'] = contrastive_image

        return sample
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
        
       # import ipdb;ipdb.set_trace()

        return masked_image

    def get_unlabeled_dataset(self):
        """
        Returns a Subset of the dataset containing only unlabeled data.
        """
        unlabeled_indices = [idx for idx in range(len(self.image_files)) 
                             if idx not in self.labeled_set or self.mask_files[idx] is None]
        return Subset(self, unlabeled_indices)

    def get_labeled_dataset(self):
        """
        Returns a Subset of the dataset containing only labeled data.
        """
        labeled_indices = [idx for idx in range(len(self.image_files)) 
                           if idx in self.labeled_set and self.mask_files[idx] is not None]
        return Subset(self, labeled_indices)
    
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
        
        # Initialize the dataset
        dataset = ImageMaskDataset(
            image_dirs=image_dirs,
            mask_dirs=mask_dirs,
            transform=image_transform,
            mask_transform=mask_transform,
            apply_mask=True,
            contrastive_transform=ContrastiveTransformations(contrast_transforms, n_views=2),
            label_fraction=0.7,  # 70% labeled, 30% unlabeled
            seed=123
        )
            
            # Create DataLoaders
        batch_size = 16
        num_workers = 4

        # DataLoader for the entire dataset
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#, num_workers=num_workers)

        # Alternatively, create separate DataLoaders for labeled and unlabeled data
        labeled_dataset = dataset.get_labeled_dataset()
        unlabeled_dataset = dataset.get_unlabeled_dataset()

        labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)#, num_workers=num_workers)
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)#, num_workers=num_workers)
        
        return labeled_dataloader, unlabeled_dataloader
    

if __name__ == '__main__':
    
    try:

        labeled_dataloader, unlabeled_dataloader = get_datasets()
        
        
    except Exception as e:
            import ipdb, traceback, sys
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
