from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
from torchvision import transforms



class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, apply_mask=False, contrastive_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.mask_transform = mask_transform
        self.apply_mask = apply_mask
        self.contrastive_transform = contrastive_transform


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Load corresponding mask
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale (single channel)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Apply the mask to the image if apply_mask is True
        if self.apply_mask:
            image = self.apply_mask_to_image(image, mask)

        # Apply contrastive transform if provided (e.g., augmentations)
        if self.contrastive_transform:
            # If it's a single image, convert and apply contrastive transform directly
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
                image = self.contrastive_transform(image)
            
            return image, mask

    def apply_mask_to_image(self, image, mask):
        # Ensure the mask is binary (0 or 1)
        mask = (mask > 0).float()

        # Apply the mask to each channel of the image
        masked_image = image.clone()
        for c in range(3):  # For each RGB channel
            masked_image[c, :, :] *= mask[0, :, :]
        
        return masked_image
    
class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


if __name__== '__main__':
    try:
        # Example transformations (ensure they convert images/masks to tensors correctly)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()  # Convert mask to tensor directly
        ])
        
        contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
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
                                          transforms.Normalize((0.5,), (0.5,))
                                         ])

        # Create dataset
        dataset = ImageMaskDataset(
            image_dir='./data/Black Sea Sprat/Black Sea Sprat',
            mask_dir='./data/Black Sea Sprat/Black Sea Sprat GT',
            transform=transform,
            mask_transform=mask_transform,
            contrastive_transform=ContrastiveTransformations(contrast_transforms, n_views=2),
            apply_mask=True  # Set to True to apply the mask during preprocessing
        )

        # DataLoader for batching and shuffling
        data_loader = DataLoader(dataset, batch_size=8, shuffle=True)