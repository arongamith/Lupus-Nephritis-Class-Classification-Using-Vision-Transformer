# Referenced from https://docs.pytorch.org/vision/stable/transforms.html

from torchvision import transforms

# Training transforms: High augmentation to prevent overfitting
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Zoom in on membranes/cells
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(180), # Renal tissue has no orientation
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1), # Focus on texture/shapes
    transforms.RandomApply([ # Reduces reliance on white spots
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing( # Forces model to look at whole
        p=0.3, # glomerulus rather than one
        scale=(0.02, 0.15), # dominant region 
        ratio=(0.3, 3.3), # separate Mesangial 
        value=0 # Proliferative confusion
    ),
])

# Validation/Test transforms
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])