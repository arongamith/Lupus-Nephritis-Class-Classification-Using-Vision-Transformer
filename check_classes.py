from torchvision.datasets import ImageFolder
from transforms import val_transforms

ds = ImageFolder("data/test", transform=val_transforms)
print("Classes:", ds.classes)
print("Class to index:", ds.class_to_idx)