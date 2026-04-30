from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transforms import train_transforms

dataset = ImageFolder("data/train", transform=train_transforms)

loader = DataLoader(
    dataset,
    batch_size=4,   
    shuffle=True,
    num_workers=2
)

print("Batch loaded successfully")
