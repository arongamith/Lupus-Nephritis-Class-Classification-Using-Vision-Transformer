from torchvision.datasets import ImageFolder

dataset = ImageFolder("data/train")

print("Classes:", dataset.classes)
print("Number of images:", len(dataset))
