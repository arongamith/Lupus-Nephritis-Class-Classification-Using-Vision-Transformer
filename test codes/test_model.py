import torch
from model import ViTClassifier

num_classes = 4
model = ViTClassifier(num_classes)

x = torch.randn(1, 3, 224, 224)
y = model(x)

print("Output shape:", y.shape)
