#References
# All - https://docs.pytorch.org/docs/stable/index.html
# Dataset Handling - https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
# Adam Optimizer - https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
# Learning rate scheduler - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
# Transfer learning / fine-tuning - https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Class imbalance handling - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# Gradient accumulation - https://huggingface.co/docs/transformers/perf_train_gpu_one#gradient-accumulation


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transforms import train_transforms, val_transforms
from model import ViTClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset and Loaders
train_dataset = ImageFolder("data/train", transform=train_transforms)
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = ImageFolder("data/val", transform=val_transforms)
val_loader  = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Train images: {len(train_dataset)}")
print(f"Val images:   {len(val_dataset)}")
print(f"Classes:      {train_dataset.classes}\n")

model = ViTClassifier(num_classes=len(train_dataset.classes))
model.to(device)

epochs             = 30
accumulation_steps = 4 

# Class Weights
# [membranous_pattern, mesangial_hypercellularity, minimal_changes, proliferative]
# Mesangial uses 60.0 instead of actual 86.0 to boost its penalty weight
# and reduce the Mesangial and Proliferative confusion
class_counts  = torch.tensor([71.0, 60.0, 115.0, 110.0])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer - only the classification head is unfrozen initially
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

best_val_acc = 0.0

for epoch in range(epochs):

    # Unfreezing Stratergy
    if epoch == 6:
        print("Unfreezing last 6 transformer blocks for fine-tuning")
        for param in model.vit.blocks[-6:].parameters():
            param.requires_grad = True
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5  # Lower LR for fine-tuning

    # Training
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    scheduler.step()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"- Train Loss: {avg_loss:.4f} "
          f"- Val Acc: {val_acc:.1f}% "
          f"- LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "vit_kidney_classifier.pth")
        print(f" New best model saved ({val_acc:.1f}%)")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")