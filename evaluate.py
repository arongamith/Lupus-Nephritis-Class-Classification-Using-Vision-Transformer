import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ViTClassifier
from transforms import val_transforms

# ── SETTINGS ──
TEST_DIR  = "data/test"
WEIGHTS   = "vit_kidney_classifier.pth"

# Must match alphabetical folder order exactly:
# 0=diffuse_proliferative, 1=membranous_pattern,
# 2=mesangial_hypercellularity, 3=minimal_changes
CLASS_NAMES = [
    "Proliferative",
    "Membranous Pattern",
    "Mesangial Hypercellularity",
    "Minimal Changes"
]

DISPLAY_NAMES = [
    "Proliferative",
    "Membranous\nPattern",
    "Mesangial\nHypercellularity",
    "Minimal\nChanges"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluating on: {device}")

# ── LOAD MODEL ──
model = ViTClassifier(num_classes=4)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.to(device)
model.eval()

# ── LOAD TEST DATA ──
test_dataset = ImageFolder(TEST_DIR, transform=val_transforms)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"Test images found: {len(test_dataset)}")
print(f"Classes (index order): {test_dataset.classes}\n")

# ── RUN EVALUATION ──
all_preds   = []
all_labels  = []
all_probs   = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs   = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── ACCURACY ──
overall_acc = 100 * np.mean(all_preds == all_labels)
print(f"Overall Accuracy: {overall_acc:.2f}%\n")

# ── PRECISION / RECALL / F1 ──
precision = precision_score(all_labels, all_preds, average='weighted')
recall    = recall_score(all_labels, all_preds, average='weighted')
f1        = f1_score(all_labels, all_preds, average='weighted')

print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Weighted F1 Score:  {f1:.4f}\n")

# ── PER CLASS REPORT ──
print("Per-class Classification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES
))

# ── CONFUSION MATRIX PLOT ──
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=DISPLAY_NAMES,
    yticklabels=DISPLAY_NAMES,
    linewidths=0.5, linecolor="#cccccc", ax=ax
)
ax.set_xlabel("Predicted Class", fontsize=12, labelpad=12)
ax.set_ylabel("Actual Class",    fontsize=12, labelpad=12)
ax.set_title("Confusion Matrix — Lupus Glomerulus ViT Classifier", fontsize=13, pad=16)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Confusion matrix saved to: confusion_matrix.png")
plt.show()

# ── ROC / AUC CURVE ──
labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
colors     = ['#D85A30', '#1D9E75', '#7F77DD', '#EF9F27']

fig, ax = plt.subplots(figsize=(8, 6))

auc_scores = []
for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{cls_name} (AUC = {roc_auc:.2f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — Lupus Glomerulus ViT Classifier", fontsize=13, pad=16)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
print("ROC curve saved to: roc_curve.png")
plt.show()

print(f"\nMacro-average AUC: {np.mean(auc_scores):.4f}")
print("\nPer-class AUC:")
for name, score in zip(CLASS_NAMES, auc_scores):
    print(f"  {name}: {score:.4f}")