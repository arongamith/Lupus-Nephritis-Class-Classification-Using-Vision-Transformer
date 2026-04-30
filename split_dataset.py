import os
import shutil
import random

SOURCE_DIR = "data/all"  
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED = 42

random.seed(SEED)

classes = [
    "diffuse_proliferative",
    "membranous_pattern",
    "mesangial_hypercellularity",
    "minimal_changes"
]

print("Starting dataset split...\n")

for class_name in classes:
    src_class = os.path.join(SOURCE_DIR, class_name)

    if not os.path.exists(src_class):
        print(f"SKIP: {src_class} does not exist")
        continue

    # Get all image files
    images = [
        f for f in os.listdir(src_class)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not images:
        print(f"SKIP: No images found in {src_class}")
        continue

    random.shuffle(images)
    total = len(images)

    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    # test gets whatever is left
    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    for split, files in splits.items():
        dest = os.path.join("data", split, class_name)
        os.makedirs(dest, exist_ok=True)

        for f in files:
            src  = os.path.join(src_class, f)
            dst  = os.path.join(dest, f)
            shutil.copy2(src, dst)   # copy2 keeps file metadata

    print(f"{class_name}:")
    print(f"  Total: {total}  →  Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

print("\nDone. Your folders are ready:")
print("  data/train/")
print("  data/val/")
print("  data/test/")
print("\nOriginal images are still in", SOURCE_DIR, "(untouched)")