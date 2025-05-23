import os
import random

# === CONFIGURATION ===
DATASET_DIR = "data/food-101/images"
META_DIR = "data/food-101/meta"
TRAIN_TXT = os.path.join(META_DIR, "train.txt")
TEST_TXT = os.path.join(META_DIR, "test.txt")

# === Load original entries
with open(TRAIN_TXT, "r") as f:
    train_lines = set(f.read().splitlines())
with open(TEST_TXT, "r") as f:
    test_lines = set(f.read().splitlines())

existing_lines = train_lines.union(test_lines)

# === Get all folders in dataset
print("🔍 Scanning for new class folders...")
new_train = []
new_test = []

for class_name in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    base_names = [f"{class_name}/{os.path.splitext(f)[0]}" for f in image_files]
    new_items = [x for x in base_names if x not in existing_lines]

    if len(new_items) < 4:
        print(f"⚠️ Skipping '{class_name}' (only {len(new_items)} new images)")
        continue

    random.shuffle(new_items)
    split_idx = int(len(new_items) * 0.75)
    new_train.extend(new_items[:split_idx])
    new_test.extend(new_items[split_idx:])

# === Append and Save
train_lines.update(new_train)
test_lines.update(new_test)

with open(TRAIN_TXT, "w") as f:
    f.write("\n".join(sorted(train_lines)))
with open(TEST_TXT, "w") as f:
    f.write("\n".join(sorted(test_lines)))

print(f"✅ Updated train.txt and test.txt with {len(new_train)} new train and {len(new_test)} new test entries.")
