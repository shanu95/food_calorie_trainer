import os
import json
from collections import defaultdict

# === Paths ===
META_DIR = "data/food-101/meta"
TRAIN_TXT = os.path.join(META_DIR, "train.txt")
TEST_TXT = os.path.join(META_DIR, "test.txt")
TRAIN_JSON = os.path.join(META_DIR, "train.json")
TEST_JSON = os.path.join(META_DIR, "test.json")

def convert_txt_to_json(txt_path, json_path):
    data = defaultdict(list)
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if "/" in line:
                class_name, img_id = line.split("/")
                data[class_name].append(line)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Created {json_path} with {len(data)} classes")

# === Run both
convert_txt_to_json(TRAIN_TXT, TRAIN_JSON)
convert_txt_to_json(TEST_TXT, TEST_JSON)
