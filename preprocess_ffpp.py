# preprocess_ffpp.py
import os
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ===== CONFIG =====
SOURCE = Path.home() / "Desktop/Rijul/DeepfakeBench/datasets/rgb/FaceForensics++"
TARGET = Path("preprocessed_ffpp")
FRAME_INTERVAL = 5  # sample every Nth frame
RESIZE = (128, 128)  # output size

# Splits and labels
SPLITS = ["train", "val", "test"]
LABELS = {"original": "real", "manipulated": "fake"}

# Manipulation folders to include
MANIPS = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]

# ===== SETUP OUTPUT DIRS =====
for split in SPLITS:
    for label in LABELS.values():
        (TARGET / split / label).mkdir(parents=True, exist_ok=True)

# ===== LOAD SPLITS =====
splits_map = {}
for split in SPLITS:
    path = SOURCE / f"{split}.json"
    with open(path, 'r') as f:
        data = json.load(f)
    for item in data:
        # JSON should have a field 'video' containing the video folder name (e.g. '000_003')
        vid = Path(str(item.get("video", item.get("file", "")))).stem
        splits_map[vid] = split

print(f"Total videos mapped: {len(splits_map)}")

# ===== PROCESS SEQUENCES =====
def process_sequence(tree_root: Path, label_key: str):
    for attack_subdir in tree_root.iterdir():
        if not attack_subdir.is_dir(): continue
        # dive into quality folders like 'c23', 'c40', ...
        for quality in attack_subdir.iterdir():
            frames_base = quality / "frames"
            if not frames_base.exists(): continue
            for vid_dir in frames_base.iterdir():
                if not vid_dir.is_dir(): continue
                vid_name = vid_dir.name
                split = splits_map.get(vid_name)
                if split is None:
                    continue
                target_dir = TARGET / split / LABELS[label_key]
                for i, imgfile in enumerate(sorted(vid_dir.iterdir())):
                    if not (imgfile.suffix.lower() in [".jpg", ".png"]): continue
                    if i % FRAME_INTERVAL != 0: continue
                    # load, resize, and save
                    img = Image.open(imgfile).convert("RGB")
                    img = img.resize(RESIZE)
                    dst = target_dir / f"{vid_name}_{imgfile.name}"
                    img.save(dst, quality=95)

# Originals
print("Processing originals...")
process_sequence(SOURCE / "original_sequences", "original")

# Manipulated
print("Processing manipulated sequences...")
process_sequence(SOURCE / "manipulated_sequences", "manipulated")

print(" Preprocessing finished.")
print(f"Check your folder: {TARGET}")
