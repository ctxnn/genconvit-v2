# GenConViT — Deepfake Detection with AE, VAE, and Vision Transformers

This repository contains a full PyTorch implementation of the **GenConViT** deepfake detection framework from the paper:

> **GenConViT: Combining Generative and Convolutional Vision Transformers for Deepfake Detection**\
> Zhuangyu Ren et al., 2023\
> [arXiv:2307.07036](https://arxiv.org/pdf/2307.07036) | [Official GitHub](https://github.com/erprogs/GenConViT)

## Overview

GenConViT combines:

- An **AutoEncoder (AE)** branch
- A **Variational AutoEncoder (VAE)** branch
- Two feature extractors: **ConvNeXt** and **Swin Transformer**

These branches reconstruct the input image and extract features from both the reconstructed and original images. A final ensemble classifier uses these features to detect whether an input image is real or fake.

## Features

- AE + VAE dual-branch architecture
- ConvNeXt and Swin Transformer feature extractors
- GELU and ReLU activated classifiers
- Ensemble decision logic for final prediction
- Modular PyTorch codebase with CLI

---

## Project Structure

```
.
├── genconvit.py           # Main script for training/evaluation
├── models.py              # Contains AE, VAE, and GenConViT model definitions
├── dataset.py             # ImageFolder-based dataset loader
├── utils.py               # Training utils, logging, metrics
├── configs.py             # Hyperparameters and CLI config
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/genconvit-from-scratch.git
cd genconvit-from-scratch
```

### 2. Environment

Create a Python environment (using `venv`, `conda`, or `uv`):

```bash
# Using virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Or using uv (if you prefer)
uv venv
uv pip install -r requirements.txt
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Format

Expected folder structure (compatible with `torchvision.datasets.ImageFolder`):

```
/path/to/data/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

Each class contains JPEG or PNG images.

---

## Usage

### Train

```bash
python genconvit.py --mode train --data /path/to/data --epochs 30 --batch-size 16 --lr 1e-4
```

### Evaluate

```bash
python genconvit.py --mode eval --data /path/to/data --save-path genconvit.pth
```

### CLI Options

| Argument       | Description                    | Default         |
| -------------- | ------------------------------ | --------------- |
| `--mode`       | `train` or `eval`              | `train`         |
| `--data`       | Path to dataset directory      | `None`          |
| `--epochs`     | Number of training epochs      | `30`            |
| `--batch-size` | Training batch size            | `16`            |
| `--lr`         | Learning rate                  | `1e-4`          |
| `--save-path`  | Path to save model checkpoints | `genconvit.pth` |

---

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

These are printed after evaluation and optionally saved to a log file.

---

## TODO

-

---

## Citation

If you use this code, please cite the original paper:

```
@article{ren2023genconvit,
  title={GenConViT: Combining Generative and Convolutional Vision Transformers for Deepfake Detection},
  author={Ren, Zhuangyu and others},
  journal={arXiv preprint arXiv:2307.07036},
  year={2023}
}
```

---

## License

MIT License

---

## Contact

For questions, open an issue or contact: [[your-email@example.com](mailto\:your-email@example.com)]

