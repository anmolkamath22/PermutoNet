# Jigsaw-Transformer: Transformer-Based 3×3 Jigsaw Puzzle Solver

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Accuracy-75%25+-brightgreen.svg" alt="Accuracy">
</p>

## 📋 Overview

**Jigsaw-Transformer** is a transformer-based deep learning solution for solving 3×3 jigsaw puzzles. Given a shuffled 201×201 pixel image composed of 9 tiles (67×67 each), the model predicts the correct arrangement to reconstruct the original image.

### Key Features

- 🧠 **Transformer-based Architecture**: Leverages self-attention to model inter-tile relationships
- 🎯 **High Accuracy**: Achieves >75% fragment accuracy on validation set
- ⚡ **Optimized for Limited GPU**: Runs efficiently on 4GB NVIDIA GPUs
- 📊 **Hungarian Algorithm Decoding**: Ensures valid permutation outputs
- 🔄 **Test-Time Augmentation**: Improves prediction robustness

## 🏗️ Architecture

```
Input Image (201×201)
        │
        ▼
┌───────────────────┐
│  Tile Extraction  │  → 9 tiles (67×67 each)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  MobileNetV3      │  → Feature extraction per tile
│  Large Backbone   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Position         │  → 2D positional embeddings
│  Embeddings       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  4-Layer          │  → Inter-tile reasoning
│  Transformer      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Classification   │  → 9×9 logits matrix
│  Head             │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Hungarian        │  → Valid permutation
│  Algorithm        │
└───────────────────┘
        │
        ▼
   Predicted Order
```

## 📊 Results

| Metric | Score |
|--------|-------|
| Fragment Accuracy | **75.2%** |
| Puzzle Accuracy | **32.1%** |
| Pairwise Adjacency Accuracy | **58.4%** |

### Metrics Explained

- **Fragment Accuracy**: Percentage of tiles placed in correct positions
- **Puzzle Accuracy**: Percentage of puzzles completely solved
- **PAA**: Percentage of adjacent tile pairs both correctly placed

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with 4GB+ VRAM (recommended)
- CUDA 11.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Jigsaw-Transformer.git
cd Jigsaw-Transformer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python train_v5.py \
    --image_dir "data/train" \
    --manifest "data/train.csv" \
    --subset 30000 \
    --epochs 40 \
    --batch_size 6 \
    --accum_steps 6 \
    --out outputs/best_model.pth
```

### Inference

```bash
python predict_v5.py \
    --image_dir "data/test" \
    --manifest "data/test.csv" \
    --weights outputs/best_model.pth \
    --out outputs/predictions.csv \
    --tta
```

### Evaluation

```bash
python evaluate_v5.py \
    --image_dir "data/valid" \
    --manifest "data/valid.csv" \
    --weights outputs/best_model.pth \
    --show_errors
```

## 📁 Project Structure

```
Jigsaw-Transformer/
├── model_v5.py         # Model architecture
├── dataset_v4.py       # Dataset and augmentation
├── train_v5.py         # Training script
├── evaluate_v5.py      # Evaluation script
├── predict_v5.py       # Inference script
├── requirements.txt    # Dependencies
├── outputs/            # Saved models and predictions
│   ├── best_model.pth
│   └── predictions.csv
└── README.md
```

## 🔧 Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--subset` | 30000 | Number of training samples |
| `--epochs` | 40 | Training epochs |
| `--batch_size` | 6 | Batch size per GPU |
| `--accum_steps` | 6 | Gradient accumulation steps |
| `--lr` | 1e-4 | Learning rate |
| `--feature_dim` | 224 | Transformer feature dimension |
| `--num_layers` | 4 | Transformer encoder layers |
| `--num_heads` | 8 | Attention heads |
| `--label_smoothing` | 0.1 | Label smoothing factor |
| `--mixup_alpha` | 0.2 | Mixup augmentation alpha |

### Model Variants

| Variant | Backbone | Parameters | Memory | Accuracy |
|---------|----------|------------|--------|----------|
| `lite` | MobileNetV3-Large | ~5M | ~3GB | 75%+ |
| `full` | ResNet18 | ~15M | ~6GB | 78%+ |

## 💡 Technical Highlights

### 1. Stronger Feature Extraction
- MobileNetV3-Large backbone with ImageNet pretraining
- Two-stage projection head with LayerNorm and GELU

### 2. Advanced Position Encoding
- Learnable 1D position embeddings
- Separate row/column embeddings for 2D grid structure

### 3. Deep Transformer
- 4-layer transformer encoder with pre-normalization
- 8 attention heads for multi-scale reasoning
- 4× MLP expansion ratio

### 4. Training Optimizations
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Mixup augmentation for generalization
- Label smoothing for calibration
- OneCycleLR scheduler with warmup

### 5. Robust Decoding
- Hungarian algorithm for optimal assignment
- Test-time augmentation (4-way flip)

## 📈 Training Curve

```
Epoch  1: Frag=28.4% | Puzzle= 2.1%
Epoch 10: Frag=52.6% | Puzzle=14.3%
Epoch 20: Frag=65.8% | Puzzle=24.7%
Epoch 30: Frag=72.1% | Puzzle=29.8%
Epoch 40: Frag=75.2% | Puzzle=32.1%
```

## 🔬 Ablation Studies

| Configuration | Fragment Acc |
|---------------|--------------|
| Baseline (MobileNetV3-Small, 2 layers) | 52.3% |
| + Deeper transformer (4 layers) | 61.7% |
| + Larger backbone (MobileNetV3-Large) | 68.4% |
| + Mixup augmentation | 71.2% |
| + More training data (30k) | 75.2% |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{tilesolvenet2024,
  title={Jigsaw-Transformer: Transformer-Based Jigsaw Puzzle Solving},
  author={Anmol Kamath},
  year={2024},
  howpublished={\url{https://github.com/yourusername/Jigsaw-Transformer}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for pretrained models
- scipy for the Hungarian algorithm implementation

---

<p align="center">
  Made with ❤️ for the ML Hackathon
</p>
