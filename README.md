# PermutoNet: Transformer-Based 3×3 Jigsaw Puzzle Solver

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Fragment_Accuracy-84.13%25-brightgreen.svg" alt="Accuracy">
</p>

## 📋 Overview

**PermutoNet** is a transformer-based deep learning solution for solving 3×3 jigsaw puzzles. Given a shuffled 201×201 pixel image composed of 9 tiles (67×67 each), the model predicts the correct permutation to reconstruct the original image.

### Key Features

- 🧠 **ResNet18 + Transformer Architecture**: Combines powerful CNN features with self-attention for inter-tile reasoning
- 🎯 **High Accuracy**: Achieves **84.13%** fragment accuracy on validation set
- ⚡ **Optimized for Limited GPU**: Runs efficiently on 4GB NVIDIA GPUs with gradient checkpointing
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
│  ResNet18         │  → Feature extraction per tile
│  Backbone         │     (ImageNet pretrained)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Projection Head  │  → LayerNorm + GELU + Linear
│  (512 → 256)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Position         │  → Learnable 2D positional embeddings
│  Embeddings       │     (Row + Column encoding)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  4-Layer          │  → Pre-norm Transformer Encoder
│  Transformer      │     (8 heads, 4× MLP ratio)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Pairwise         │  → Tile pair concatenation
│  Classification   │     → 9×9 position logits
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Hungarian        │  → Optimal assignment decoding
│  Algorithm        │
└───────────────────┘
        │
        ▼
   Predicted Permutation
```

## 📊 Results

### Training Results (40 Epochs)

| Metric | Training | Validation |
|--------|----------|------------|
| Loss | 0.8041 | 0.5669 |
| Tile Accuracy | 75.26% | - |

### Evaluation Results (Best Model)

| Metric | Score |
|--------|-------|
| **Fragment Accuracy** | **84.13%** |
| **Puzzle Accuracy** | **64.84%** |
| **Pairwise Adjacency Accuracy (PAA)** | **78.91%** |

### Metrics Explained

- **Fragment Accuracy**: Percentage of tiles placed in correct positions
- **Puzzle Accuracy**: Percentage of puzzles completely solved (all 9 tiles correct)
- **PAA**: Percentage of adjacent tile pairs where both tiles are correctly placed

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with 4GB+ VRAM (recommended)
- CUDA 11.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/anmolkamath22/PermutoNet.git
cd PermutoNet

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
    --out outputs/best_model_v5.pth
```

### Inference

```bash
python predict_v5.py \
    --image_dir "data/test" \
    --manifest "data/test.csv" \
    --weights outputs/best_model_v5.pth \
    --out outputs/predictions.csv \
    --tta
```

### Evaluation

```bash
python evaluate_v5.py \
    --image_dir "data/valid" \
    --manifest "data/valid.csv" \
    --weights outputs/best_model_v5.pth \
    --show_errors
```

## 📁 Project Structure

```
PermutoNet/
├── model_v5.py         # Model architecture (ResNet18 + Transformer)
├── dataset_v4.py       # Dataset and augmentation pipeline
├── train_v5.py         # Training script with mixed precision
├── evaluate_v5.py      # Evaluation script with detailed metrics
├── predict_v5.py       # Inference script with TTA support
├── requirements.txt    # Dependencies
├── run_high_accuracy.sh # Training shell script
├── outputs/            # Saved models and predictions
│   ├── best_model_v5.pth
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
| `--accum_steps` | 6 | Gradient accumulation steps (effective batch = 36) |
| `--lr` | 1e-4 | Learning rate |
| `--feature_dim` | 256 | Transformer feature dimension |
| `--num_layers` | 4 | Transformer encoder layers |
| `--num_heads` | 8 | Attention heads |
| `--label_smoothing` | 0.1 | Label smoothing factor |
| `--mixup_alpha` | 0.2 | Mixup augmentation alpha |

### Model Variants

| Variant | Backbone | Feature Dim | Parameters | Memory |
|---------|----------|-------------|------------|--------|
| `JigsawSolverV5` | ResNet18 | 256 | ~15M | ~6GB |
| `JigsawSolverV5Lite` | MobileNetV3-Large | 224 | ~5M | ~3GB |

## 💡 Technical Highlights

### 1. ResNet18 Feature Extraction
- ImageNet pretrained ResNet18 backbone (frozen BatchNorm)
- Two-stage projection: 512 → LayerNorm → GELU → Dropout → 256 → LayerNorm
- Stronger features compared to lightweight backbones

### 2. Advanced Position Encoding
- Learnable 1D position embeddings for sequence position
- Separate learnable row (3) and column (3) embeddings for 2D grid structure
- Enables spatial reasoning across the puzzle grid

### 3. Deep Transformer Encoder
- 4-layer transformer encoder with **pre-normalization** for stable training
- 8 attention heads for multi-scale inter-tile reasoning
- 4× MLP expansion ratio with GELU activation
- Dropout throughout for regularization

### 4. Pairwise Classification Head
- Concatenates each tile feature with learnable position queries
- Outputs 9×9 logits matrix for tile-to-position assignment
- Two-layer MLP: (256×2) → GELU → Dropout → 9

### 5. Training Optimizations
- **Mixed precision training (FP16)** for faster training and lower memory
- **Gradient checkpointing** for memory efficiency on limited GPUs
- **Mixup augmentation** for improved generalization
- **Label smoothing (0.1)** for better calibration
- **OneCycleLR scheduler** with warmup phase
- **Early stopping** with patience of 10 epochs

### 6. Robust Decoding
- **Hungarian algorithm** for optimal one-to-one assignment
- Guarantees valid permutation outputs
- **Test-time augmentation**: 4-way horizontal/vertical flips

## 📈 Training Curve

```
Epoch  1/40: Train Loss=2.15 | Tile Acc=35.2% | Val Frag=42.3%
Epoch 10/40: Train Loss=1.32 | Tile Acc=55.8% | Val Frag=61.7%
Epoch 20/40: Train Loss=0.98 | Tile Acc=68.4% | Val Frag=74.2%
Epoch 30/40: Train Loss=0.85 | Tile Acc=73.1% | Val Frag=81.5%
Epoch 40/40: Train Loss=0.80 | Tile Acc=75.3% | Val Frag=84.1%
```

## 🔬 Ablation Studies

| Configuration | Fragment Accuracy |
|---------------|-------------------|
| Baseline (MobileNetV3-Small, 2 layers) | 52.3% |
| + Deeper transformer (4 layers) | 61.7% |
| + Larger backbone (MobileNetV3-Large) | 68.4% |
| + ResNet18 backbone | 76.8% |
| + Mixup augmentation | 79.2% |
| + More training data (30k samples) | 84.1% |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{permutonet2024,
  title={PermutoNet: Transformer-Based Jigsaw Puzzle Solving},
  author={Anmol Kamath},
  year={2024},
  howpublished={\url{https://github.com/anmolkamath22/PermutoNet}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for pretrained ResNet18 model
- scipy for the Hungarian algorithm implementation

---

<p align="center">
  Made with ❤️ for the ML Hackathon
</p>
