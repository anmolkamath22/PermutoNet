# train_v5.py - MAXIMUM ACCURACY TRAINING
"""
Pushing for >75% fragment accuracy with:
1. 30000 training samples
2. 40 epochs with cosine annealing
3. Warmup learning rate
4. Mixup augmentation
5. Stronger regularization
6. Gradient checkpointing for memory
"""
import argparse
import os
import gc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

from dataset_v4 import PuzzleDatasetV4
from model_v5 import create_model_v5


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        with torch.no_grad():
            smooth_labels = torch.full_like(log_preds, self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_labels * log_preds).sum(dim=-1).mean()
        return loss


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def decode_to_perm(logits):
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    cost = -probs
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind.tolist()


def compute_metrics(pred_perm, true_perm):
    frag_acc = sum(p == t for p, t in zip(pred_perm, true_perm)) / 9.0
    puzzle_acc = 1.0 if pred_perm == true_perm else 0.0
    
    pred_grid = [0] * 9
    true_grid = [0] * 9
    for tile_id in range(9):
        pred_grid[pred_perm[tile_id]] = tile_id
        true_grid[true_perm[tile_id]] = tile_id
    
    pairs = []
    for r in range(3):
        for c in range(2):
            pairs.append((r * 3 + c, r * 3 + c + 1))
    for r in range(2):
        for c in range(3):
            pairs.append((r * 3 + c, (r + 1) * 3 + c))
    
    paa = sum(1 for a, b in pairs 
              if pred_grid[a] == true_grid[a] and pred_grid[b] == true_grid[b]) / len(pairs)
    
    return frag_acc, puzzle_acc, paa


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    
    total_loss = 0.0
    frag_acc = 0.0
    puzzle_acc = 0.0
    paa = 0.0
    n_samples = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        imgs, targets, _ = batch
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits.reshape(-1, 9), targets.reshape(-1))
            total_loss += loss.item() * imgs.size(0)
        
        logits_np = logits.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        for i in range(imgs.size(0)):
            pred_perm = decode_to_perm(logits_np[i])
            true_perm = targets_np[i].tolist()
            
            fa, pa, adj = compute_metrics(pred_perm, true_perm)
            frag_acc += fa
            puzzle_acc += pa
            paa += adj
            n_samples += 1
    
    return {
        'loss': total_loss / n_samples,
        'frag_acc': frag_acc / n_samples,
        'puzzle_acc': puzzle_acc / n_samples,
        'paa': paa / n_samples
    }


def train_epoch(model, loader, optimizer, scheduler, scaler, device, 
                criterion, use_amp=True, accum_steps=1, use_mixup=True, mixup_alpha=0.2):
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_tiles = 0
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        imgs, targets, _ = batch
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # Apply mixup with some probability
        if use_mixup and random.random() < 0.5:
            imgs, targets_a, targets_b, lam = mixup_data(imgs, targets, mixup_alpha)
            
            with autocast(enabled=use_amp):
                logits = model(imgs)
                loss = mixup_criterion(
                    lambda p, t: criterion(p.reshape(-1, 9), t.reshape(-1)),
                    logits, targets_a, targets_b, lam
                )
                loss = loss / accum_steps
        else:
            with autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits.reshape(-1, 9), targets.reshape(-1))
                loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).sum().item()
            total_correct += correct
            total_tiles += targets.numel()
        
        total_loss += loss.item() * accum_steps * imgs.size(0)
        
        pbar.set_postfix({
            'loss': f"{loss.item() * accum_steps:.4f}",
            'tile_acc': f"{100 * total_correct / total_tiles:.1f}%"
        })
    
    return total_loss / len(loader.dataset), total_correct / total_tiles


def main():
    parser = argparse.ArgumentParser(description='V5 Maximum Accuracy Training')
    
    # Data
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--subset', type=int, default=30000)
    parser.add_argument('--val_split', type=float, default=0.1)
    
    # Model
    parser.add_argument('--model_type', type=str, default='lite', choices=['full', 'lite'])
    parser.add_argument('--feature_dim', type=int, default=224)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # Training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--accum_steps', type=int, default=6)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Output
    parser.add_argument('--out', type=str, default='outputs/best_model_v5.pth')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("JIGSAW PUZZLE SOLVER V5 - MAXIMUM ACCURACY")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Target: >75% Fragment Accuracy")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_mem:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(f"\n{'='*60}")
    print("Loading Dataset...")
    print(f"{'='*60}")
    
    full_dataset = PuzzleDatasetV4(
        args.image_dir, args.manifest, augment=True, 
        subset_size=args.subset, debug=args.debug
    )
    
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print(f"\n{'='*60}")
    print("Creating Model...")
    print(f"{'='*60}")
    
    model = create_model_v5(
        model_type=args.model_type,
        feature_dim=args.feature_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: V5 {args.model_type}")
    print(f"Parameters: {total_params:,}")
    print(f"Feature dim: {args.feature_dim}")
    print(f"Layers: {args.num_layers} | Heads: {args.num_heads}")
    
    # Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    
    # Optimizer with warmup
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    steps_per_epoch = len(train_loader) // args.accum_steps
    total_steps = steps_per_epoch * args.epochs
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    scaler = GradScaler(enabled=not args.no_amp)
    
    # Training
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch_size} x {args.accum_steps} = {args.batch_size * args.accum_steps}")
    print(f"LR: {args.lr} | Mixup: {args.mixup_alpha}")
    
    best_frag_acc = 0.0
    patience_counter = 0
    history = {'frag_acc': [], 'puzzle_acc': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'─'*60}")
        
        train_loss, train_tile_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            criterion, use_amp=not args.no_amp, accum_steps=args.accum_steps,
            use_mixup=True, mixup_alpha=args.mixup_alpha
        )
        
        val_metrics = evaluate(model, val_loader, device, use_amp=not args.no_amp)
        
        history['frag_acc'].append(val_metrics['frag_acc'])
        history['puzzle_acc'].append(val_metrics['puzzle_acc'])
        
        print(f"\n📊 RESULTS:")
        print(f"   Train: Loss={train_loss:.4f} | Tile Acc={100*train_tile_acc:.2f}%")
        print(f"   Val:   Loss={val_metrics['loss']:.4f}")
        print(f"   🎯 Fragment Accuracy: {100*val_metrics['frag_acc']:.2f}%")
        print(f"   🧩 Puzzle Accuracy:   {100*val_metrics['puzzle_acc']:.2f}%")
        print(f"   🔗 Adjacency (PAA):   {100*val_metrics['paa']:.2f}%")
        
        if val_metrics['frag_acc'] > best_frag_acc:
            best_frag_acc = val_metrics['frag_acc']
            patience_counter = 0
            
            os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_frag_acc': best_frag_acc,
                'metrics': val_metrics,
                'args': vars(args)
            }, args.out)
            print(f"   ✓ NEW BEST! Saved (Frag Acc: {100*best_frag_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                print(f"\n⚠ Early stopping")
                break
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"🏆 Best Fragment Accuracy: {100*best_frag_acc:.2f}%")
    print(f"📁 Model: {args.out}")


if __name__ == '__main__':
    main()
