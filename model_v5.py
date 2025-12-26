# model_v5.py - MAXIMUM ACCURACY MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.checkpoint import checkpoint
import numpy as np


class ResNet18TileEncoder(nn.Module):
    """ResNet18-based tile encoder - stronger features."""
    def __init__(self, feature_dim=256, pretrained=True):
        super().__init__()
        
        resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1' if pretrained else None)
        
        # Use all conv layers except the last FC
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Strong projection head
        self.project = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.pool(x).flatten(1)
        return self.project(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and stronger FFN."""
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        
        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))
        return x


class JigsawSolverV5(nn.Module):
    """
    Maximum accuracy jigsaw solver.
    ResNet18 + 4-layer Transformer + Strong head.
    """
    def __init__(self, feature_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Strong tile encoder
        self.tile_encoder = ResNet18TileEncoder(feature_dim)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 9, feature_dim) * 0.02)
        
        # Row/Col embeddings for 2D structure
        self.row_embed = nn.Embedding(3, feature_dim // 2)
        self.col_embed = nn.Embedding(3, feature_dim // 2)
        
        # Deep transformer
        self.transformer = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(feature_dim)
        
        # Strong classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 9)
        )
    
    def _split_to_tiles(self, x):
        B, C, H, W = x.shape
        h = w = 67
        tiles = []
        for i in range(3):
            for j in range(3):
                tiles.append(x[:, :, i*h:(i+1)*h, j*w:(j+1)*w])
        tiles = torch.stack(tiles, dim=1)
        return tiles.view(B * 9, C, h, w), B
    
    def forward(self, x):
        tiles, B = self._split_to_tiles(x)
        
        # Encode tiles
        features = self.tile_encoder(tiles)
        features = features.view(B, 9, -1)
        
        # Add position embeddings
        features = features + self.pos_embed
        
        # Add 2D structure
        positions = torch.arange(9, device=x.device)
        rows = positions // 3
        cols = positions % 3
        row_emb = self.row_embed(rows).unsqueeze(0).expand(B, -1, -1)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(B, -1, -1)
        pos_2d = torch.cat([row_emb, col_emb], dim=-1)
        features = features + pos_2d
        
        # Transformer layers
        for block in self.transformer:
            features = block(features)
        
        features = self.norm(features)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def predict_permutation(self, x):
        from scipy.optimize import linear_sum_assignment
        
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        B = logits.size(0)
        predictions = []
        
        for b in range(B):
            cost = -probs[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            predictions.append(col_ind.tolist())
        
        return predictions


class JigsawSolverV5Lite(nn.Module):
    """
    Memory-efficient version for 4GB GPU.
    Uses gradient checkpointing.
    """
    def __init__(self, feature_dim=224, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # MobileNetV3 Large for better features
        mobilenet = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.IMAGENET1K_V1')
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Project
        self.project = nn.Sequential(
            nn.Linear(960, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 9, feature_dim) * 0.02)
        self.row_embed = nn.Embedding(3, feature_dim // 2)
        self.col_embed = nn.Embedding(3, feature_dim // 2)
        
        # Transformer
        self.transformer = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(feature_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, 9)
        )
        
        self.use_checkpoint = True
    
    def _split_to_tiles(self, x):
        B, C, H, W = x.shape
        h = w = 67
        tiles = []
        for i in range(3):
            for j in range(3):
                tiles.append(x[:, :, i*h:(i+1)*h, j*w:(j+1)*w])
        tiles = torch.stack(tiles, dim=1)
        return tiles.view(B * 9, C, h, w), B
    
    def _encode_tiles(self, tiles):
        feat = self.backbone(tiles)
        feat = self.pool(feat).flatten(1)
        return self.project(feat)
    
    def forward(self, x):
        tiles, B = self._split_to_tiles(x)
        
        # Encode with gradient checkpointing
        if self.training and self.use_checkpoint:
            features = checkpoint(self._encode_tiles, tiles, use_reentrant=False)
        else:
            features = self._encode_tiles(tiles)
        
        features = features.view(B, 9, -1)
        
        # Position embeddings
        features = features + self.pos_embed
        
        positions = torch.arange(9, device=x.device)
        rows = positions // 3
        cols = positions % 3
        row_emb = self.row_embed(rows).unsqueeze(0).expand(B, -1, -1)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(B, -1, -1)
        pos_2d = torch.cat([row_emb, col_emb], dim=-1)
        features = features + pos_2d
        
        # Transformer
        for block in self.transformer:
            if self.training and self.use_checkpoint:
                features = checkpoint(block, features, use_reentrant=False)
            else:
                features = block(features)
        
        features = self.norm(features)
        logits = self.classifier(features)
        
        return logits
    
    def predict_permutation(self, x):
        from scipy.optimize import linear_sum_assignment
        
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        B = logits.size(0)
        predictions = []
        
        for b in range(B):
            cost = -probs[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            predictions.append(col_ind.tolist())
        
        return predictions


def create_model_v5(model_type='lite', feature_dim=224, **kwargs):
    if model_type == 'full':
        return JigsawSolverV5(feature_dim=feature_dim, **kwargs)
    elif model_type == 'lite':
        return JigsawSolverV5Lite(feature_dim=feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
