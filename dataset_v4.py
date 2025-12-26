# dataset_v4.py - ENHANCED DATASET WITH STRONG AUGMENTATION

import os
import re
import json
import random
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np


def _parse_sequence(s):
    
    # Checks if it is list
    if isinstance(s, list):
        return [int(x) for x in s]
    # Checks if it is NaN
    if pd.isna(s):
        return None
    # Applies the String Preprocessing on default
    s_clean = str(s).strip()
    if not s_clean:
        return None
    parts = re.split(r'[,\s]+', s_clean) # Regex Splitting
    result = [int(x) for x in parts if x.strip() != ''] 
    # returns result only if the length is 9
    return result if len(result) == 9 else None


class JigsawAugmentation:
    # Augmentation: To add value to the images 
    
    # Initializes the class
    def __init__(self, training=True):
        self.training = training
    # Defines the Call method
    def __call__(self, img):
        # If not training, return the image as tensor
        if not self.training:
            return TF.to_tensor(img)
        
        # Color augmentation
        if random.random() < 0.8:
            # Color jitter
            # Brightness (+/-30%)
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            # Contrast (+/-30%)
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
            # Saturation (+/-30%)
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
            # Hue (+/-10%)
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
        
        # Grayscale with low probability
        if random.random() < 0.1:
            img = TF.to_grayscale(img, num_output_channels=3)
        
        # Gaussian blur with low probability
        if random.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Sharpness
        if random.random() < 0.3:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.5))
        
        return TF.to_tensor(img)


class PuzzleDatasetV4(Dataset):
    """
    Enhanced dataset for high-accuracy training.
    """
    def __init__(self, image_dir, manifest_path, augment=True, subset_size=None, 
                 seed=42, debug=False):
        self.image_dir = image_dir
        self.augment = augment
        self.debug = debug
        self.samples = []
        
        # Load manifest
        ext = manifest_path.lower().split('.')[-1]
        
        if ext == 'json':
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            for item in data.get('images', []):
                fn = item.get('filename', item.get('image', ''))
                seq = _parse_sequence(item.get('sequence', item.get('label', [])))
                if seq and len(seq) == 9 and sorted(seq) == list(range(9)):
                    self.samples.append((fn, seq))
        else:
            df = pd.read_csv(manifest_path)
            fn_col = 'filename' if 'filename' in df.columns else 'image'
            seq_col = 'sequence' if 'sequence' in df.columns else 'label'
            
            for _, row in df.iterrows():
                fn = row[fn_col]
                seq = _parse_sequence(row[seq_col])
                if seq and len(seq) == 9 and sorted(seq) == list(range(9)):
                    self.samples.append((fn, seq))
        
        print(f"[Dataset V4] Loaded {len(self.samples)} valid samples")
        
        # Debug info
        if debug and len(self.samples) > 0:
            print(f"[Debug] First 3 samples:")
            for fn, seq in self.samples[:3]:
                print(f"  {fn}: {seq}")
        
        # Subset for training
        if subset_size is not None and subset_size < len(self.samples):
            random.seed(seed)
            self.samples = random.sample(self.samples, subset_size)
            print(f"[Dataset V4] Using subset of {len(self.samples)} samples")
        
        # Augmentation
        self.augmentor = JigsawAugmentation(training=augment)
        
        # Normalization (ImageNet stats)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fn, seq = self.samples[idx]
        path = os.path.join(self.image_dir, fn)
        
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = Image.new('RGB', (201, 201), color=(128, 128, 128))
            seq = list(range(9))
        
        # Ensure correct size
        if img.size != (201, 201):
            img = img.resize((201, 201), Image.BICUBIC)
        
        # Apply augmentation
        img_tensor = self.augmentor(img)
        
        # Normalize
        img_tensor = self.normalize(img_tensor)
        
        # Target
        target = torch.tensor(seq, dtype=torch.long)
        
        return img_tensor, target, fn


class PuzzleDatasetV4WithEdges(Dataset):
    """
    Dataset that also returns edge features for auxiliary losses.
    """
    def __init__(self, image_dir, manifest_path, augment=True, subset_size=None, seed=42):
        self.image_dir = image_dir
        self.augment = augment
        self.samples = []
        
        # Load manifest
        df = pd.read_csv(manifest_path)
        fn_col = 'filename' if 'filename' in df.columns else 'image'
        seq_col = 'sequence' if 'sequence' in df.columns else 'label'
        
        for _, row in df.iterrows():
            fn = row[fn_col]
            seq = _parse_sequence(row[seq_col])
            if seq and len(seq) == 9 and sorted(seq) == list(range(9)):
                self.samples.append((fn, seq))
        
        print(f"[Dataset V4 Edges] Loaded {len(self.samples)} samples")
        
        if subset_size is not None and subset_size < len(self.samples):
            random.seed(seed)
            self.samples = random.sample(self.samples, subset_size)
        
        self.augmentor = JigsawAugmentation(training=augment)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Define adjacency pairs (horizontal and vertical)
        self.adj_pairs = []
        for r in range(3):
            for c in range(2):
                self.adj_pairs.append((r * 3 + c, r * 3 + c + 1))  # Horizontal
        for r in range(2):
            for c in range(3):
                self.adj_pairs.append((r * 3 + c, (r + 1) * 3 + c))  # Vertical
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fn, seq = self.samples[idx]
        path = os.path.join(self.image_dir, fn)
        
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (201, 201), color=(128, 128, 128))
            seq = list(range(9))
        
        if img.size != (201, 201):
            img = img.resize((201, 201), Image.BICUBIC)
        
        img_tensor = self.normalize(self.augmentor(img))
        target = torch.tensor(seq, dtype=torch.long)
        
        # Compute adjacency labels based on ground truth
        # Two tiles are "originally adjacent" if they should be adjacent in the solved puzzle
        inv_seq = [0] * 9
        for curr_pos, orig_pos in enumerate(seq):
            inv_seq[orig_pos] = curr_pos
        
        adj_labels = []
        for p1, p2 in self.adj_pairs:
            # p1, p2 are positions in the ORIGINAL grid
            # We need to find which tiles are at these positions in the solved puzzle
            # and check if they're also adjacent in the current (shuffled) arrangement
            curr_tile1 = inv_seq[p1]  # Current position of tile that should be at p1
            curr_tile2 = inv_seq[p2]  # Current position of tile that should be at p2
            
            # Check if these are currently adjacent
            r1, c1 = curr_tile1 // 3, curr_tile1 % 3
            r2, c2 = curr_tile2 // 3, curr_tile2 % 3
            is_adj = (abs(r1 - r2) + abs(c1 - c2)) == 1
            adj_labels.append(1.0 if is_adj else 0.0)
        
        adj_labels = torch.tensor(adj_labels, dtype=torch.float32)
        
        return img_tensor, target, adj_labels, fn
