# predict_v5.py - Inference for V5 models
import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

from model_v5 import create_model_v5


def load_filenames(manifest_path):
    df = pd.read_csv(manifest_path)
    fn_col = 'filename' if 'filename' in df.columns else 'image'
    return df[fn_col].tolist()


def decode_logits_to_perm(logits):
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    cost = -probs
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind.tolist()


def tta_augment(model, device, img, transform, use_tta=True):
    def get_logits(image):
        x = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
        return logits.cpu().numpy()[0]
    
    if not use_tta:
        return get_logits(img)
    
    all_logits = [get_logits(img)]
    
    # Horizontal flip
    hflip_map = [2, 1, 0, 5, 4, 3, 8, 7, 6]
    hf_logits = get_logits(ImageOps.mirror(img))
    remapped = np.zeros_like(hf_logits)
    for i in range(9):
        for j in range(9):
            remapped[hflip_map[i], hflip_map[j]] = hf_logits[i, j]
    all_logits.append(remapped)
    
    # Vertical flip
    vflip_map = [6, 7, 8, 3, 4, 5, 0, 1, 2]
    vf_logits = get_logits(ImageOps.flip(img))
    remapped = np.zeros_like(vf_logits)
    for i in range(9):
        for j in range(9):
            remapped[vflip_map[i], vflip_map[j]] = vf_logits[i, j]
    all_logits.append(remapped)
    
    # Both
    hvflip_map = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    hvf_logits = get_logits(ImageOps.flip(ImageOps.mirror(img)))
    remapped = np.zeros_like(hvf_logits)
    for i in range(9):
        for j in range(9):
            remapped[hvflip_map[i], hvflip_map[j]] = hvf_logits[i, j]
    all_logits.append(remapped)
    
    return np.mean(all_logits, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("JIGSAW PUZZLE INFERENCE V5")
    print(f"{'='*60}")
    
    checkpoint = torch.load(args.weights, map_location='cpu')
    saved_args = checkpoint.get('args', {})
    
    model = create_model_v5(
        model_type=saved_args.get('model_type', 'lite'),
        feature_dim=saved_args.get('feature_dim', 224),
        num_layers=saved_args.get('num_layers', 4),
        num_heads=saved_args.get('num_heads', 8)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    filenames = load_filenames(args.manifest)
    print(f"Images: {len(filenames)}")
    
    results = []
    for fn in tqdm(filenames, desc="Inference"):
        img_path = os.path.join(args.image_dir, fn)
        if not os.path.exists(img_path):
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            if img.size != (201, 201):
                img = img.resize((201, 201), Image.BICUBIC)
            
            logits = tta_augment(model, device, img, transform, use_tta=args.tta)
            perm = decode_logits_to_perm(logits)
            results.append((fn, perm))
        except Exception as e:
            print(f"Error {fn}: {e}")
    
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    
    df = pd.DataFrame({
        'image': [r[0] for r in results],
        'label': [' '.join(map(str, r[1])) for r in results]
    })
    df.to_csv(args.out, index=False)
    print(f"✓ Saved {len(results)} predictions to {args.out}")


if __name__ == '__main__':
    main()
