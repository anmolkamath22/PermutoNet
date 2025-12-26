# evaluate_v5.py - Evaluation for V5 models
import argparse
import os
import re
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from model_v5 import create_model_v5


def _parse_sequence(s):
    if isinstance(s, list):
        return [int(x) for x in s]
    if pd.isna(s):
        return None
    s_clean = str(s).strip()
    if not s_clean:
        return None
    parts = re.split(r'[,\s]+', s_clean)
    result = [int(x) for x in parts if x.strip() != '']
    return result if len(result) == 9 else None


def load_samples(manifest_path):
    samples = []
    df = pd.read_csv(manifest_path)
    fn_col = 'filename' if 'filename' in df.columns else 'image'
    seq_col = 'sequence' if 'sequence' in df.columns else 'label'
    
    for _, row in df.iterrows():
        fn = row[fn_col]
        seq = _parse_sequence(row[seq_col])
        if seq and len(seq) == 9:
            samples.append((fn, seq))
    return samples


def decode_logits_to_perm(logits):
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    cost = -probs
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--show_errors', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("JIGSAW PUZZLE EVALUATION V5")
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
    print(f"✓ Model loaded (Training best: {100*checkpoint.get('best_frag_acc', 0):.2f}%)")
    
    samples = load_samples(args.manifest)
    print(f"Samples: {len(samples)}")
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frag_acc_total = 0.0
    puzzle_acc_total = 0.0
    paa_total = 0.0
    n = 0
    
    errors = []
    
    for fn, true_perm in tqdm(samples, desc="Evaluating"):
        img_path = os.path.join(args.image_dir, fn)
        if not os.path.exists(img_path):
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            if img.size != (201, 201):
                img = img.resize((201, 201), Image.BICUBIC)
            
            x = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(x)
            
            pred_perm = decode_logits_to_perm(logits.cpu().numpy()[0])
            
            frag = sum(p == t for p, t in zip(pred_perm, true_perm)) / 9.0
            puzzle = 1.0 if pred_perm == true_perm else 0.0
            
            # PAA
            pred_grid = [0] * 9
            true_grid = [0] * 9
            for i in range(9):
                pred_grid[pred_perm[i]] = i
                true_grid[true_perm[i]] = i
            pairs = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(0,3),(3,6),(1,4),(4,7),(2,5),(5,8)]
            paa = sum(1 for a,b in pairs if pred_grid[a]==true_grid[a] and pred_grid[b]==true_grid[b]) / len(pairs)
            
            frag_acc_total += frag
            puzzle_acc_total += puzzle
            paa_total += paa
            n += 1
            
            if puzzle < 1.0 and len(errors) < 5:
                errors.append({'file': fn, 'pred': pred_perm, 'true': true_perm, 'frag': frag})
        except Exception as e:
            print(f"Error {fn}: {e}")
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"📊 Fragment Accuracy: {100*frag_acc_total/n:.2f}%")
    print(f"🧩 Puzzle Accuracy:   {100*puzzle_acc_total/n:.2f}%")
    print(f"🔗 PAA (Adjacency):   {100*paa_total/n:.2f}%")
    
    if args.show_errors and errors:
        print(f"\n🔍 Errors:")
        for e in errors:
            print(f"   {e['file']}: Pred={e['pred']} True={e['true']} Frag={100*e['frag']:.1f}%")
    print()


if __name__ == '__main__':
    main()
