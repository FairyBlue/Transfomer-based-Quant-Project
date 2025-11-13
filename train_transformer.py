import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import pandas as pd

from feature_engineering import add_all_features
from label_generator import generate_labels
from timeseries_dataset import SequenceBuilder, LOBSTERSequenceDataset
from transformer_model import TimeSeriesTransformer

# ---------------------
# Utilities
# ---------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_prf(y_true: np.ndarray, y_pred: np.ndarray):
    # binary metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }


def rename_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    # Attempt to map common LOBSTER sample names to our expected schema
    col_map = {}
    if 'AskPrice1' in df.columns and 'BidPrice1' in df.columns:
        col_map.update({
            'AskPrice1': 'ask_price_1',
            'BidPrice1': 'bid_price_1',
            'AskSize1': 'ask_size_1',
            'BidSize1': 'bid_size_1',
        })
    return df.rename(columns=col_map)

# ---------------------
# Training Loop
# ---------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, yh = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=-1)
            ys.append(yb.cpu().numpy())
            yh.append(preds.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yh)
    metrics = metric_prf(y_true, y_pred)
    return total_loss / len(loader.dataset), metrics

# ---------------------
# Main
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to LOBSTER CSV file')
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--ff', type=int, default=256, help='dim_feedforward')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) Load & prepare data
    df = pd.read_csv(args.data)
    df = rename_columns_if_needed(df)
    df = add_all_features(df)
    df = generate_labels(df)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['spread', 'ofi', 'qi', 'momentum', 'mid_price']
    X = df[feature_cols].to_numpy().astype(np.float32)
    y = df['label'].to_numpy().astype(np.int64)

    builder = SequenceBuilder(seq_len=args.seq_len, horizon=args.horizon, step=args.step, zscore=True)
    X = builder.fit_transform(X)
    X_seq, y_seq = builder.build(X, y)

    # Time-based split (no shuffling)
    N = len(X_seq)
    n_val = int(N * args.val_ratio)
    n_train = N - n_val
    train_ds = LOBSTERSequenceDataset(X_seq[:n_train], y_seq[:n_train])
    val_ds = LOBSTERSequenceDataset(X_seq[n_train:], y_seq[n_train:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2) Model
    model = TimeSeriesTransformer(
        num_features=len(feature_cols),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        num_classes=2,
        pooling='last',
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_path = os.path.join(args.results_dir, 'transformer_best.pt')

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': val_loss, **val_metrics})
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | F1={val_metrics['f1']:.4f} | Acc={val_metrics['accuracy']:.4f}")
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'model_state': model.state_dict(),
                'config': {
                    'feature_cols': feature_cols,
                    'seq_len': args.seq_len,
                    'horizon': args.horizon,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'layers': args.layers,
                    'ff': args.ff,
                    'dropout': args.dropout,
                },
                'scaler': {'mean': builder.mean_, 'std': builder.std_},
            }, best_path)
            print(f"Saved best model to {best_path}")

    # Save training history
    with open(os.path.join(args.results_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Final eval print
    print("Training completed. Best F1:", best_f1)

if __name__ == '__main__':
    main()
