# tal_for_har/train_tinyhar_wear_metrics.py
import os, sys, json, time, random
from collections import Counter
import numpy as np
import pandas as pd
import glob


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)


HERE = os.path.dirname(__file__)
sys.path.append(os.path.join(HERE, "inertial_baseline"))
from inertial_baseline.TinyHAR import TinyHAR


DATASET = "wear"
RAW_ROOT  = os.path.join(HERE, "data", DATASET, "raw", "inertial")
SAVE_ROOT = os.path.join(HERE, "data", DATASET, "metrics")
os.makedirs(SAVE_ROOT, exist_ok=True)

SUBJECTS   = list(range(6))   # sbj_0..sbj_5
WINDOW_SIZE = 50              # samples
OVERLAP     = 0.5             # 50% 
BATCH_SIZE  = 100
EPOCHS      = 30
LR          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 1

# label maps (same as your data_creation.py; train WITHOUT 'null')
LABEL_DICT = {
    'null': 0,
    'jogging': 1, 'jogging (rotating arms)': 2, 'jogging (skipping)': 3,
    'jogging (sidesteps)': 4, 'jogging (butt-kicks)': 5,
    'stretching (triceps)': 6, 'stretching (lunging)': 7,
    'stretching (shoulders)': 8, 'stretching (hamstrings)': 9,
    'stretching (lumbar rotation)': 10, 'push-ups': 11, 'push-ups (complex)': 12,
    'sit-ups': 13, 'sit-ups (complex)': 14, 'burpees': 15,
    'lunges': 16, 'lunges (complex)': 17, 'bench-dips': 18
}
LABEL_DICT_NO_NULL = {
    'jogging': 0, 'jogging (rotating arms)': 1, 'jogging (skipping)': 2,
    'jogging (sidesteps)': 3, 'jogging (butt-kicks)': 4,
    'stretching (triceps)': 5, 'stretching (lunging)': 6,
    'stretching (shoulders)': 7, 'stretching (hamstrings)': 8,
    'stretching (lumbar rotation)': 9, 'push-ups': 10, 'push-ups (complex)': 11,
    'sit-ups': 12, 'sit-ups (complex)': 13, 'burpees': 14,
    'lunges': 15, 'lunges (complex)': 16, 'bench-dips': 17
}
IDX2CLASS = {v: k for k, v in LABEL_DICT_NO_NULL.items()}
NUM_CLASSES = len(LABEL_DICT_NO_NULL)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_subject_csv(sid: int) -> pd.DataFrame:
    path = os.path.join(RAW_ROOT, f"sbj_{sid}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)

def df_to_numeric(df: pd.DataFrame):
    df2 = df.copy()
    df2.iloc[:, -1] = df2.iloc[:, -1].map(LABEL_DICT).fillna(0).astype(int)
    arr = df2.to_numpy()
    X = arr[:, :-1].astype(np.float32)
    y = arr[:, -1].astype(np.int64)
    return X, y

def make_windows(X, y, win, overlap):
    step = max(1, int(win * (1.0 - overlap)))
    n = len(X)
    if n < win:
        return np.empty((0, win, X.shape[1]), np.float32), np.empty((0,), np.int64)

    wins, labels = [], []
    i = 0
    while i + win <= n:
        seg_X = X[i:i+win]
        seg_y = y[i:i+win]
        nnz = seg_y[seg_y != 0]
        if nnz.size > 0:
            maj_full = Counter(nnz).most_common(1)[0][0]
            label_str = next((k for k, v in LABEL_DICT.items() if v == maj_full), None)
            if label_str and label_str != 'null':
                wins.append(seg_X)
                labels.append(LABEL_DICT_NO_NULL[label_str])
        i += step

    if not wins:
        return np.empty((0, win, X.shape[1]), np.float32), np.empty((0,), np.int64)
    return np.stack(wins), np.array(labels, np.int64)

def build_subject_windows(sid: int):
    df = load_subject_csv(sid)
    X, y = df_to_numeric(df)
    return make_windows(X, y, WINDOW_SIZE, OVERLAP)

class WearDataset(Dataset):
    def __init__(self, subject_ids):
        Xs, ys = [], []
        for sid in subject_ids:
            Xw, yw = build_subject_windows(sid)
            if Xw.shape[0] > 0:
                Xs.append(Xw); ys.append(yw)
        if not Xs:
            raise ValueError("No windows produced; check labels/window params.")
        self.X = torch.from_numpy(np.concatenate(Xs, 0)).float()  # [N,L,C]
        self.y = torch.from_numpy(np.concatenate(ys, 0)).long()

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward(); opt.step()

        loss_sum += loss.item() * xb.size(0)
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += xb.size(0)
    return loss_sum/total, correct/total, time.perf_counter() - t0

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true_all, y_pred_all = [], []
    t0 = time.perf_counter()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)

        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

        y_true_all.append(yb.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())
    dur = time.perf_counter() - t0

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=int)

    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weight = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_mac  = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    rec_mac   = recall_score(y_true, y_pred, average="macro",       zero_division=0)
    cm        = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    return (loss_sum/total, acc, f1_macro, f1_weight, prec_mac, rec_mac, cm, dur)

def main():
    set_seed(SEED)
    # input probe
    C = load_subject_csv(SUBJECTS[0]).shape[1] - 1
    input_shape = (1, 1, WINDOW_SIZE, C)

    per_fold = []   # will store per-subject metrics (best epoch)
    for test_sid in SUBJECTS:
        train_ids = [s for s in SUBJECTS if s != test_sid]
        test_ids  = [test_sid]

        train_ds = WearDataset(train_ids)
        test_ds  = WearDataset(test_ids)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TinyHAR(
            input_shape=input_shape, number_class=NUM_CLASSES, filter_num=16,
            nb_conv_layers=4, cross_channel_interaction_type="attn",
            cross_channel_aggregation_type="FC",
            temporal_info_interaction_type="lstm",
            temporal_info_aggregation_type="naive",
            dropout=0.1, activation="ReLU", feature_extract=None,
        ).to(DEVICE)

        crit = nn.CrossEntropyLoss()
        opt  = torch.optim.Adam(model.parameters(), lr=LR)

        print(f"\n=== LOSO fold: test sbj_{test_sid} ===")
        best = {"f1_macro": -1}
        history = []

        for ep in range(1, EPOCHS+1):
            tr_loss, tr_acc, tr_time = train_one_epoch(model, train_loader, crit, opt, DEVICE)
            va_loss, va_acc, va_f1_m, va_f1_w, va_prec_m, va_rec_m, cm, va_time = evaluate(
                model, test_loader, crit, DEVICE
            )
            history.append({
                "epoch": ep,
                "train": {"loss": tr_loss, "acc": tr_acc, "time_sec": tr_time},
                "test":  {"loss": va_loss, "acc": va_acc,
                          "precision_macro": va_prec_m,
                          "recall_macro": va_rec_m,
                          "f1_macro": va_f1_m,
                          "f1_weighted": va_f1_w,
                          "time_sec": va_time}
            })
            print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.3f} ({tr_time:.1f}s) | "
                  f"test {va_loss:.4f} acc {va_acc:.3f} "
                  f"prec {va_prec_m:.3f} rec {va_rec_m:.3f} f1 {va_f1_m:.3f} ({va_time:.1f}s)")

            if va_f1_m > best["f1_macro"]:
                best = {
                    "epoch": ep, "loss": va_loss, "acc": va_acc,
                    "precision_macro": va_prec_m, "recall_macro": va_rec_m,
                    "f1_macro": va_f1_m, "f1_weighted": va_f1_w,
                    "confusion_matrix": cm
                }

        # save per-fold JSON
        out = {
            "fold": f"loso_sbj_{test_sid}",
            "subject_test": int(test_sid),
            "subjects_train": train_ids,
            "config": {
                "window_size": WINDOW_SIZE, "overlap": OVERLAP,
                "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
                "device": DEVICE, "num_classes": NUM_CLASSES, "seed": SEED
            },
            "class_index": IDX2CLASS,
            "num_train_windows": int(len(train_ds)),
            "num_test_windows": int(len(test_ds)),
            "per_epoch": history,
            "best_epoch": best["epoch"],
            "best_test_metrics": {
                "accuracy": best["acc"],
                "precision_macro": best["precision_macro"],
                "recall_macro": best["recall_macro"],
                "f1_macro": best["f1_macro"],
                "f1_weighted": best["f1_weighted"]
            },
            "best_confusion_matrix": best["confusion_matrix"]
        }
        with open(os.path.join(SAVE_ROOT, f"loso_sbj_{test_sid}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved → {os.path.join(SAVE_ROOT, f'loso_sbj_{test_sid}.json')}")

        per_fold.append({
            "subject": int(test_sid),
            "accuracy": best["acc"],
            "precision_macro": best["precision_macro"],
            "recall_macro": best["recall_macro"],
            "f1_macro": best["f1_macro"],
            "f1_weighted": best["f1_weighted"]
        })

    # print & save summary (mean across subjects)
    mean_acc = float(np.mean([r["accuracy"] for r in per_fold]))
    mean_prec = float(np.mean([r["precision_macro"] for r in per_fold]))
    mean_rec  = float(np.mean([r["recall_macro"] for r in per_fold]))
    mean_f1   = float(np.mean([r["f1_macro"] for r in per_fold]))
    mean_f1w  = float(np.mean([r["f1_weighted"] for r in per_fold]))

    summary = {
        "per_subject": per_fold,
        "means": {
            "accuracy": mean_acc,
            "precision_macro": mean_prec,
            "recall_macro": mean_rec,
            "f1_macro": mean_f1,
            "f1_weighted": mean_f1w
        }
    }
    with open(os.path.join(SAVE_ROOT, "loso_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== LOSO Summary ===")
    for r in per_fold:
        print(f"sbj_{r['subject']}: acc {r['accuracy']:.3f} | "
              f"P {r['precision_macro']:.3f} R {r['recall_macro']:.3f} F1 {r['f1_macro']:.3f}")
    print(f"MEAN → acc {mean_acc:.3f} | P {mean_prec:.3f} R {mean_rec:.3f} F1 {mean_f1:.3f}")

if __name__ == "__main__":
    main()
