"""Train one neural model.   python src/train.py --model lstm --data data --epochs 30"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_prep import VOCAB, load_songs, make_windows, split_songs
from models import build

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["mlp", "lstm", "seq2seq"], required=True)
ap.add_argument("--data", default="data")
ap.add_argument("--max_files", type=int, default=None)
ap.add_argument("--seq_len", type=int, default=32)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--out", default="outputs")
args = ap.parse_args()

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
horizon = 8 if args.model == "seq2seq" else 1

train_s, val_s, test_s = split_songs(load_songs(args.data, args.max_files))
sets = [make_windows(s, args.seq_len, horizon) for s in (train_s, val_s, test_s)]
print("windows (train/val/test):", [len(X) for X, _ in sets])
loaders = [DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=args.bs, shuffle=(i == 0)) for i, (X, y) in enumerate(sets)]

model = build(args.model, args.seq_len, horizon).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
lossf = nn.CrossEntropyLoss()


def run(loader, train):
    model.train(train)
    tot_loss = tot_hit = n_ex = n_tok = 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(dev), y.to(dev)
            out = model(X, y if train else None)
            loss = lossf(out.reshape(-1, VOCAB), y.reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot_loss += loss.item() * len(X)
            n_ex += len(X)
            tot_hit += (out.argmax(-1) == y).sum().item()
            n_tok += y.numel()
    return tot_loss / max(n_ex, 1), tot_hit / max(n_tok, 1)


os.makedirs(args.out, exist_ok=True)
ckpt = os.path.join(args.out, f"{args.model}.pt")
hist, best = {"train": [], "val": []}, float("inf")
for ep in range(1, args.epochs + 1):
    tr_loss, tr_acc = run(loaders[0], True)
    va_loss, va_acc = run(loaders[1], False)
    hist["train"].append(tr_loss)
    hist["val"].append(va_loss)
    print(f"epoch {ep:3d} | train {tr_loss:.4f} ({tr_acc:.3f}) | val {va_loss:.4f} ({va_acc:.3f})")
    if va_loss < best:
        best = va_loss
        torch.save({"state": model.state_dict(), "args": vars(args), "horizon": horizon}, ckpt)

model.load_state_dict(torch.load(ckpt, map_location=dev)["state"])
te_loss, te_acc = run(loaders[2], False)
res = {"model": args.model, "best_val_loss": best, "test_loss": te_loss, "test_next_note_acc": te_acc,
       "final_train_loss": hist["train"][-1]}
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(args.out, f"results_{args.model}.json"), "w"), indent=2)

plt.plot(hist["train"], label="train")
plt.plot(hist["val"], label="val")
plt.xlabel("epoch")
plt.ylabel("cross-entropy")
plt.title(f"{args.model} loss")
plt.legend()
plt.savefig(os.path.join(args.out, f"{args.model}_loss.png"), dpi=150)
print("saved", ckpt)
