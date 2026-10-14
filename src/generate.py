"""Generate a MIDI file from a trained model.
   python src/generate.py --model lstm --notes 200 --temperature 0.9"""
import argparse
import os

import numpy as np
import torch

from data_prep import LOW, VOCAB, load_songs, melody, split_songs, write_midi
from models import build

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["mlp", "lstm", "seq2seq"], required=True)
ap.add_argument("--notes", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.9)
ap.add_argument("--out", default="outputs")
a = ap.parse_args()

ck = torch.load(os.path.join(a.out, f"{a.model}.pt"), map_location="cpu")
cfg, horizon = ck["args"], ck["horizon"]
model = build(a.model, cfg["seq_len"], horizon)
model.load_state_dict(ck["state"])
model.eval()

_, val_s, test_s = split_songs(load_songs(cfg["data"], cfg["max_files"]))
seed_song = (test_s or val_s)[0]
seq = melody(seed_song)[:cfg["seq_len"]]                # seed = first notes of an unseen song
rng = np.random.RandomState(0)


def sample(logits):
    z = logits / a.temperature
    p = np.exp(z - z.max())
    return int(rng.choice(VOCAB, p=p / p.sum()))


with torch.no_grad():
    while len(seq) < cfg["seq_len"] + a.notes:
        x = torch.tensor([seq[-cfg["seq_len"]:]])
        out = model(x)[0].numpy()                       # (VOCAB,) or (horizon, VOCAB)
        if out.ndim == 1:
            seq.append(sample(out))
        else:
            seq.extend(sample(row) for row in out)

path = os.path.join(a.out, f"{a.model}_generated.mid")
write_midi([[t + LOW] for t in seq], path)
print("saved", path)
