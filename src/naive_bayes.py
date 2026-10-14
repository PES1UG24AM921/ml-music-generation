"""Naive-Bayes-like model (report section 4.1): learn P(right-hand note | left-hand chord).

Deviation from the Stanford report: a 'chord' here is the pitch-class set of the left-hand
notes at that onset (the report used 3 consecutive left-hand notes). Mention this in your write-up.
"""
import argparse
import collections
import json
import os

import numpy as np

from data_prep import SPLIT, VOCAB, LOW, load_songs, split_songs, write_midi


def chord_key(ev):
    return tuple(sorted({p % 12 for p in ev if p < SPLIT}))


def right_notes(ev):
    return [p for p in ev if p >= SPLIT]


def fit(songs):
    counts, overall = collections.defaultdict(collections.Counter), collections.Counter()
    for s in songs:
        for ev in s:
            for p in right_notes(ev):
                counts[chord_key(ev)][p] += 1
                overall[p] += 1
    return counts, overall


def dist(counts, overall, ck):
    c = counts.get(ck) or overall          # unseen chord -> fall back to global distribution
    pitches = np.array(list(c.keys()))
    w = np.array(list(c.values()), dtype=float)
    return pitches, w / w.sum()


def evaluate(counts, overall, songs, seed=0):
    rng = np.random.RandomState(seed)
    top_global = overall.most_common(1)[0][0]
    n = hit = hit_pc = hit_global = hit_rand = 0
    for s in songs:
        for ev in s:
            truth = right_notes(ev)
            if not truth:
                continue
            ps, pr = dist(counts, overall, chord_key(ev))
            pred = int(ps[pr.argmax()])
            n += 1
            hit += pred in truth
            hit_pc += (pred % 12) in {t % 12 for t in truth}
            hit_global += top_global in truth
            hit_rand += (int(rng.randint(LOW, LOW + VOCAB))) in truth   # flat distribution
    return {
        "events_evaluated": n,
        "acc_exact_pitch": hit / max(n, 1),
        "acc_pitch_class": hit_pc / max(n, 1),
        "baseline_most_common_note": hit_global / max(n, 1),
        "baseline_flat_random": hit_rand / max(n, 1),
    }


def generate(counts, overall, song, seed=0):
    """For every left-hand chord of `song`, sample a right-hand note from P(note|chord)."""
    rng = np.random.RandomState(seed)
    out = []
    for ev in song:
        left = [p for p in ev if p < SPLIT]
        ps, pr = dist(counts, overall, chord_key(ev))
        out.append(left + [int(rng.choice(ps, p=pr))])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    train, val, test = split_songs(load_songs(a.data, a.max_files))
    counts, overall = fit(train)
    print(f"{len(counts)} distinct chords learned from {len(train)} training songs")
    res = {"train": evaluate(counts, overall, train), "test": evaluate(counts, overall, test or val)}
    print(json.dumps(res, indent=2))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "results_naive_bayes.json"), "w"), indent=2)
    src = (test or val or train)[0]
    write_midi(generate(counts, overall, src), os.path.join(a.out, "naive_bayes_generated.mid"))
    write_midi(src, os.path.join(a.out, "original_for_comparison.mid"))
    print("saved outputs/naive_bayes_generated.mid (+ the original song to compare against)")
