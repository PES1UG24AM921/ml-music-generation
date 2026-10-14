"""MIDI -> onset events -> training windows. Shared by every model."""
import glob
import os

import numpy as np
import pretty_midi

LOW, HIGH = 21, 108            # the 88 piano keys
VOCAB = HIGH - LOW + 1         # 88 tokens
SPLIT = 60                     # pitch < 60 = left hand, >= 60 = right hand


def load_events(path, tol=0.03):
    """One song -> list of onset events. Each event = sorted pitches that start together."""
    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted(
        (n.start, n.pitch)
        for inst in pm.instruments if not inst.is_drum
        for n in inst.notes if LOW <= n.pitch <= HIGH
    )
    events, cur, t0 = [], [], None
    for start, pitch in notes:
        if t0 is None or start - t0 > tol:
            if cur:
                events.append(sorted(set(cur)))
            cur, t0 = [pitch], start
        else:
            cur.append(pitch)
    if cur:
        events.append(sorted(set(cur)))
    return events


def load_songs(folder, max_files=None, min_events=64):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.mid*"), recursive=True))
    if max_files:
        files = files[:max_files]
    songs = []
    for f in files:
        try:
            ev = load_events(f)
        except Exception as e:  # corrupt midi etc.
            print("skipping", f, "->", e)
            continue
        if len(ev) >= min_events:
            songs.append(ev)
    print(f"loaded {len(songs)} songs, {sum(len(s) for s in songs)} onset events")
    return songs


def split_songs(songs, seed=42):
    """80/10/10 split by SONG (not by window) so test songs are never seen in training."""
    idx = np.random.RandomState(seed).permutation(len(songs))
    n = len(songs)
    a, b = int(0.8 * n), int(0.9 * n)
    pick = lambda ids: [songs[i] for i in ids]
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])


def melody(song):
    """Top note of every onset event as a token 0..87."""
    return [ev[-1] - LOW for ev in song]


def make_windows(songs, seq_len=32, horizon=1):
    """Sliding windows: X = previous seq_len notes, y = next `horizon` notes."""
    X, y = [], []
    for s in songs:
        m = melody(s)
        for i in range(len(m) - seq_len - horizon + 1):
            X.append(m[i:i + seq_len])
            y.append(m[i + seq_len:i + seq_len + horizon])
    X = np.array(X, dtype=np.int64).reshape(-1, seq_len)
    y = np.array(y, dtype=np.int64).reshape(-1, horizon)
    return X, (y[:, 0] if horizon == 1 else y)


def write_midi(events, path, step=0.4):
    """events = list of lists of MIDI pitches -> .mid file (fixed step per event)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)  # acoustic grand piano
    for i, ev in enumerate(events):
        for p in ev:
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(p),
                                               start=i * step, end=i * step + step * 0.95))
    pm.instruments.append(inst)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pm.write(path)
