# Generating Music with Machine Learning
UE24CS352A Machine Learning mini-project, PES University, Sec-H Team H-10 (problem statement 22)

**Team:** Rohan Agarwal (PES1UG24AM921), Pranav Prabhakar (PES1UG24AM463)

Reference project: *Music Composition with Machine Learning*, Kang, Kim, Ringdahl (Stanford CS229, 2018)
https://cs229.stanford.edu/proj2018/report/18.pdf

## Problem
Learn from MIDI piano music and generate new note sequences. Four approaches are compared, as in the reference:
1. Naive-Bayes-like model: P(right-hand note | left-hand chord)
2. Vanilla neural network (MLP on the previous notes)
3. LSTM RNN (next-note prediction)
4. GRU encoder-decoder (sequence in, sequence out)

## Setup
```bash
git clone <this repo>
cd ml-music-generation
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/get_data.sh                                 # or put .mid files into data/ manually
```
Quick pipeline check without real data: `python scripts/make_fake_data.py` then use `--data data/fake`.

## Run
```bash
python src/naive_bayes.py --data data --max_files 300
python src/train.py --model mlp     --data data --max_files 300 --epochs 30
python src/train.py --model lstm    --data data --max_files 300 --epochs 30
python src/train.py --model seq2seq --data data --max_files 300 --epochs 30
python src/generate.py --model lstm --notes 200 --temperature 0.9
```
Outputs (metrics JSON, loss plots, generated `.mid` files) are written to `outputs/`.
Play a `.mid` file with VLC + a soundfont, or upload it to https://onlinesequencer.net.

## Data and representation
- Onset events: notes starting together form one event (`src/data_prep.py`).
- Naive Bayes: left hand (pitch < 60) = chord, right hand (pitch >= 60) = note.
- Neural models: top note of every event, 88 piano keys, windows of 32 previous notes.
- 80/10/10 train/val/test split by song.

## Results (fill in from `outputs/results_*.json`)
| Model | Train loss | Test loss | Test next-note acc | Notes |
|---|---|---|---|---|
| Naive Bayes | - | - | | |
| MLP | | | | |
| LSTM | | | | |
| Encoder-decoder | | | | |

## Contributions
- Rohan: ...
- Pranav: ...
