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
git clone https://github.com/PES1UG24AM921/ml-music-generation.git
cd ml-music-generation
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Windows PowerShell: if activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

Data (MAESTRO v3.0.0, MIDI only, ~58 MB):
```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\get_data.ps1
# Mac/Linux
bash scripts/get_data.sh
```
Quick pipeline check without real data: `python scripts/make_fake_data.py`, then use `--data data/fake`. Delete `data/fake` before real runs, because the loader reads every `.mid` file under `data/`.

## Run
All reported results use `--max_files 100 --epochs 15`.
```bash
python src/naive_bayes.py --data data --max_files 100
python src/train.py --model mlp     --data data --max_files 100 --epochs 15
python src/train.py --model lstm    --data data --max_files 100 --epochs 15
python src/train.py --model seq2seq --data data --max_files 100 --epochs 15
python src/generate.py --model lstm --notes 200 --temperature 0.9
```
Outputs (metrics JSON, loss plots, generated `.mid` files) are written to `outputs/`. Model checkpoints (`*.pt`) are git-ignored, so train locally before running `generate.py`.
Play a `.mid` file with VLC + a soundfont, or upload it to https://onlinesequencer.net.

## Data and representation
- Dataset: MAESTRO v3.0.0 (classical piano performances). Files are sorted by path and the first 100 are used (`--max_files 100`), giving 100 songs and 360,047 onset events.
- Onset events: notes starting within 30 ms of each other form one event (`src/data_prep.py`). Only the 88 piano keys (MIDI 21-108) are kept.
- Naive Bayes: left hand (pitch < 60) = chord, right hand (pitch >= 60) = note. A chord is the pitch-class set of the left-hand notes at that event (the reference used 3 consecutive left-hand notes).
- Neural models: top note of every event as a token (88 keys), windows of 32 previous notes. MLP/LSTM predict the next note; seq2seq predicts the next 8 notes.
- 80/10/10 train/val/test split by song, so test songs are never seen in training. MLP/LSTM windows: 290,706 train / 24,441 val / 41,700 test.
- Generated MIDI uses a fixed time step per event, so durations and velocities are not modelled.

## Results
Metrics come from `outputs/results_*.json`.

| Model | Train loss | Test loss | Test accuracy | Notes |
|---|---|---|---|---|
| Naive Bayes | - | - | 12.3% exact pitch (19.4% pitch class) | Baselines on the same events: most-common-note 5.4%, flat random 1.7%. Train accuracy 12.8% |
| MLP | 1.498 (epoch 15) | 3.353 | 15.1% next-note | Best val loss 3.435 at epoch 1, then val loss rises: overfitting |
| LSTM | TBD | TBD | TBD | |
| Encoder-decoder | TBD | TBD | TBD | Accuracy is per token over 8 predicted notes |

Naive Bayes counts a hit if its guess matches any right-hand note of the event, and the neural models predict the top note of the next event, so their accuracies are not directly comparable. Seq2seq accuracy is per token over 8 notes.

## Contributions
- Rohan: environment and data setup, Naive Bayes and MLP runs and results, README, write-up draft
- Harsha: LSTM and encoder-decoder (seq2seq) training, results and loss plots, generated MIDI, presentation slides