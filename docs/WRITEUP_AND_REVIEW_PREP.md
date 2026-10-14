# Write-up skeleton (PDF, 1-2 pages; confirm the exact limit with faculty)
1. **Problem statement**: generate music from MIDI with ML (project 22, reference: CS229 2018, Kang/Kim/Ringdahl).
2. **Dataset**: name, number of songs, number of notes/events, split (80/10/10 by song), preprocessing (onset events, 88 keys).
3. **Approach**: Naive-Bayes-like P(note|chord); MLP; LSTM; GRU encoder-decoder. One line each on input, output, loss.
4. **Implementation**: repo structure, PyTorch, pretty_midi, hyper-parameters (seq_len 32, Adam, lr 1e-3, 30 epochs).
5. **Results**: your own table from `outputs/results_*.json`, one loss plot, short listening notes.
6. **Conclusions**: which model did best, overfitting observed, limitations (no durations/velocity), future work (Transformer, velocity).

# Reference numbers (Stanford poster/report, NOT yours: cite them as the reference only)
| Model | Train | Test |
|---|---|---|
| Naive Bayes | 22.1% acc | 3.18% acc |
| LSTM | 0.5345 | 0.6526 (poster) / 0.7986 (milestone report) |
| Encoder-decoder | 0.04473 | 0.06773 |
Their conclusion: LSTM sounded best; NB gave clear harmony but overfit; MLP repeated one note (88% test error); encoder-decoder was repetitive.

# Slide outline (7 slides)
1. Title, team, problem 22
2. Problem + why it is hard (music is sequential, evaluation is subjective)
3. Dataset + representation (onset events, 88 keys, left/right hand)
4. Four models (one diagram or line each)
5. Results table + loss curves
6. Live demo: play original vs generated MIDI
7. Conclusions, limitations, future work, contributions

# Likely Q&A
- Why split by song, not by window? Windows from one song leak into test otherwise.
- Why is NB test accuracy low? Many chords, sparse data, and a single note guess per chord.
- Why does the MLP/LSTM overfit? Small data, high capacity; used dropout and best-val checkpointing.
- What does temperature do? Scales logits before sampling: low = safe/repetitive, high = random.
- Why teacher forcing in the seq2seq? Faster, more stable training; free-running at eval.
- How would you evaluate music quality? Next-note accuracy only measures prediction; a listening survey (the reference used a Turing-style test).
