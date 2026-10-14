"""Three neural models from the Stanford report (sections 4.2 - 4.4), in PyTorch."""
import torch
import torch.nn as nn

from data_prep import VOCAB


class MLP(nn.Module):
    """'Vanilla NN': previous seq_len notes -> one hidden layer (1024) -> next note."""
    def __init__(self, seq_len, emb=32, hidden=1024):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb)
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * emb, hidden), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(hidden, VOCAB))

    def forward(self, x, y=None):
        return self.net(self.emb(x))


class LSTMNet(nn.Module):
    """Single LSTM layer (256 hidden, as in the report) + softmax over 88 keys."""
    def __init__(self, emb=64, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(hidden, VOCAB)

    def forward(self, x, y=None):
        h, _ = self.lstm(self.emb(x))
        return self.out(self.drop(h[:, -1]))


class Seq2Seq(nn.Module):
    """GRU encoder-decoder: seq_len notes in -> next `horizon` notes out (teacher forcing 0.5)."""
    def __init__(self, emb=64, hidden=256, horizon=8):
        super().__init__()
        self.horizon = horizon
        self.emb = nn.Embedding(VOCAB + 1, emb)          # index VOCAB = <bos>
        self.enc = nn.GRU(emb, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.dec = nn.GRU(emb, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.out = nn.Linear(hidden, VOCAB)

    def forward(self, x, y=None, tf=0.5):
        _, h = self.enc(self.emb(x))
        inp = torch.full((x.size(0), 1), VOCAB, dtype=torch.long, device=x.device)
        logits = []
        for t in range(self.horizon):
            o, h = self.dec(self.emb(inp), h)
            lg = self.out(o[:, 0])
            logits.append(lg)
            if y is not None and torch.rand(1).item() < tf:
                inp = y[:, t:t + 1]                       # teacher forcing
            else:
                inp = lg.argmax(-1, keepdim=True)
        return torch.stack(logits, 1)                     # (B, horizon, VOCAB)


def build(name, seq_len, horizon):
    if name == "mlp":
        return MLP(seq_len)
    if name == "lstm":
        return LSTMNet()
    if name == "seq2seq":
        return Seq2Seq(horizon=horizon)
    raise ValueError(name)
