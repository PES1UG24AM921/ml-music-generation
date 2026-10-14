"""Creates 20 synthetic piano MIDIs in data/fake so you can smoke-test the pipeline
before downloading a real dataset. NOT for your final results."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import write_midi

CHORDS = [[48, 52, 55], [53, 57, 48 + 12], [43, 47, 50], [48, 52, 55]]
SCALE = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79]

random.seed(0)
os.makedirs("data/fake", exist_ok=True)
for k in range(20):
    ev, pos = [], random.randrange(len(SCALE))
    for i in range(128):
        pos = min(max(pos + random.choice([-2, -1, 0, 1, 2]), 0), len(SCALE) - 1)
        left = CHORDS[(i // 8) % 4] if i % 4 == 0 else []
        ev.append(sorted(left + [SCALE[pos]]))
    write_midi(ev, f"data/fake/fake_{k:02d}.mid", step=0.3)
print("wrote 20 files to data/fake")
