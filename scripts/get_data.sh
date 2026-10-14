#!/usr/bin/env bash
# Option A (bulk, easy): MAESTRO v3 MIDI-only (~60 MB zip, ~1200 classical piano performances)
set -e
mkdir -p data
curl -L -o data/maestro_midi.zip \
  https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip -q data/maestro_midi.zip -d data/
echo "MAESTRO ready in data/. Tip: run training with --max_files 300 first."
# Option B (what the Stanford report used): classical piano MIDIs from http://www.piano-midi.de/midi_files.htm
# download composer zips by hand (Bach, Beethoven, Brahms, Mozart, Chopin) into data/
