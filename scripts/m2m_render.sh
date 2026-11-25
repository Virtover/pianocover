#!/bin/sh
# Music2MIDI covers (scripts/m2m_bench.py run) -> MP3 with the YDP grand piano (CC BY 3.0).
# Runs in pianocover:voice (fluidsynth + ffmpeg). Timing is kept 1:1 with the source song
# (no silence trimming), so the MP3 can be laid over the original.
#   sh scripts/m2m_render.sh /data/m2m/20261124
set -eu
OUT=${1:-/data/m2m/20261124}
SF2=${SF2:-/data/ext/sf2/YDP-GrandPiano-20160804.sf2}
mkdir -p "$OUT/mp3"
TMP=$(mktemp -d)
for mid in "$OUT"/midi/*.mid; do
    mp3="$OUT/mp3/$(basename "$mid" .mid).mp3"
    [ -s "$mp3" ] && continue
    nice fluidsynth -ni -q -g 0.6 -r 44100 -F "$TMP/a.wav" "$SF2" "$mid"
    nice ffmpeg -nostdin -loglevel error -y -i "$TMP/a.wav" -af loudnorm=I=-18 -ar 44100 -b:a 192k "$mp3"
    echo "mp3 $(basename "$mp3")"
done
rm -rf "$TMP"
