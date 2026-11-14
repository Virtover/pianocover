#!/usr/bin/env bash
# Build the AMT-APC training set with downloads and cover transcription overlapped.
#   DATA=/data/raw DS=/data/ds bash scripts/build_dataset_pipeline.sh
# Resumable: every step skips finished files. GPU use is capped (--vram-fraction 0.5).
set -uo pipefail
DATA=${DATA:-/data/raw}
DS=${DS:-/data/ds}
LISTS=${LISTS:-"pairs/p2p_filtered.csv pairs/extra_covers.csv pairs/worship.csv"}
mkdir -p "$DATA"

echo "== $(date) download + transcribe (overlapped)"
# shellcheck disable=SC2086
python -m pianocover.download $LISTS --out "$DATA" > "$DATA/download.log" 2>&1 &
DL=$!
while kill -0 "$DL" 2>/dev/null; do
  python -m pianocover.transcribe_covers --data "$DATA" 2>&1 | grep -v -i warn | tail -2
  sleep 120
done
wait "$DL"; tail -2 "$DATA/download.log"
python -m pianocover.transcribe_covers --data "$DATA" 2>&1 | grep -v -i warn | tail -2

echo "== $(date) align + build"
python -m pianocover.build_dataset --data "$DATA" --out "$DS" --workers 4 2>&1 | tail -4
echo "== $(date) AMT-APC inputs"
python -m pianocover.apc.prepare --ds "$DS" 2>&1 | grep -v -i warn | tail -3
echo "== $(date) done"
