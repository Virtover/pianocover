#!/usr/bin/env bash
# Build the AMT-APC training set with downloads and cover transcription overlapped.
#   DATA=/data/raw DS=/data/ds bash scripts/build_dataset_pipeline.sh
# Resumable: download, transcription and AMT-APC inputs skip finished files; build_dataset
# redoes every pair. GPU use is capped (--vram-fraction 0.5).
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
# Finish transcription. A pass skips covers claimed by other workers (scripts/transcribe_worker.sh),
# so repeat until no live claim is left; claims older than 30 min are stale and a pass takes them over.
while true; do
  python -m pianocover.transcribe_covers --data "$DATA" 2>&1 | grep -v -i warn | tail -2
  claimed=$(find "$DATA/cover_events" -name '*.lock' -mmin -30 | wc -l)
  [ "$claimed" -eq 0 ] && break
  echo "waiting for $claimed cover(s) other workers are transcribing"
  sleep 60
done

echo "== $(date) align + build"
python -m pianocover.build_dataset --data "$DATA" --out "$DS" --workers 4 2>&1 | tail -4
echo "== $(date) AMT-APC inputs"
python -m pianocover.apc.prepare --ds "$DS" 2>&1 | grep -v -i warn | tail -3
echo "== $(date) done"
