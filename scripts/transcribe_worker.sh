#!/usr/bin/env bash
# Extra cover-transcription worker for a download that is still running: repeats passes
# (each skips covers that are done or claimed by another worker) until the download log
# says "done:" and nothing is left.  DATA=/data/raw VRAM=0.15 bash scripts/transcribe_worker.sh
DATA=${DATA:-/data/raw}
VRAM=${VRAM:-0.15}
while true; do
  python -m pianocover.transcribe_covers --data "$DATA" --vram-fraction "$VRAM" 2>&1 | grep -E "realtime|skip|Error" | tail -2
  if grep -q "^done:" "$DATA/download.log" 2>/dev/null; then
    left=$(comm -23 <(ls "$DATA/cover" | grep -v part | sed 's/\.[^.]*$//' | sort) <(ls "$DATA/cover_events" | grep '\.npy$' | sed 's/\.npy$//' | sort) | wc -l)
    [ "$left" -eq 0 ] && break
  fi
  sleep 60
done
echo "worker finished"
