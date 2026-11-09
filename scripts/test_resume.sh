#!/usr/bin/env bash
# End-to-end check that training survives an interruption and resumes EXACTLY.
#   bash scripts/test_resume.sh /data/ds_teacher_proto [/data/runs/resume_test]
# A: N steps uninterrupted. B: SIGINT (Ctrl+C) mid-run, then the same command again.
# Passes if B stops cleanly (exit 130), resumes, finishes, and the chunks B consumed
# (first run up to its saved step + resumed run) are exactly the chunks A consumed.
set -euo pipefail
DS=${1:?dataset dir with train.pkl/val.pkl/style_stats.json}
OUT=${2:-/data/runs/resume_test}
STEPS=${STEPS:-24}
COMMON=(--train "$DS/train.pkl" --val "$DS/val.pkl" --style-stats "$DS/style_stats.json"
        --steps "$STEPS" --warmup 2 --val-every 1000 --val-chunks 2 --val-decode 0
        --workers "${WORKERS:-2}" --chunk-sec 8 --save-every-min 100)
rm -rf "$OUT" && mkdir -p "$OUT"

echo "== A: uninterrupted, $STEPS steps"
PIANOCOVER_TRACE_BATCHES="$OUT/a.trace" python -W ignore -m pianocover.train "${COMMON[@]}" --out "$OUT/a" \
    > "$OUT/a.log" 2>&1

echo "== B: interrupted with SIGINT, then resumed"
PIANOCOVER_TRACE_BATCHES="$OUT/b1.trace" python -W ignore -m pianocover.train "${COMMON[@]}" --out "$OUT/b" \
    > "$OUT/b1.log" 2>&1 &
PID=$!
until [ -f "$OUT/b1.trace" ] && [ "$(tail -1 "$OUT/b1.trace" | cut -d' ' -f1)" -ge $((STEPS / 2)) ]; do sleep 0.3; done
kill -INT "$PID"
set +e; wait "$PID"; RC=$?; set -e
echo "first run exit code: $RC (expect 130)"
grep -E "\[stop\]|\[ckpt\]|stopped at" "$OUT/b1.log" || true
STOPPED=$(grep -oE "stopped at step [0-9]+" "$OUT/b1.log" | grep -oE "[0-9]+$")

PIANOCOVER_TRACE_BATCHES="$OUT/b2.trace" python -W ignore -m pianocover.train "${COMMON[@]}" --out "$OUT/b" \
    > "$OUT/b2.log" 2>&1
grep -E "resuming|epoch .*starting|training finished" "$OUT/b2.log"

python - "$OUT" "$STOPPED" "$RC" <<'EOF'
import sys
out, stopped, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
read = lambda n: open(f"{out}/{n}").read().splitlines()
a, b1, b2 = read("a.trace"), read("b1.trace"), read("b2.trace")
# steps after the saved one (a step cut short by the stop) are redone by the resumed run
merged = [l for l in b1 if int(l.split()[0]) <= stopped] + b2
ok = merged == a and rc == 130
print(f"A consumed {len(a)} chunks; B consumed {len([l for l in b1 if int(l.split()[0]) <= stopped])} "
      f"before the stop at step {stopped} + {len(b2)} after resuming")
print("RESUME EXACT: PASS" if ok else "RESUME EXACT: FAIL")
sys.exit(0 if ok else 1)
EOF
