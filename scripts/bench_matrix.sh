#!/usr/bin/env bash
# AMT-APC training benchmark over precisions x batch sizes, with a hard time limit per run.
#
#   bash scripts/bench_matrix.sh                    # inside the container, repo root
#   DRY_RUN=1 bash scripts/bench_matrix.sh          # print the commands only
#
# Effective batch stays 4 (accum = 4 / batch), so configurations train the same way.
# Each run: STEPS timed optimizer steps after 5 warm-up steps (pianocover.apc.train --bench),
# killed after TIMEOUT seconds (recorded as TIMEOUT). An out-of-memory error is recorded as OOM
# and skips the larger batches of that precision; fp16 and fp32 only try batches that fitted in
# bf16 (it needs the least memory). EXTRAS=0 skips the --freeze-encoder and
# --no-compile runs. 5.5 min on the 3060 Ti (2026-11-16); hard limit TIMEOUT (600 s) per run.
# Results: one line per run in $OUT (TSV) plus each run's log next to it.
set -uo pipefail
# reuse torch.compile results between runs (same shapes compile once)
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1

DS=${DS:-/data/apc_test/ds}
INIT=${INIT:-/data/ext/amt-apc/models/params/apc.pth}
STEPS=${STEPS:-10}
TIMEOUT=${TIMEOUT:-600}
PRECISIONS=${PRECISIONS:-"bf16 fp16 fp32"}
BATCHES=${BATCHES:-"1 2 4"}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
OUT=${OUT:-/data/bench/apc_matrix_${GPU:-gpu}_$(date +%Y%m%d_%H%M).tsv}

[ -n "${DRY_RUN:-}" ] || { mkdir -p "$(dirname "$OUT")"; printf "precision\tbatch\taccum\textra\tresult\tms_per_step\twindows_per_s\tpeak_gib\n" > "$OUT"; }

run() {  # precision batch extra-flags...
  local prec=$1 b=$2; shift 2
  local accum=$(( 4 / b )); [ "$accum" -lt 1 ] && accum=1
  local tag="${prec}_b${b}${*:+_$(echo "$*" | tr -d ' -')}"
  local log="${OUT%.tsv}_${tag}.log"
  local cmd=(python -m pianocover.apc.train --ds "$DS" --init "$INIT" --out "/tmp/bench_$tag"
             --bench "$STEPS" --precision "$prec" --batch "$b" --accum "$accum" "$@")
  if [ -n "${DRY_RUN:-}" ]; then echo "timeout $TIMEOUT ${cmd[*]}"; return 0; fi
  echo "== $(date +%T) $tag" >&2
  timeout "$TIMEOUT" "${cmd[@]}" > "$log" 2>&1
  local rc=$? line result ms wps peak
  line=$(grep -m1 "^BENCH" "$log")
  if [ -n "$line" ]; then
    result=ok
    ms=$(sed -E 's/.*: ([0-9]+) ms\/step.*/\1/' <<<"$line")
    wps=$(sed -E 's/.* ([0-9.]+) windows\/s.*/\1/' <<<"$line")
    peak=$(sed -E 's/.*peak ([0-9.]+) GiB.*/\1/' <<<"$line")
  elif grep -qi "out of memory" "$log"; then result=OOM
  elif [ $rc -eq 124 ]; then result=TIMEOUT
  else result="error(rc=$rc)"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$prec" "$b" "$accum" "${*:--}" "$result" "${ms:--}" "${wps:--}" "${peak:--}" >> "$OUT"
  echo "   $result ${line#BENCH }" >&2
  [ "$result" = ok ]
}

# bf16 uses the least memory, so fp16 / fp32 only try batches that fitted in bf16
cap=999
for prec in $PRECISIONS; do
  best=0
  for b in $BATCHES; do
    if [ "$b" -gt "$cap" ]; then echo "   $prec batch $b skipped (did not fit in bf16)" >&2; continue; fi
    if run "$prec" "$b"; then best=$b
    elif [ -z "${DRY_RUN:-}" ] && grep -q "OOM" <(tail -n1 "$OUT"); then
      echo "   larger batches skipped for $prec" >&2; break
    fi
  done
  [ "$prec" = bf16 ] && [ -z "${DRY_RUN:-}" ] && cap=$best
done
if [ "${EXTRAS:-1}" = 1 ]; then
  run bf16 1 --freeze-encoder   # decoder + style gate only
  run bf16 1 --no-compile       # how much torch.compile gives on this GPU
fi

[ -n "${DRY_RUN:-}" ] || { echo; column -t -s $'\t' "$OUT" 2>/dev/null || cat "$OUT"; echo; echo "results: $OUT"; }
