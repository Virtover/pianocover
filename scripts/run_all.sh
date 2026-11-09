#!/usr/bin/env bash
# RESEARCH ONLY: trains on YouTube pairs (license=unknown). For a product, use import_aligned + merge_datasets --licensed-only (see README).
# End-to-end run on one GPU machine. Every step is resumable: rerun the script after an interruption.
#   DATA=/data/raw DS=/data/ds RUN=/data/runs/cover1 bash scripts/run_all.sh
set -euo pipefail

PAIRS=${PAIRS:-"pairs/pop2piano_train_dataset.csv pairs/amtapc_src.json"}
DATA=${DATA:-/data/raw}
DS=${DS:-/data/ds}
RUN=${RUN:-/data/runs/cover1}
WORKERS=${WORKERS:-$(nproc)}
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}

echo "== 1/5 download pairs"
# shellcheck disable=SC2086  # PAIRS is a space-separated list
python -m pianocover.download $PAIRS --out "$DATA" --workers 8

echo "== 2/5 transcribe covers with pretrained Transkun ($GPUS GPUs)"
for ((i = 0; i < GPUS; i++)); do
  CUDA_VISIBLE_DEVICES=$i python -m pianocover.transcribe_covers --data "$DATA" --shard $i --num-shards "$GPUS" &
done
wait

echo "== 3/5 align + build dataset"
if [[ ! -f "$DS/train.pkl" ]]; then
  python -m pianocover.build_dataset --data "$DATA" --out "$DS" --workers "$WORKERS"
fi

echo "== 4/5 fine-tune"
python -m pianocover.train --train "$DS/train.pkl" --val "$DS/val.pkl" \
  --style-stats "$DS/style_stats.json" --out "$RUN" "$@"

echo "== 5/5 evaluate vs. plain Transkun"
python -m pianocover.evaluate --val "$DS/val.pkl" --ckpt "$RUN/best.pt" --out "$RUN/eval.csv"
python -m pianocover.evaluate --val "$DS/val.pkl" --baseline --out "$RUN/eval_transkun_baseline.csv"
