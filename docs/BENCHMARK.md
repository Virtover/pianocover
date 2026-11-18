# Benchmarking a new GPU (e.g. the RTX 5070): runbook for an AI agent

Follow the steps in order. Every number you report must come from a command you ran on the
target GPU; anything else is an estimate and must be labelled as one. Reference results from
the dev machine (RTX 3060 Ti 8 GB, sm_86) are at the end for comparison.

## Rules

- **Say which GPU a number comes from** (name, VRAM, compute capability, driver, date). The dev
  card (3060 Ti) and the target (5070, sm_120) are different machines.
- **Don't change defaults or code to make a benchmark pass.** If something fails, report the
  error and what you tried.
- **One GPU job at a time.** Close other GPU work, or the timings are wrong; say if the
  machine's desktop was in use during the run.
- **Never pass `--vram-fraction` above the default 0.92.** On Windows, going past the cap
  spills into system RAM, which runs about 50x slower and has crashed WSL.
- **Don't commit, push or delete data** unless the user asks. Write results where step 6 says.

## 0. Prerequisites

- NVIDIA driver 570 or newer (RTX 50-series); Docker with GPU support (`docker run --gpus all`).
- The image: `docker build -t pianocover .` in the repo (PyTorch 2.9.1, CUDA 12.8; sm_120 needs
  CUDA 12.8+).
- **Data**, one of:
  - **Benchmark kit (about 140 MB, enough for steps 1-3):** on the dev machine, copy out the
    small test set, the AMT-APC weights and the 10 CC songs, then load them into a volume
    named `pc-data` on the target (same paths as on the dev machine):
    ```bash
    # dev machine (Git Bash)
    MSYS_NO_PATHCONV=1 docker run --rm -v pc-data:/data:ro alpine tar cf - -C /data \
        apc_test/ds ext/amt-apc/models/params/apc.pth fma_cc > bench_kit.tar
    # target machine
    MSYS_NO_PATHCONV=1 docker run --rm -i -v pc-data:/data alpine tar xf - -C /data < bench_kit.tar
    ```
  - **Everything:** restore the whole volume from the USB backup (`F:\pianocover-backup\pc-data.tar`):
    `MSYS_NO_PATHCONV=1 docker run --rm -i -v pc-data:/data alpine tar xf - -C /data < F:/pianocover-backup/pc-data.tar`.
    The full set is only needed for step 4's epoch estimate on the real dataset (`/data/ds_all`)
    and for actual training.

Run every command below inside the container, from the repo root:

```bash
docker run --gpus all -it --rm --shm-size=8g -v pc-data:/data -v "$PWD:/w" -w /w pianocover
```

(`--shm-size=8g` is required: DataLoader workers crash with Docker's 64 MB default.)

## 1. Environment check

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python scripts/check_env.py
```

It must end with `ALL CHECKS PASSED`. Record: torch / CUDA versions, compute capability, the
train-step and generation peak VRAM, and the generation time it prints. A `no kernel image` /
sm_120 error means the image isn't the CUDA 12.8 one or the driver is too old; stop and report.

## 2. AMT-APC training throughput (the main benchmark)

Run the matrix script; don't run configurations by hand:

```bash
bash scripts/bench_matrix.sh          # DRY_RUN=1 prints the commands first
```

- **What it measures:** precision bf16 / fp16 / fp32 × batch 1 / 2 / 4 (accumulation keeps the
  effective batch at 4), plus bf16 with `--freeze-encoder` and bf16 without `torch.compile`.
  Each run times 10 optimizer steps after 5 warm-up steps and writes no checkpoint.
- **Built-in time limits:**
  - each run is killed after 600 s (`TIMEOUT`) and recorded as `TIMEOUT`;
  - an out-of-memory error is recorded as `OOM` and skips the larger batches of that precision;
  - fp16 and fp32 only try batches that fitted in bf16, which uses the least memory;
  - compiled kernels are cached between runs.

  It took 5.5 minutes on the 3060 Ti (6 runs). Don't raise `TIMEOUT` or `STEPS`: a slow
  configuration showing up as `TIMEOUT` is itself the answer.
- **Output:** a table at the end and a TSV in `/data/bench/` (one row per run, with ms/step,
  windows/s and peak GiB), plus each run's log next to it. `OOM` and `TIMEOUT` are results, not
  failures; report them as "does not fit" / "too slow".
- **Optional, only if it fits in the time:** while the default configuration runs, check GPU
  utilisation with `nvidia-smi`. Well under 90% means data loading is the bottleneck; then
  rerun just that configuration (its command is printed by `DRY_RUN=1`) with `--workers 4` and
  `--workers 12` appended.

## 3. Teacher pipeline (optional; only if the target machine will run the teacher)

```bash
python -m pianocover.teacher.bench /data/fma_cc/audio/*.mp3 --json /data/bench_5070.json
```

Record resident VRAM, peak device VRAM and the steady-state per-song latency from the JSON.
The 3060 Ti result is in `/data/bench_final.json` on the dev machine.

## 4. Epoch estimate for the real dataset

Count the training windows (needs `/data/ds_all`, i.e. the full restore), then divide by the
best windows/s from step 2:

```bash
python -c "
from pianocover.apc.dataset import Windows, load_rows
n = len(Windows(load_rows('/data/ds_all/apc/train.pkl'), '/data/ds_all/apc/spec', jitter=False))
print(n, 'windows per epoch')"
```

Hours per epoch = windows / (windows per second) / 3600. Label it as an estimate: the windows/s
come from the 7-pair test set, and real data loading may be slower.

## 5. Optional: a short real run

Only if the user asks: about 400 steps on the real data, to confirm loss goes down and nothing
crashes over time:

```bash
python -m pianocover.apc.train --ds /data/ds_all --init /data/ext/amt-apc/models/params/apc.pth \
    --out /data/runs/bench_5070 --steps 400 --val-every 200
```

## 6. Report

1. Add a dated block under the 3060 Ti one in README.md (section "Setup"), for example
   `**Measured (RTX 5070 12 GB, sm_120, <date>):**`, and remove the line "sm_120 (RTX 5070)
   has not been run yet." once `check_env.py` passed there.
2. Add a row per configuration to the table in README.md ("Training speed, measured on ...");
   name the GPU in the row or in a new table.
3. In your reply to the user, give: what passed, a configuration table, the recommended
   training command for this GPU (the fastest configuration that fit), the epoch estimate
   (labelled as an estimate), and anything that failed with its error message.

## Reference: RTX 5070 12 GB (sm_120), 2026-11-17

`bench_matrix.sh` on `/data/apc_test/ds`; the logs are in `/data/runs/apc_5070/logs`.

| Setting | Result |
|---|---|
| bf16, batch 1 × accum 4, compiled | 501 ms/step, 7.98 windows/s (65 s of audio per s), 6.09 GiB |
| fp16, batch 1 | 501 ms/step, 7.99 windows/s, 6.09 GiB |
| fp32, batch 1 | 1208 ms/step, 3.31 windows/s, 9.54 GiB |
| bf16, batch 2 | out of memory (10.96 GiB allowed under the 0.92 cap) |
| bf16 `--freeze-encoder` | 327 ms/step, 12.22 windows/s, 3.25 GiB |
| bf16 not compiled | 740 ms/step, 5.41 windows/s, 7.71 GiB |
| full training, `/data/ds_all`, 40,000 steps | 5 h 16 min (about 2.5 epochs) |

## Reference: RTX 3060 Ti 8 GB (sm_86), dev machine

| Benchmark | Setting | Result |
|---|---|---|
| `check_env.py` (2026-11-09) | train step, batch 2 × 16 s | 2.9 GiB peak |
| | generation, 30 s of audio | 0.83 GiB peak |
| `bench_matrix.sh` (2026-11-16, 7 pairs, 5.5 min) | bf16, batch 1 × accum 4, compiled | 833 ms/step, 4.80 windows/s (39 s of audio per s), 6.09 GiB |
| | fp16, batch 1 | 816 ms/step, 4.90 windows/s, 6.09 GiB |
| | fp32, batch 1 | OOM under the 0.92 cap (earlier uncapped run: 10.5 GiB, spilled to RAM) |
| | bf16, batch 2 | OOM (needs about 12 GiB); fp16/fp32 batch 2 and 4 skipped |
| | bf16 `--freeze-encoder` | 548 ms/step, 7.29 windows/s, 3.26 GiB |
| | bf16 not compiled | OOM under the cap (earlier uncapped: 0.38 windows/s, 7.7 GiB) |
| teacher bench | 10 CC songs | `/data/bench_final.json` (peak device use 4.6 GB) |
| windows per epoch | `/data/ds_all` (2026-11-16, 2431 training pairs) | 64,210, i.e. about 3.8 h per epoch at 4.7 windows/s (estimate) |
