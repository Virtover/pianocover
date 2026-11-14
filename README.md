# pianocover

Turns a song into a piano cover (MIDI) in **one model pass**, using **Transkun V2** as the base.

AMT-APC showed that a strong piano *transcriber* fine-tuned on pairs of (original song,
piano cover) makes a good cover generator. pianocover applies that recipe to Transkun V2,
which had the best published MAESTRO results when its paper came out (ISMIR 2024).
It adds a **style vector** with named controls, so one model can produce anything from an
easy to a virtuoso cover.

The model is Transkun V2 plus about 30k conditioning parameters. Inference costs about the
same as plain Transkun: an estimated 1–2 GB of VRAM, a single pass, no extra models.

## How it works

1. **Data**, either license-clear (recommended for a paid product) or from the web (research only):
   - **teacher:** CC BY / CC0 songs arranged offline by a rule-based reduction pipeline;
   - **silver:** band versions generated with ACE-Step 1.5 from piano arrangements you own;
   - **gold:** commissioned human covers, recorded as MIDI while following the song;
   - **web (research only):** YouTube pop/cover pairs, aligned with DTW (`pairs/` holds
     Pop2Piano's and AMT-APC's lists).
2. **Model:** pretrained Transkun V2. After every encoder block, FiLM layers (a learned scale
   and shift) apply the style vector. They start at zero, so at step 0 the model is exactly
   pretrained Transkun.
3. **Training:** semi-CRF log-likelihood of the cover given the song audio and that cover's
   style. It also uses random key shifts (audio and notes together), optional audio
   augmentation, style dropout, and per-source sampling weights.
4. **Inference:** pick a style preset, or override single features. Post-processing then
   removes ghost notes, keeps chords playable with two hands, and can snap notes to a beat
   grid for sheet music.

Style features (each in standard deviations from the training average): `log_onset_rate`,
`polyphony`, `velocity_mean`, `velocity_std`, `pitch_mean`, `pitch_std`, `left_hand_ratio`,
`pedal_ratio`. Presets: `normal`, `easy`, `simple`, `full`, `virtuoso`.

## Setup (RTX 5070 / any RTX 50-series)

The image uses PyTorch 2.9.1 with CUDA 12.8, which Blackwell cards (sm_120) need. The host
needs an NVIDIA driver of version 570 or newer.

```bash
docker build -t pianocover .
docker run --gpus all -it --shm-size=8g -v /data:/data pianocover
python scripts/check_env.py      # run this first: GPU support, one training step, one generation
```

`--shm-size=8g` (or `--ipc=host`) is required for training with `--workers > 0`: Docker's
default 64 MB `/dev/shm` makes DataLoader workers crash.

`check_env.py` loads pretrained Transkun into the model and runs one bf16 training step
through the semi-CRF and the style layers. It checks the style layers actually get
gradients, and that gradient checkpointing recomputes the same forward pass as the original
(FiLM included), comparing gradients with checkpointing on and off. It then generates 30 s
of audio and prints time and peak VRAM. If it says `ALL CHECKS PASSED`, the stack works on
your card.

**Measured (RTX 3060 Ti 8 GB, sm_86, 2026-11-09):**
- train step, batch 2 × 16 s: 2.9 GiB peak;
- generation, 30 s of audio: 0.83 GiB peak;
- checkpointed vs plain FiLM gradients: relative difference 1e-7.

sm_120 (RTX 5070) has not been run yet.

## Stopping and resuming training

Stop training whenever you like and run **the same command** again to continue. It resumes
from `<out>/last.pt` at the exact step and data position (the same chunks, in the same
order), with the same optimizer, scheduler and RNG state.

- **Ctrl+C, `docker stop -t 60 <container>`, or an OS shutdown (SIGTERM):** training finishes
  the current step, saves, and exits with code 130. A stop during validation skips the rest
  of that validation. Press Ctrl+C twice to abort without saving.
- **Crash or power loss:** `last.pt` is also written every `--save-every-min` minutes
  (default 10), so at most that much training is lost. Writes are atomic (temp file, fsync,
  rename). The previous copy is kept as `last.prev.pt` and is used automatically if
  `last.pt` is unreadable.
- **`log.csv`:** rows written after the resumed checkpoint are dropped, so they're not
  duplicated.

`scripts/test_resume.sh <dataset>` checks all of this end to end. It interrupts a run with
SIGINT, resumes it, and compares the consumed chunks with an uninterrupted run. **Verified:**
`RESUME EXACT: PASS`. Also verified: SIGINT and SIGTERM sent to the whole process group,
and a truncated `last.pt` falling back to `last.prev.pt`.

## Teacher pipeline (license-clear "teacher" data)

`pianocover/teacher/` arranges CC BY / CC0 songs into piano covers offline by **reduction**:
every output note was heard in the song, moved by octaves at most. Chords are used only to
filter and correct notes.
- **RH melody:** the vocal line. Where there are no vocals, guitar riffs and solos (or
  synth hooks) take its place.
- **LH:** the real bass line.
- **Middle voices:** placed on real strum, keys and drum onsets, and voiced from the
  transcribed guitar/keys notes.
- **Density:** follows each section's energy.
- **Difficulty:** five levels (`easy` … `virtuoso`).

Models: Demucs htdemucs (4 stems), basic-pitch (ONNX on GPU), Transkun, beat_this and BTC.
Licenses and check dates are in `docs/MODEL_LICENSES.md`.

```bash
# 1. pick and fetch commercially usable FMA tracks (license gate = import_aligned's)
python -m pianocover.teacher.curate_fma --meta /data/fma/fma_metadata --out /data/fma_cc --limit 10 --download
# 2. analyse + arrange (models stay loaded; analyses are cached as .npz)
python -m pianocover.teacher.pipeline /data/fma_cc/songs.csv --out /data/teacher --levels simple,normal,full
#    re-tune arrange.py, then re-arrange from the cache without the GPU:
python -m pianocover.teacher.pipeline /data/fma_cc/songs.csv --out /data/teacher --rearrange
# 3. import (one WAV per song, however many levels)
python -m pianocover.import_aligned /data/teacher/manifest.csv --out /data/ds_teacher
# benchmark: resident VRAM, per-step latency and peak VRAM
python -m pianocover.teacher.bench /data/fma_cc/audio/*.mp3 --json bench.json
```

- **FMA:** `fma_full.zip` (879 GiB) is never downloaded whole. Curation reads only the
  selected MP3s with HTTP range requests.
- **Jamendo:** not used. Its API terms allow free use for non-commercial purposes only;
  commercial use needs a quote from licensing@jamendo.com.
- **MTG-Jamendo:** not used. It has no per-track license field, and secondary sources
  report it as non-commercial.

## License-clear training

**1. Prepare manifests** (CSV, paths relative to the file), one per source or combined:

```csv
id,audio,midi,source,license,song_id,offset
fma_000123,audio/000123.mp3,midi/000123.mid,teacher,CC-BY-4.0,fma_000123,0
acestep_007_jazz,audio/007_jazz.wav,midi/007.mid,silver,ace-step,arr_007,0
gold_0042,audio/0042.mp3,midi/0042_cover.mid,gold,work-for-hire,song_0042,0
```

The MIDI must already be on the song's timeline. `offset` shifts it by a fixed number of
seconds if needed. `song_id` keeps all versions of one song in the same train/val split.

**2. Import** each manifest. Every row passes a license gate first:

```bash
python -m pianocover.import_aligned manifests/teacher.csv --out /data/ds_teacher
python -m pianocover.import_aligned manifests/silver.csv  --out /data/ds_silver
python -m pianocover.import_aligned manifests/gold.csv    --out /data/ds_gold
```

- **Accepted:** `CC0`, `CC-BY-*`, `public-domain`, `owned`, `work-for-hire`, `commissioned`,
  `ace-step`.
- **Always refused:** anything NC (non-commercial) or ND (no derivatives).
- **Refused unless `--allow-sharealike`:** share-alike (SA) licenses.
- **Also refused:** unknown or unrecognised license strings.

Refusals are listed in `import_report.csv`.

**3. Merge** into one dataset. `--licensed-only` drops anything without an accepted license:

```bash
python -m pianocover.merge_datasets /data/ds_teacher /data/ds_silver /data/ds_gold \
    --out /data/ds_all --licensed-only
```

**4. Train in two stages on the same merged dataset**, so both stages share one set of style
statistics:

```bash
# stage 1: learn faithful, playable reductions (starts from pretrained Transkun)
python -m pianocover.train --train /data/ds_all/train.pkl --val /data/ds_all/val.pkl \
    --style-stats /data/ds_all/style_stats.json --out /data/runs/stage1 \
    --source-weights teacher=1,silver=1,gold=1

# stage 2: polish on human covers, select the checkpoint on gold validation songs only
python -m pianocover.train --train /data/ds_all/train.pkl --val /data/ds_all/val.pkl \
    --style-stats /data/ds_all/style_stats.json --out /data/runs/stage2 \
    --init /data/runs/stage1/best.pt --source-weights gold=1,teacher=0.1,silver=0 \
    --val-sources gold --lr-backbone 2e-5 --lr-heads 1e-4 --lr-film 3e-4 --steps 15000
```

A small share of teacher data in stage 2 limits forgetting. `0` excludes a source entirely.

## Training on a 12 GB card

The defaults are `--batch-size 2 --accum 2`, an effective batch of 4 that fits an RTX 5070.
On a 24 GB card, use `--batch-size 4 --accum 1` for speed. If you still run out of memory,
use `--chunk-sec 12`.

**Rough budget.** These figures are estimates, not measurements:
- A full 60k-step stage 1 takes a few days on a 5070. Do a smoke test first with
  `--steps 500 --val-every 250`.
- Disk: about 21 MB of 44.1 kHz mono WAV per 4-minute song (4.7 MB with the default compressed downloads).

## Web pairs and the AMT-APC backend

The pair lists in `pairs/` passed the rights and content filters described in
docs/MODEL_LICENSES.md (2026-11-12 to 11-14):

| List | Pairs | Songs |
|---|---|---|
| `pairs/p2p_filtered.csv` | 890 | 590 |
| `pairs/extra_covers.csv` | 1,938 | the same 590 |
| `pairs/worship.csv` | 251 | 97 |

The current plan fine-tunes AMT-APC (`apc.pth`, MIT) on them rather than Transkun:

```bash
python -m pianocover.download pairs/p2p_filtered.csv pairs/extra_covers.csv pairs/worship.csv --out /data/raw
python -m pianocover.transcribe_covers --data /data/raw          # Transkun, ~19x real time on a 3060 Ti
python -m pianocover.build_dataset --data /data/raw --out /data/ds
python -m pianocover.apc.prepare --ds /data/ds                   # log-mel spectrograms + style vectors
python -m pianocover.apc.train --ds /data/ds --init /data/ext/amt-apc/models/params/apc.pth --out /data/runs/apc1
```

- **Downloads** keep YouTube's compressed audio (about 4.7 MB per song; `--format wav` gives
  WAV instead). Two parallel downloads with a pause each, to avoid rate limiting.
- **VRAM caps:** `transcribe_covers` and `apc.prepare` default to `--vram-fraction 0.5` so the
  machine stays usable; `apc.train` defaults to 0.92. A cap makes CUDA fail with an OOM error
  instead of spilling into system RAM, which on Windows was about 50x slower and crashed WSL.
- **Training speed, measured on the 3060 Ti** (8 GB, 400-step run on 7 pairs):

  | Setting | Throughput | Peak VRAM |
  |---|---|---|
  | default: batch 1, accum 4, bf16, compiled | 4.7 windows/s (38 s of audio per s) | 6.1 GiB |
  | `--freeze-encoder` | 7.6 windows/s | 3.3 GiB |
  | not compiled | 0.38 windows/s | 7.7 GiB |
  | fp32, not compiled (close to AMT-APC's own trainer) | 0.05 windows/s (spilled to RAM) | 10.5 GiB |

  One window (8.2 s) already fills the GPU, so a larger batch is not faster; batch 2 needs
  12 GiB. Estimate for the full set (about 180 h of aligned audio, about 80k windows per epoch):
  about 5 h per epoch on the 3060 Ti; the 5070 should be faster (not measured).
- `--bench 30` times a configuration and exits. `pianocover.apc.model` loads `apc.pth`
  strictly and reproduces AMT-APC's outputs (max difference 2e-5 in fp32); saved checkpoints
  keep the `apc.pth` layout, so the transcription server can load them as they are.

`build_dataset` tags web pairs `license=unknown`, so `merge_datasets --licensed-only` drops
them; train on `/data/ds` directly instead.

## Use it from the transcription server

```python
from pianocover.infer import generate_cover
from pianocover.events import to_pretty_midi

ev = generate_cover("song.mp3", "/data/runs/stage2/best.pt", device="cuda", style="simple", quantize=True)
to_pretty_midi(ev).write("cover.mid")
```

The model is loaded once and stays cached for later calls.

## Monitoring

`log.csv` has the training loss per second of audio, plus these validation metrics:
- `val_nll_per_sec`: what `best.pt` is selected on;
- `val_onset_f1`: the match against the reference cover. It is low by nature because many
  valid covers exist; AMT-APC reported about 0.27;
- `val_density_ratio`: should stay near 1.0 when the cover's own style is used.

Each epoch also prints the share of chunks sampled from each source.
`evaluate.py --baseline` scores plain Transkun on the same songs.

## Troubleshooting

- **The loss becomes NaN.** Add `--no-amp`. Normally only the backbone runs in bf16 and the
  semi-CRF stays in fp32, but turning bf16 off rules it out.
- **`no kernel image` / sm_120 errors.** You're not on the CUDA 12.8 image, or the host
  driver is older than 570. `check_env.py` reports which.
- **Covers come out too dense or too sparse.** Check `val_density_ratio`, then steer with
  `--style` / `--set log_onset_rate=...`.
- **Unit tests.** Run `python tests/test_core.py` and `python tests/test_teacher.py`. These
  cover the code that doesn't need PyTorch, including the arranger's reduction guarantees.
  `scripts/test_resume.sh` covers interrupt/resume, and needs a GPU and a dataset.

## Licensing

- **Code:** this repo, Transkun and AMT-APC are all MIT.
- **Transkun's pretrained weights:** they were trained on MAESTRO (CC BY-NC-SA). You rely
  on the position that model weights aren't copies or adaptations of their training data.
  An email to Transkun's author confirming the bundled checkpoint is MIT would remove
  any doubt.
- **License-clear data:** only teacher, silver and gold pairs with accepted licenses.
  Keep your work-for-hire contracts and the CC license records of every song.
- **CC BY songs** require attribution. Keep the import manifests; they are your
  attribution list.
- I'm not a lawyer; get proper advice before selling a model trained this way.
