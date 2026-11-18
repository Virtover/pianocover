# Plan: Transkun for any-instrument audio, with "full" and "reduced" output

Status: plan, 2026-11-18. Nothing below is implemented yet unless marked **done**.
Numbers marked *(estimate)* are not measured.

## Goal

One transcription model for any song or ensemble audio (piano, organ, choir, strings,
orchestra, band with drums, guitar), wrapped in a function with two modes:

```python
transcribe(audio_path, mode="full")     # every pitched note heard (drums excluded)
transcribe(audio_path, mode="reduced")  # the same, made playable on piano where it is not
```

- **full** = the model's output as is.
- **reduced** = a symbolic post-process of the full output. It changes only what is not
  playable. A passage that is already playable comes out unchanged (see "Reduction").

Why one model plus post-processing, not a model with a mode switch: the reduction targets
would come from the same rule anyway, so a conditioned model would only learn to imitate the
rule, with extra errors. A learned reduction is worth it only with human-arranged targets
(later, e.g. from web cover pairs).

## Base model

Transkun V2 (`2.0.pt`, pip `transkun` 2.0.1), 14.1 M parameters (counted from the checkpoint),
44.1 kHz input, 16 s segments, its own `train.py` with augmentation hooks (`--augment`,
`--noiseFolder`, `--irFolder`). License: MIT code; weights ship in the MIT package
(see MODEL_LICENSES.md, open item: confirm with the author).

Capacity is probably not the bottleneck: instrument-agnostic pitch detection works even in
tiny models (basic-pitch). Phase 2 tests this before any long training run.

## Risks the dataset must cover

| Risk | What covers it |
|---|---|
| Synthetic timbre gap (GM soundfonts sound artificial) | Real recordings with verified MIDI, oversampled; several soundfonts per render; room IRs, EQ, compression, noise, MP3 artefacts |
| Forgetting real piano | Piano replay set (real piano audio) in every batch |
| Soft onsets (bowed strings, choir, organ, winds) | Many renders of these families; legato and expression (CC7/CC11) curves kept in the audio MIDI |
| Harmonics that are not notes (organ 4'/2' stops and mixtures, brass) | Organ renders with extra octave/fifth "ranks" added to the audio MIDI only, never to the target |
| Masking in dense textures, drums over everything | Orchestral and band renders; drum tracks mixed at random gain; never in the target |
| Long sustained notes | Targets keep score durations (no piano-like decay) |
| Unisons of several instruments | Merged into one target note (Transkun's `resolveOverlapping`) |
| Velocity in scores is not loudness | Velocity loss down-weighted on synthetic data (check how `train.py` weights it) |

## Data sources

Every source gets a row in MODEL_LICENSES.md before use. NC sources follow the NC policy
there (flags, log, no redistribution, delete raw copies after training).

### Tier 1: real recordings with correct MIDI (highest priority, oversampled)

| Source | Content | License (status) |
|---|---|---|
| MusicNet + MusicNetEM labels | ~330 classical recordings, ~34 h: piano, chamber, orchestra | EM labels CC BY 4.0 (Zenodo). Audio: authors say public-domain recordings; **verify on the original site** |
| Dagstuhl ChoirSet | Real choir multitracks with score MIDI and F0 | CC BY 4.0 (TISMIR article, mirdata). Verify the note alignment |
| GuitarSet | Real guitar, note annotations from a hexaphonic pickup (~3 h) | **Verify** (Zenodo record / GitHub). Guitar matters for this project |
| URMP | 44 chamber pieces, mixes + MIDI scores | No license found yet (Dryad doi:10.5061/dryad.ng3r749). **Verify**; if research-only, treat as NC |
| Bach10 | 10 Bach chorales, quartet recordings + MIDI | Unclear; a derivative is CC BY-NC-SA 3.0. **Verify**, else treat as NC |
| PDMX organ, real recordings | 19 organ recordings DTW-aligned to scores (/data/ds_pdmx) | PDMX CC BY 4.0, scores PD/CC0; recordings from YouTube under the 2026-11-12 policy |

Small sets are duplicated in the training pickle (Transkun samples segments by duration), so
together they make about 30% of the training segments *(target, tune after phase 2)*.
Held-out pieces from each go to the test set (split by piece, never by excerpt).

### Tier 2: piano replay (no new labels)

- Real piano audio we already have: ~2.5 k piano covers in /data/ds_all with Transkun 2.0
  transcriptions (`cover_events`). Training on the model's own output is self-distillation:
  it keeps the current piano behaviour on exactly the kind of audio users upload.
- Optional: MAESTRO v3 (CC BY-NC-SA 4.0, ~200 h, real labels) under the NC policy.
- About 25% of training segments *(target)*.

### Tier 3: synthetic renders of public-domain scores (scale and variety)

Scores: PDMX (CC BY 4.0), `no_license_conflict` + `deduplicated`, licenses publicdomain /
cc-zero, composer died before 1955 (`scripts/organ_synth.py`: `OLD` and `MODERN` filters).
About 45% of training segments *(target)*.

| Family | Selection (PDMX) | State |
|---|---|---|
| Organ | organ-only scores + Bach organ works for keyboard: 89 scores | **done** at 16 kHz for AMT-APC (/data/synth/organ). Re-render at 44.1 kHz for Transkun |
| Chamber (string quartet, duos, trios) | GM 40-47 only, 2-6 tracks | to do |
| Orchestra, incl. percussion | ≥ 6 programs across families | to do |
| Choir and choir + organ / orchestra | GM 52-54 | to do |
| Band with drums (march, ragtime, big band, rock remixes of PD pieces) | MIDI with a drum track | to do (5 examples rendered in Downloads/pianocover_multiinstrument_perkusja) |
| Piano solo | PDMX piano scores | to do, with several piano soundfonts (YDP, UprightPianoKW), so synthetic piano is not "the GM piano" |
| Guitar (classical) | GM 24-25 | to do, small |

Rendering, per render and seeded, all recorded in meta.csv:
- soundfont: GeneralUser GS, FluidR3_GM (MIT), TimGM6mb (GPL-2; audio only, never shipped),
  MuseScore_General (MIT), plus optionally VSCO-2 CE / VSCL (CC0, SFZ, needs the sfizz
  renderer; **verify the license**);
- program: random within the track's instrument family (violin ↔ viola ↔ string ensemble, etc.);
- room: fluidsynth reverb presets, and in training Transkun's `--irFolder` with synthetic IRs
  (generated decaying noise, no license issue);
- mix: per-track gain ±6 dB, drums −12…+3 dB, random EQ tilt, light compression, noise via
  `--noiseFolder` (generated or CC0 noise), random MP3 round-trip (64-192 kbps);
- organ only: 30% of renders add a 4' rank (+12) and 20% a mixture (+12, +19) to the audio
  MIDI at lower velocity, not to the target.

Target for every render: all non-drum notes of the score, folded into 21-108, unisons merged,
score durations kept, no pedal.

Output format (Transkun's DatasetMaestro): `<root>/<source>/<id>.flac` at 44.1 kHz plus a
pickle list of `{"audio_filename", "duration", "fs", "nSamples", "nChannel", "notes"
(transkun.Data.Note list), "source", "split"}`. Audio must be **16-bit PCM WAV**:
`readAudioSlice` uses `scipy.io.wavfile` with mmap and integer normalisation (checked in
transkun 2.0.1). Mono 44.1 kHz is about 0.32 GB per hour, so 100 h is about 32 GB; the Docker
disk lives on C:, which had 153 GB free on 2026-11-18. Keep FLAC copies off the training path
only if needed for backups.

Size target: about 60-100 h synthetic + all of tier 1 + tier 2 sampled *(estimate)*.

## Phases

1. **Data, tier 3 first (CPU, runs in the background):** generalise `scripts/organ_synth.py`
   into `scripts/score_synth.py` (families, 44.1 kHz, Transkun pickle, augment options).
   Listen to 5 samples per family before the full run.
2. **Capacity test (about 1-2 h on the 3060 Ti, VRAM ≤ 5 GiB, including batch size and
   checkpointing):** fine-tune 2.0.pt on 2-3 h of synthetic data. Pass: training-set onset F1
   ≥ 0.95. Fail → larger config or a different architecture, before collecting more data.
3. **Tier 1 download + licence check:** MusicNet(+EM), Dagstuhl, GuitarSet, URMP, Bach10.
   Licence rows go in MODEL_LICENSES.md; NC ones in the NC log. Convert to the same pickle.
4. **Tier 2:** piano replay pickle from /data/ds_all (audio + `cover_events`).
5. **Training:** init 2.0.pt, lower max_lr than the default 2e-4 (start at 5e-5), mixed
   sampling as above, validation per source every N steps. Memory: ≤ 5 GiB VRAM; measure batch
   size × segment length × checkpointing first (`bench`-style, like `scripts/bench_matrix.sh`).
6. **Evaluation:** note F1 (onset; onset+offset) per source on held-out **real** data:
   MusicNet test, Dagstuhl subset, URMP/Bach10 subset, PDMX organ real, plus a piano regression
   check against 2.0.pt on held-out piano covers. Optional reference: YourMT3+ on the same
   files (evaluation only; check its weights' licence). Then listening on the 7 request songs.
7. **Reduction module and wrapper** (can be built in parallel with 1-5, it only needs MIDI).

## Reduction ("reduced" mode)

Principle: **change only what is not playable.** A playable passage passes through unchanged.

1. **Hand assignment:** per onset cluster (notes within 35 ms), split into left and right hand
   at the widest gap near the current split point (with hysteresis, so hands do not jump).
2. **Playability check** for each hand at each moment, all limits configurable:
   - at most 5 notes per hand; span at most a 10th (16 semitones) for a chord, an octave
     (12) for fast repeated chords;
   - no more than 2 hands' worth of simultaneously sounding notes that need fingers (held notes
     count; pedal-held notes do not);
   - leaps: a hand cannot move more than ~1.5 octaves between onsets closer than ~120 ms;
   - fast repeated notes above ~12 per second on one key are thinned.
3. **Minimal fix, only where a check fails,** in this order until it passes:
   1. drop octave doublings (keep the outer one);
   2. drop doubled pitch classes in inner voices;
   3. move an inner note by an octave into the other hand if that hand has room;
   4. drop inner voices furthest from the melody and bass, keeping chord tones by importance
      (root, third, seventh before fifth);
   5. shorten held notes that block a hand (end them at the next onset of that hand).
   The top voice (melody) and the lowest voice (bass) are never removed; they may move by an
   octave only as a last resort.
4. **Report:** the number of changed onsets and why, so we can see how much was reduced.

Tests:
- property test: any input that passes the checks is returned bit-identical;
- known cases (orchestral tutti, string quartet, organ trio, the 10 PDMX samples);
- idempotence: reduce(reduce(x)) == reduce(x).

## Wrapper

`pianocover/transcribe_any.py`:
- `transcribe(audio_path, mode="full"|"reduced", model=..., device=...) -> events, report`;
- CLI `python -m pianocover.transcribe_any song.mp3 -o out.mid --mode reduced`;
- full runs Transkun (fine-tuned weights), reduced runs the same and then `reduce()`.

## Open items

- Verify licences: MusicNet audio, GuitarSet, URMP, Bach10, VSCO-2/VSCL, MuseScore_General;
  confirm the Transkun 2.0 weights licence with the author.
- Decide on MAESTRO (NC policy) vs piano replay only.
- The AMT-APC organ 1:1 fine-tune (/data/runs/apc_organ11, 2026-11-18) is a separate
  experiment; its listening results decide nothing here.
