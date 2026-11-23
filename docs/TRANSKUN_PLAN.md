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
| Dagstuhl ChoirSet | Real choir multitracks with score MIDI and F0 | Data: CC BY-NC-SA 4.0 (Zenodo 4618287; the CC BY 4.0 is the article's). NC policy. Verify the note alignment |
| GuitarSet | Real guitar, note annotations from a hexaphonic pickup (~3 h) | **Verify** (Zenodo record / GitHub). Guitar matters for this project |
| URMP | 44 chamber pieces, mixes + MIDI scores | No license found yet (Dryad doi:10.5061/dryad.ng3r749). **Verify**; if research-only, treat as NC |
| Bach10 | 10 Bach chorales, quartet recordings + MIDI | Unclear; a derivative is CC BY-NC-SA 3.0. **Verify**, else treat as NC |
| vocadito | 40 solo singing excerpts, 7 languages, notes by 2 annotators | CC BY 4.0 (Zenodo 5578807). **Real voice**, tiny |
| Choral Singing Dataset, ESMUC Choir, Cantoría | real choir / SATB quartet (+ organ) per-singer recordings with F0 and notes | CC BY-NC 4.0, NC policy (log in MODEL_LICENSES.md). **Real voice** |
| ~~PDMX organ, real recordings~~ | **Excluded (2026-11-19).** 19 organ recordings DTW-aligned to scores: listening showed the alignment breaks the rhythm and flow, which would teach wrong onsets | — |

**Gate for every tier-1 source:** labels made by aligning a score to a recording (DTW etc.) are
suspect. Before a source is used, export 3-5 pairs with a control file (recording on the left
channel, labels rendered on the right) and get the owner's OK by ear. Sources whose notes were
captured during the recording (hexaphonic pickup, per-instrument tracks) are preferred.
MusicNet's original labels are score-aligned; MusicNetEM claims onsets within 32 ms, still gated.

Small sets are duplicated in the training pickle (Transkun samples segments by duration), so
together they make about 30% of the training segments *(target, tune after phase 2)*.
Held-out pieces from each go to the test set (split by piece, never by excerpt).

### Tier 2: piano replay (no new labels)

- Real piano audio we already have: ~2.5 k piano covers in /data/ds_all with Transkun 2.0
  transcriptions (`cover_events`). Training on the model's own output is self-distillation:
  it keeps the current piano behaviour on exactly the kind of audio users upload.
- Optional: MAESTRO v3 (CC BY-NC-SA 4.0, ~200 h, real labels) under the NC policy.
- About 25% of training segments *(target)*.
- **Built 2026-11-20** (`scripts/piano_replay.py`; the owner calls it "fortepian przeciw
  zapominaniu"): /data/piano_replay/piano, real piano covers only (/data/raw, organ_pilot,
  organ_partial; pdmx_organ excluded, its covers are renders), 16-bit WAV + Transkun pickles
  with notes and pedals (-64 / -67), split as in ds_all: 2410 train (161.4 h) + 76 val
  (4.9 h), 49.3 GB. **WAVs deleted 2026-11-21** to free disk (pickles and meta.csv kept); they
  are rebuilt from /data/raw/cover etc. by re-running the script (ffmpeg decode, same image). Far more than the 25% share needs; sample it in training. Note: transkun 2.0.1
  `DatasetMaestro.sampleSlice` unpacks `fetchData` wrongly (3 values, expects 2); `fetchData`
  itself works on this set.

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
| **Vocal** (solo song + accompaniment, SATB choir + organ / piano / orchestra) | PDMX `has_lyrics`, tracks named Voice / Soprano / Alto / Tenor / Bass ... (987 candidates after the PD filters, `/data/meta/vocal_candidates.pkl`) | `scripts/vocal_synth.py` written 2026-11-19: vocal lines sung by SoulX-Singer (Apache-2.0, score control: the model makes its own vibrato and slides), accompaniment by fluidsynth; words from PD texts laid onto the notes. **5 samples rendered 2026-11-20** (Downloads/pianocover_vocal_samples: mix, control file, target MIDI; runs offline in `pianocover:voice`, weights loaded with `weights_only=True`): Amazing Grace (40679), Ave Verum / organ (190160), Gute Nacht (204049), Handel Eternal Source (234565), Lacrymosa SATB + piano (227929); owner: "sounds fine". **Changes 2026-11-21:** voices from vocadito (`scripts/vocadito_prompts.py`, 8 English singers, CC BY; owner OK), a solo line is transposed toward its voice; 70% of renders drop accompaniment notes that double the voice (same onset ±50 ms, same pitch or octave; `--undouble-p`), since many scores double the melody in the piano; score_synth swaps only to instruments whose range covers the line (no trumpet part on a tuba). Samples after the fixes: Downloads/pianocover_vocal_v2. **Full run started 2026-11-21** (container `pc-vocal`, /data/synth_vocal, log /data/synth_vocal.log): 879 of the 987 candidates after a PD re-check (modern songs and arrangements dropped) and the content filter (Dido witches, Erlkönig, Moorhexe, Der Tod und das Mädchen); resumable per piece (survives hibernation; `<out>/PAUSE` pauses it), 2 workers sharing the GPU (bit-identical output, about 1.2× throughput), VRAM ≤ 6 GB. Measured: singing is about 97% of the time; SDPA attention was rejected (only 4% faster, audio not identical). **Owner's changes 2026-11-21 (from piece 239):** (1) ready-made voice data replaces about 17.5 h of the render (see the voice-data row); (2) long scores are cut to a window of about 3 min (`--max-sec 180`). It starts at a vocal phrase start and ends at a phrase end; the beginning, the last ~3 min (climax, ending) and a random middle are each a third of the cuts. The chosen window is in meta `cropped_at`; (3) `skip_ids.txt`: 236 SATB pieces are not sung (random, seed 1121, plus 14 that already have a voiceless synth_tk2 render); (4) 222 of them are rendered voiceless on the CPU instead (`--voiceless`, container `pc-voiceless`, /data/synth_vocal/voiceless, about 4 s per piece; the vocal tracks are played by their MIDI programs); (5) about 16 h remain to be sung: 145 solo, 88 duet and 169 SATB pieces. |
| **Voice data** (ready-made, real or synthetic voice with notes) | VocalSet + Annotated-VocalSet (CC BY), Children's Song Dataset (CC BY-NC-SA), HumTrans (CC BY-NC), ACE-KiSing (CC BY-NC), ChoralSynth (CC BY-SA); licence check in MODEL_LICENSES.md | Owner 2026-11-21: about 17-18 h. ChoralSynth 3 h, pieces with organ first; HumTrans 3 h, one take per person and melody; all of VocalSet that passes the label check (written MIDI vs estimated MIDI), about 6-8 h; CSD one key per song, about 2.4 h; ACE-KiSing 2 h, songs checked against the content filter. About half of the solo voice is mixed with a synth_tk2 excerpt whose key matches the voice (Krumhansl; relative major/minor counts; at most ±2 semitones by resampling), so the target is complete (voice + accompaniment notes); the other half stays a cappella. Scripts: `scripts/voice_fetch.py` (downloads only what is used, md5 checked, resumable), `scripts/voice_data.py` → /data/voice_data/<src>. ChoralSynth has only ~0.8 h of full mixes (its 3.8 h counts every voice track), so it gives the full mix plus one variant with a subset of voices, about 1.5 h. pyarrow 17 (Apache-2.0) is in /data/ext/pyarrow_lib for the ACE parquet files. **Risk (owner 2026-11-21):** mixes without a harmonic link could weaken the use of chord context. A second capacity-test variant with the mixes runs **only if the first variant is weak**; real-song validation decides in the end |

GM soundfont "Choir Aahs" (52-54) is the only voice in the first renders and sounds nothing
like singing, so the vocal family replaces it with neural singing. Risks: sung pitch drifts
from the score (scoops, vibrato), so the target is the score note; voice timbres are only 8
short vocadito phrases (amateur recordings), so voice variety is limited.

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
transkun 2.0.1). Mono 44.1 kHz is about 0.32 GB per hour, so 100 h is about 32 GB. The Docker
disk is **D:\DockerDisk\DockerDesktopWSL\disk\docker_data.vhdx** (not C:; not sparse, so it
never shrinks by itself). On 2026-11-20 it reached 205 GB and filled D: (13 MB free): I/O
errors, Docker hung. Cleanup on 2026-11-21 (owner's go): old AMT-APC data that is in the F:
backup, unpacked tier-1 folders, test renders and the piano replay WAVs removed; volume 185 →
113 GB used, so about 90 GB is free inside the file before it grows again. Check `df -h /d` and
the volume before every large write. Transport to the 5070 machine: FLAC on the USB drive,
unpacked to 16-bit WAV there.

Size target: about 60-100 h synthetic + all of tier 1 + tier 2 sampled *(estimate)*.

## Phases

1. **Data, tier 3 first (CPU, runs in the background):** generalise `scripts/organ_synth.py`
   into `scripts/score_synth.py` (families, 44.1 kHz, Transkun pickle, augment options).
   Listen to 5 samples per family before the full run.
   **Done 2026-11-19:** /data/synth_tk2, 1406 renders, 78 h train + 3.7 h val, 22.9 GB, checked
   (16-bit WAV, lengths, notes). Drums re-rendered louder after listening (CC7 gain bug).
2. **Capacity test (about 1-2 h on the 3060 Ti, VRAM ≤ 5 GiB, including batch size and
   checkpointing):** fine-tune 2.0.pt on 2-3 h of synthetic data. Pass: training-set onset F1
   ≥ 0.95. Fail → larger config or a different architecture, before collecting more data.
   **Script ready 2026-11-21:** `scripts/tk_capacity.py`. It uses a 2.5 h subset balanced over
   the 7 families (one 60 s segment per piece) and plain Transkun 2.0.pt. The batch size is
   found automatically under the VRAM cap, with an effective batch of 8. By the owner's
   request (2026-11-21), the cap is set to the maximum that does not make the computer stutter:
   free VRAM at start minus 1 GB, but never below 5 GB. If less than 5 GB is free, the
   script refuses to start. Training stops at
   F1 0.95 or after 120 training minutes. The run is resumable. Its output is a step-0
   baseline, per-family train F1, and val F1 as a hint. A CPU smoke test passed, including
   resume. Not run on the GPU yet: the vocal render is paused first (PAUSE file).
   **Owner 2026-11-23: run it right after the dataset is done.** Container `pc-capacity`
   (pianocover:dev, 6 CPUs, 8 GB RAM, restart unless-stopped) waits for "ALL DONE" in
   /data/synth_vocal.log, then runs the test (log /data/runs/tk_capacity.log, marker
   /data/runs/tk_capacity/DONE so it does not run twice).
   **Result 2026-11-23 (measured): PASS.** Train onset F1 0.771 at step 0 → 0.953 at step 750
   (35.5 training minutes, 2.58 h subset, batch 3 × accum 3, peak VRAM 4.44 GB of a 5.96 GB
   cap). Per family at the end: guitar 0.998, piano 0.984, chamber 0.969, organ 0.946,
   choir 0.938, orchestra 0.923, band 0.916. Val hint 0.715 → 0.910. The second variant
   (voice mixes) is not needed, since the first one is not weak.
   Vocal render finished 2026-11-23: 637 pieces, train 23.84 h + val 1.43 h.
3. **Tier 1 download + licence check:** MusicNet(+EM), Dagstuhl, GuitarSet, URMP, Bach10.
   Licence rows go in MODEL_LICENSES.md; NC ones in the NC log. Convert to the same pickle.
   **Started 2026-11-19:** licences checked; `scripts/tier1_download.py` fetches the Zenodo
   sources into /data/tier1/<source>/raw (vocadito, GuitarSet annotations + mono mic,
   Dagstuhl, CSD, ESMUC, Cantoría, MusicNet; all downloaded, md5 checked, about 21 GB).
   URMP (Dryad) not added yet. Next: converters,
   then the listening gate (3-5 control files per source).
   **Converted 2026-11-23** (`scripts/tier1_convert.py` → /data/tier1_data/<src>, splits
   train/val/test by piece, audio as recorded with peak at -1 dBFS, velocity 80 since no set has
   it): vocadito, guitarset (test/val = one held-out progression per style each), dagstuhl
   (room mic STM, 22.05 kHz source; Tebe Poem = test), esmuc (room mic ORTF or the close mics
   summed; unison singers merged; Die Himmel = test), choral (Cuesta CSD: full 16-singer mix +
   one quartet variant; Niño Dios = test), cantoria (SATB mix without the organ, whose notes
   are unknown; notes estimated from each singer's pYIN F0, the weakest labels), musicnet
   (original DTW labels; official test pieces = test; MusicNetEM not used, see
   MODEL_LICENSES.md). Hours (train / val / test): musicnet 29.8 / 1.4 / 2.9 (all recordings
   of an official test composition go to test), guitarset 2.0 / 0.5 / 0.5, dagstuhl
   0.5 / 0.04 / 0.1, esmuc 0.4 / 0.02 / 0.06, cantoria 0.5 / 0.04 / 0.03, choral 0.18 / 0 / 0.06,
   vocadito 0.17 / 0.02 / 0.03; 11.8 GB. Gate files (4 per source, 5 for musicnet, 45 s, left recording / right labels on piano)
   in Downloads\pianocover_tier1_gate, **awaiting the owner's listening**.
   URMP: Dryad now refuses scripted downloads (API 401, web 403); the owner can fetch
   Dataset.tar.gz (12.1 GB, md5 8452b1a734fad6f3c10bd19940bc09b4) in a browser if wanted.
   Bach10: licence still unverified, not converted.
4. **Tier 2:** piano replay pickle from /data/ds_all (audio + `cover_events`).
   **WAVs rebuilt 2026-11-23 for a subset** (`piano_replay.py --train-hours 40`, seed 0): 597
   train covers (40.1 h) + all 76 val (4.9 h). The pickles list only these; the full list of
   2026-11-20 is kept as meta_full_2026-11-20.csv.
   **Backup before training (ask the owner first, do not start without an explicit yes):** copy
   the repo and **only what the Transkun training needs**, not the whole volume, to the USB
   drive, F:\pianocover-backup:
   - /data/synth_tk2 (tier 3 renders, pickles, meta);
   - the tier 1 data as converted for training (audio + pickles of the sources that passed the
     gate), plus their licence notes;
   - the tier 2 piano replay set: its pickle and only the cover audio files it references
     (not all of /data/raw);
   - anything else the training command reads (noise / IR folders, split files).
   Not needed: AMT-APC data and spectrograms (/data/ds*, /data/runs), raw web downloads not in
   the replay set, PDMX, aria, fma. The full-volume snapshot already on F: (pc-data.tar,
   2026-11-17) stays as it is; the new copy is a separate `pc-data-transkun.tar`.
   Method as for the earlier backups: the repo with `robocopy /MIR` into
   F:\pianocover-backup\repo; the data as a tar stream through stdout (Docker cannot mount F:
   directly) written as `pc-data-transkun.tar.part`, its file list verified against the
   source, then renamed. No writes to these folders while the tar runs (stop GPU and data jobs
   first). Before asking, list the folders with their sizes, the expected time, and the free
   space on F:.
5. **Training:** init 2.0.pt, lower max_lr than the default 2e-4 (start at 5e-5), mixed
   sampling as above, validation per source every N steps. Memory: ≤ 5 GiB VRAM; measure batch
   size × segment length × checkpointing first (`bench`-style, like `scripts/bench_matrix.sh`).
6. **Evaluation:** note F1 (onset; onset+offset) per source on held-out **real** data:
   MusicNet test, Dagstuhl subset, URMP/Bach10 subset (each only if it passed the gate), plus a piano regression
   check against 2.0.pt on held-out piano covers. Optional reference: YourMT3+ on the same
   files (evaluation only; check its weights' licence). Then listening on the 7 request songs.
7. **Reduction module and wrapper** (can be built in parallel with 1-5, it only needs MIDI).
   Reduction in progress (see its status below); wrapper not started.
8. **Pianist Transformer, only after Transkun is trained and passed phase 6.**
   [yhj137/PianistTransformer](https://github.com/yhj137/PianistTransformer) (ICML 2026):
   score MIDI → expressive performance MIDI (timing, velocity, articulation, pedal); 135 M
   parameters, about 270 MB, runs on CPU; code and weights Apache-2.0 (HF metadata; training
   data not named, which is no blocker under the 2026-11-12 policy). Two uses:
   1. **More realistic synthetic piano data, second Transkun round:** pass the piano (and
      chamber) scores through it before rendering, so the audio has human rubato, dynamics and
      pedal instead of a metronome. The pair stays 1:1: the audio is rendered from the same
      humanised MIDI that becomes the target. Compare against the first-round model on the real
      piano test data (phase 6); keep it only if piano F1 does not drop.
   2. **A "performance" version of the output** (wrapper option, below): the reduced,
      quantised result rendered expressively for playback, next to the quantised one for
      learning / sheet music. Its input must be score-like (quantised): reduced or teacher
      output, not raw transcription, which already has performance timing.
   Before use: read the paper's own numbers (the README only says "indistinguishable from human
   pianists" in the authors' listening test) and a 1-hour listening test on the 5a samples and a
   few synth_tk2 piano renders. README warning: very low temperature gives extreme expression or
   repetitive patterns.

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

**Status 2026-11-19** (`pianocover/reduce.py`, 21 tests in `tests/test_reduce.py`; samples in
Downloads/pianocover_perkusja_test, variants 2-5b, listened to by the owner):
- done: hand assignment, playability check (5 notes / a 10th per hand), minimal fixes 1, 2, 4,
  5 above; notes held longer than 0.3 s count as pedal-held (`Limits.pedal_after`; without it
  an organ trio lost 61% of its notes); the two outer voices are never removed;
- not done: step 3 (moving a note to the other hand), leap limits;
- repeated notes and fast textures, after listening:
  - rejected: per-key repetition rules ("hold" missed fast sequences, octave "tremolo" did not
    help playability, "half" left gaps) and per-hand rate limits on a beat grid (4a/4b: even
    after fixing bar-boundary gaps, beat_this jitter and phase, they flattened the piece's
    characteristic details);
  - current candidate `simplify()` (5a / 5b): only REPETITIVE fast textures longer than a beat
    are simplified (tremolo / repeated chords → chord on the beats or octave alternation;
    Alberti / ostinato on <= 4 keys → same figure on a coarser grid; trills → mordent);
    melody, short bursts, scales and passages stay. Awaiting the owner's verdict.
- beat grid: beat_this beats are regularised first (local linear fit, then phase to strong
  onsets): raw beats jittered ±40 ms and sat 11-39 ms early on the test renders.

## Wrapper

`pianocover/transcribe_any.py`:
- `transcribe(audio_path, mode="full"|"reduced", model=..., device=...) -> events, report`;
- CLI `python -m pianocover.transcribe_any song.mp3 -o out.mid --mode reduced`;
- full runs Transkun (fine-tuned weights), reduced runs the same and then `reduce()`;
- later (phase 8): `--perform` adds a Pianist Transformer rendering of the reduced result for
  playback; the quantised file stays the main output.

## Open items

- Verify licences: Bach10, VSCO-2/VSCL, MuseScore_General (MusicNet, GuitarSet, URMP done
  2026-11-19);
  confirm the Transkun 2.0 weights licence with the author.
- Decide on MAESTRO (NC policy) vs piano replay only.
- The AMT-APC organ 1:1 fine-tune (/data/runs/apc_organ11, 2026-11-18) is a separate
  experiment; its listening results decide nothing here.
