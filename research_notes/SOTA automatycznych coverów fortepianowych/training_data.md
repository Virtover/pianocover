# Training data for automatic piano cover / arrangement systems (SOTA as of 2026-11)

Scope: how Pop2Piano, PiCoGen, PiCoGen2, AMT-APC, Etude, and the 2026 successors built their training data, what they report about data quality, and which datasets and metrics are usable. Dates are marked; anything before 2024 is labelled "older".

## 1. How the main systems collected, aligned, transcribed and filtered pairs

### Takeaway
Every audio-to-piano-cover system since 2022 uses the same basic recipe: YouTube song/cover pairs, a piano AMT model to transcribe the cover, and synctoolbox DTW to align. They differ mainly in the time grid (beat-quantised, measure-level, or frame-level) and the filters. Numeric filters are simple: melody chroma accuracy (MCA), a length-difference limit, and in Etude a warping-path-std limit. Dataset sizes run from about 1.3k covers (AMT-APC) to about 5.5k pairs (PiCoGen2) and 4.75k pairs / 500 h (Etude).

### Cited Findings

**Pop2Piano (older: arXiv Nov 2022, ICASSP 2023)**
- Collected 5,989 piano covers from 21 arrangers on YouTube, plus the matching pop songs. After synchronisation and filtering, 4,989 pairs (307 h) remain as the "PSP" set ("~300 h" in the text) — [Pop2Piano, ar5iv](https://ar5iv.labs.arxiv.org/html/2211.00895)
- Transcription: covers were transcribed with Kong et al.'s high-resolution piano transcription model — [Pop2Piano](https://ar5iv.labs.arxiv.org/html/2211.00895)
- Alignment: synctoolbox gives a warping path, and note times are moved by linear interpolation along it. Beats are tracked on the pop audio with Essentia, and note onsets and offsets are quantised to the nearest 8th-note beat. If an onset and offset fall on the same beat, the offset is pushed to the next beat — [Pop2Piano](https://ar5iv.labs.arxiv.org/html/2211.00895)
- Filter: a pair is dropped if MCA ≤ 0.15 or if the audio lengths differ by ≥ 20%. MCA is computed on vocals separated with Spleeter, with f0 from librosa pYIN (44.1 kHz, hop 1024). About 17% of pairs (1,000) were removed in total; no per-criterion breakdown is given — [Pop2Piano](https://ar5iv.labs.arxiv.org/html/2211.00895)
- PSP is not distributed as audio or MIDI. The repo ships a `train_dataset.csv` list and a downloader script, and I found no license file in the repo — [pop2piano GitHub](https://github.com/sweetcocoa/pop2piano)

**PiCoGen (ICMR 2024, arXiv 2407.20883)**
- Needs no paired data. Stage 1: SheetSage, used off the shelf, turns audio into a lead sheet. SheetSage was trained on about 40k HookTheory audio/lead-sheet pairs — [PiCoGen](https://arxiv.org/html/2407.20883)
- Stage 2: the performer is trained on Pop1K7 (about 1,700 piano covers of anime, K-pop and Western pop). Its lead-sheet inputs are derived symbolically from each piano performance (skyline-style melody extraction following Wu & Yang 2023) — [PiCoGen](https://arxiv.org/html/2407.20883)
- The authors argue that scaling is easier because only piano covers are needed, with no original-song audio — [PiCoGen](https://arxiv.org/html/2407.20883)

**PiCoGen2 (ISMIR 2024, arXiv 2408.01551)**
- Rebuilt Pop2Piano's dataset from its released code: 5,844 pairs, of which 5,503 remain after filtering (MCA < 0.05 or length difference > 15%) — [PiCoGen2](https://arxiv.org/html/2408.01551)
- "Weak alignment": notes are not remapped. Beats come from Beat Transformer, and Pop2Piano's warping path maps piano beats to the nearest song beat; the data is then cut into bars. Bars whose first and last beats map to the same song beat are discarded, which removes about 50% of bars — [PiCoGen2](https://arxiv.org/html/2408.01551)
- Transfer learning: pre-training (100K steps) uses piano-only data, with SheetSage lead sheets as input, taken from the covers in the 5,503 pairs. Fine-tuning (70K steps) uses the paired data, with loss weight α = 0.25 on the piano-only loss — [PiCoGen2](https://arxiv.org/html/2408.01551)
- Test set: 95 pairs (19 C-pop, 20 K-pop, 16 J-pop, 20 anime, 20 Western) — [PiCoGen2](https://arxiv.org/html/2408.01551)

**AMT-APC (arXiv 2409.14086, v2 revised 2026)**
- 332 songs and 1,267 covers from YouTube, collected following the Pop2Piano method. Songs were chosen from Uta-Net artist rankings, i.e. J-pop — [AMT-APC](https://arxiv.org/html/2409.14086v2)
- Aligned "using synctoolbox". Cover MIDI was extracted "using the AMT model". The fetched text does not name the model in that sentence; the base model of the system is hFT-Transformer (inference: most likely their own AMT, which is the user's premise) — [AMT-APC](https://arxiv.org/html/2409.14086v2)
- Filter: covers in a different key or with poor audio quality were removed, automatically and by hand. No numeric thresholds are given — [AMT-APC](https://arxiv.org/html/2409.14086v2)
- Training: 80/20 split by song (265 songs for training). Adam, lr 1e-4, batch 8, 5 epochs, about 3 h on 2×A100. Loss parameters: β = 0.75, θ_onset = 0.07, θ_frame = 0.2, θ_velocity = 0.01 — [AMT-APC](https://arxiv.org/html/2409.14086v2)
- Ablation (loss / average F1): full model 0.36 / 0.31; without AMT pre-training 0.41 / 0.27; without APC fine-tuning 0.92 / 0.12; without style vector 0.43 / 0.30 — [AMT-APC](https://arxiv.org/html/2409.14086v2)

**Etude (arXiv 2509.16522, ICASSP 2026; v2 Jan 2026)**
- About 7,700 pop/cover audio pairs, mostly J-pop and K-pop. After filtering, 4,752 pairs (about 500 h) remain — [Etude](https://arxiv.org/html/2509.16522v2)
- Filter: a pair is dropped if the lengths differ by more than 30 s or if WP-std (warping-path standard deviation, a metric taken from Music2MIDI) exceeds 1.0 — [Etude](https://arxiv.org/html/2509.16522v2)
- Alignment: a modified version of PiCoGen2's weak alignment. Beat-Transformer runs on the source audio only. Beat timestamps are mapped to the cover with MrMsDTW (synctoolbox), and pairs are then aligned at measure level — [Etude](https://arxiv.org/html/2509.16522v2)
- The fetched text does not name the AMT model used for the covers.
- Architecture and data flow:
  - The Extractor is adapted from AMT-APC: one-to-one pairs, no style vector, a lowered θ_matrix, 10 epochs.
  - The Decoder is a 25.5M-parameter GPT-NeoX (8 layers, hidden size 512). It is trained on bar-interleaved feature/target "Tiny-REMI" sequences with a 4-bar context, for 100 epochs at batch 128.
  - Tiny-REMI keeps only Bar, Pos (16th notes), Note, Duration and Grace tokens; tempo, chord and velocity tokens are dropped.
  - — [Etude](https://arxiv.org/html/2509.16522v2)
- Test set: 100 unseen songs, split evenly across C-pop, J-pop, K-pop and Western pop — [Etude](https://arxiv.org/html/2509.16522v2)

**Music2MIDI** is cited by Etude as the source of the WP-std metric and is included as a baseline. I could not find the paper itself.

**Cross-modal bootstrapping (Zhao, Xia, Wang, Wang; ISMIR 2026, arXiv 2608.03050)**
- Trains on POP909 (909 piano arrangements with band audio, split 90/5/5 by song) and PIAST (about 8K piano recordings with transcriptions). The training unit is a 10 s audio clip paired with a 4-bar MIDI segment taken near the clip centre with up to ±1 s shift, randomly transposed to all 12 keys — [Zhao et al. 2026](https://arxiv.org/html/2608.03050v1)
- Uses no cover-transcription step of its own. A frozen MusicGen-Large (1.7B) and a frozen MuseCoco-xLarge (1.2B) are bridged by a trainable 186M Q-Former. Training has two stages: contrastive alignment of audio style with symbolic expression, then generation. At inference the model takes a SheetSage lead sheet (content) plus a reference audio clip (style) — [Zhao et al. 2026](https://arxiv.org/html/2608.03050v1)
- PIAST: 9,673 YouTube solo-piano tracks (jazz, classical, new-age, pop covers), transcribed with Kong et al. 2021 and beat-tracked — [PIAST](https://arxiv.org/html/2411.02551)

### Inferences
- Since 2024 the trend is away from Pop2Piano-style note remapping, which distorts rhythm, toward beat- or measure-level weak alignment on a beat grid of the source song (PiCoGen2, then Etude). Etude, the current subjective leader, uses Beat-Transformer + MrMsDTW + measure-level alignment + WP-std ≤ 1.0. That combination is the most concrete published recipe to copy.
- None of these papers checks that the cover transcriber matches the model's label convention. AMT-APC (inferred) used its own AMT both to build its labels and to initialise the model. Fine-tuning on Transkun labels therefore changes the target distribution (onset, offset and velocity conventions, pedal handling), which by itself could lower validation loss on the new labels while making the audible result worse. This is an inference, not a measured result.

### Gaps
- Pop2Piano does not report how many pairs each filter removed, and Etude does not report per-filter counts.
- The AMT model Etude used for covers was not stated in the fetched text. The AMT-APC sentence on this is ambiguous.
- I could not retrieve the Music2MIDI paper, the original source of WP-std, so the definition of WP-std was not verified.
- None of these datasets is publicly released as audio/MIDI pairs; Pop2Piano releases only a URL list.

## 2. Effect of data quality vs quantity, alignment errors, transcription mismatch and noisy pairs

### Takeaway
No paper runs a controlled quality-vs-quantity or transcriber-mismatch ablation. The measured evidence is indirect:
- Real cover pairs beat an alternative dataset (Pop2Piano: PSP vs POP909).
- Alignment is measurably imperfect, and about 50% of bars are discarded under weak alignment (PiCoGen2).
- Swapping in Etude's larger, better-filtered data gave a non-significant gain for PiCoGen2, while Etude's architecture gave a significant one (Etude ablation).
- AMT pre-training matters a lot for the AMT-APC model (AMT-APC ablation).

### Cited Findings
- **Pop2Piano, PSP vs POP909 (measured):**
  - MOS: Pop2Piano trained on PSP 3.216, trained on POP909 2.856, human 3.771.
  - 70% of listeners preferred the PSP model (Wilcoxon p = 3.34e-05). Win rates against the human cover: 29.6% (PSP) vs 22.24% (POP909). 25 listeners, 10 s clips.
  - The authors attribute the difference to the dataset — [Pop2Piano](https://ar5iv.labs.arxiv.org/html/2211.00895)
- **Pop2Piano style control (measured):** generated note density tracks each target arranger's density with R² = 0.804. Conditioning on arranger identity therefore absorbs between-arranger variation, and similar per-cover conditioning appears later as AMT-APC's style vector — [Pop2Piano](https://ar5iv.labs.arxiv.org/html/2211.00895)
- **PiCoGen2, alignment quality (measured):**
  - Original covers vs songs have duration, tempo and IOI deviation ratios of 1.10 ± 0.12, 1.16 ± 0.25 and 1.14 ± 0.17, where 1.00 would be perfect.
  - The authors say note remapping distorts rhythm and loses piano information, and that weak alignment still leaves beat-level errors.
  - About 50% of bars were dropped as invalid, which suggests Pop2Piano's alignment "may not be robust enough" (authors' interpretation) — [PiCoGen2](https://arxiv.org/html/2408.01551)
- **Etude ablation, data vs architecture (measured, 30 listeners):**
  - PiCoGen2 retrained on Etude's dataset scored OVL 3.18 vs 2.98 for the original PiCoGen2, a non-significant difference (p = 0.0656).
  - Etude Default (OVL 3.55) vs PiCoGen2 on the same data: p = 0.0007.
  - The authors attribute the gain mainly to architecture — [Etude](https://arxiv.org/html/2509.16522v2)
- **AMT-APC ablation (measured):** without AMT pre-training, average F1 drops from 0.31 to 0.27 (loss 0.36 → 0.41). This shows that the AMT initialisation, and with it the label convention used in pre-training, carries much of the performance. The fine-tuning set is small (1,267 covers) — [AMT-APC](https://arxiv.org/html/2409.14086v2)
- **Transcription error propagation in two-stage systems (author interpretation):** Zhao et al. 2026 attribute lower MCA and chord accuracy on POP909 to SheetSage lead-sheet errors carrying into the output — [Zhao et al. 2026](https://arxiv.org/html/2608.03050v1)
- **Pop2Piano's caution:** higher MCA does not necessarily mean a better cover — [Pop2Piano](https://ar5iv.labs.arxiv.org/html/2211.00895)

### Inferences
- The best evidence suggests that once you have a few thousand reasonably aligned pairs, architecture and representation (beat/measure grid, structure stage) matter more than adding or cleaning pairs. Etude's data swap moved OVL by about +0.2, not significantly.
- For AMT-APC-style fine-tuning, label consistency with the pre-training transcriber is probably more important than pair count. AMT-APC got its result with only 1,267 covers. This is inference; no paper isolates the effect.
- The usual way to reduce noisy-pair damage is to discard data aggressively: about 17% of pairs (Pop2Piano), about 6% of pairs plus about 50% of bars (PiCoGen2), and about 38% of pairs (Etude, 7.7k → 4.75k).

### Gaps
- No paper reports a curve of quality against dataset size, or compares transcribers (Kong vs Transkun vs hFT) as the label source.
- No paper measures the effect of DTW alignment error on subjective quality directly.

## 3. Synthetic and two-stage approaches (lead sheet / structure intermediate, symbolic arrangers, rendering, self-training)

### Takeaway
Two-stage pipelines (audio → lead sheet or features → symbolic piano model) avoid the need for paired data in the second stage and win in listening tests: PiCoGen over Pop2Piano, PiCoGen2 over Pop2Piano, Etude over everything. The second stage can be pre-trained on piano-only data, either covers or Pop1K7/PIAST transcriptions. I found no published cover system that creates training pairs by rendering band audio from MIDI. Self-training/distillation for covers is unreported. Symbolic reduction work (orchestra to piano) uses semi-supervised learning because paired data is scarce.

### Cited Findings
- **PiCoGen, lead-sheet intermediate (measured, 30 listeners, Pop subset):**
  - PiCoGen: OVL 3.35, MCA 0.17. Pop2Piano: OVL 2.55, MCA 0.25.
  - The difference was not statistically significant because of the large SD.
  - Ablations: without chord tokens OVL 3.05; with MIDI-like tokens instead of REMI OVL 3.05 — [PiCoGen](https://arxiv.org/html/2407.20883)
- **PiCoGen2, piano-only pre-training then paired fine-tuning (measured, 52 listeners):**
  - OVL: PiCoGen2 3.48 ± 0.93, Pop2Piano 2.71, human 4.30. SI: 3.55 vs 2.63. FL: 3.66 vs 2.72 — [PiCoGen2](https://arxiv.org/html/2408.01551)
- **Etude, Extract → strucTUralize (beats/measures) → DEcode (measured, 101 listeners, OVL):**

  | System | OVL |
  |---|---|
  | Human | 3.92 |
  | Etude Default | 3.50 |
  | Etude Extractor-only | 3.33 |
  | PiCoGen2 | 2.97 |
  | AMT-APC | 2.46 |
  | Music2MIDI | 2.27 |

  - All Etude variants beat the baselines with corrected p < 0.001.
  - Etude Extractor-only, which is essentially AMT-APC-style output, gets the highest similarity score (SI 3.41) but lower fluency (FL 3.31 vs 3.73 for Default). The structure and decoding stage buys fluency — [Etude](https://arxiv.org/html/2509.16522v2)
- **Cross-modal bootstrapping (2026), symbolic LM conditioned on audio style:**
  - Trained on POP909 + PIAST, with lead-sheet content from SheetSage.
  - Text-to-audio retrieval Acc@1: 71.4% vs 3.4% for CLaMP3.
  - Better than PiCoGen2 on texture, groove and voice metrics and on out-of-domain TA (79.4 vs 57.6 on Ballroom/GTZAN), but lower POP909 MCA (32.0 vs 39.3) and chord accuracy (33.3 vs 40.0).
  - Subjective study (21 participants): significantly better than all baselines on Coherence and Musicality, comparable to PiCoGen2 on Naturalness — [Zhao et al. 2026](https://arxiv.org/html/2608.03050v1)
- **Piano reduction (Dec 2025, arXiv 2512.21324):**
  - The paper finds no large dataset of orchestral scores with matching piano reductions.
  - It uses a semi-supervised BERT pipeline: "simplification followed by harmonization", with per-note keep/drop classification and labels from note-matching against human arrangements on LOP.
  - Data resources it lists: LOP (about 200 aligned orchestra/piano pairs), GiantMIDI-Piano (about 10k pieces, "mediocre quality" from transcription), Symbolic Orchestral Database (about 6k pieces) — [Piano reduction BERT](https://arxiv.org/html/2512.21324v1)
- **Larger-scale symbolic corpora:** Gangqinpu (Oct 2026) has 18,786 score–performance pairs and 1,007.1 h of audio across multiple genres including pop. It is built by AMT + score alignment + a classifier that filters out non-expressive performances, and is aimed at transcription and rendering, not cover generation — [Gangqinpu](https://arxiv.org/abs/2610.34694)

### Inferences
- The setup with the strongest evidence is: a frame-level AMT-APC-style extractor (high similarity), then a beat/measure-grid symbolic decoder trained on cover transcriptions (fluency).
  - Only the decoder needs large amounts of piano data, and piano-only data works for pre-training (PiCoGen2).
  - This suggests the user's ~2.5k pairs are better spent training a decoder on beat-quantised features than fine-tuning AMT-APC on Transkun labels.
- Rendering band audio from multitrack MIDI (Lakh/Slakh-style) to create synthetic pairs is untested in this literature. POP909 already has band-audio links and is what Zhao et al. use. The rule-based reduction teacher matches the "simplification + harmonization" family, which even the reduction literature treats only as a fallback.

### Gaps
- I found no paper that trains a cover model on band audio rendered from MIDI, or that uses self-training or distillation on unlabeled songs.
- I found no use of PDMX, Aria-MIDI, ATEPP or PiJAMA in pop-cover training.

## 4. Public datasets and their licenses

### Takeaway
There is no public, commercially clean, paired pop-song/piano-cover audio dataset. PSP and the Etude and AMT-APC sets are built from YouTube and released at most as URL lists. Clear licenses:

| Dataset | License | Commercial use |
|---|---|---|
| POP909 | MIT (as shown on the repo) | Appears allowed |
| PDMX | CC-BY / CC-0 (sources differ) | Yes |
| GiantMIDI-Piano | CC BY 4.0 (from a mirror), plus a download disclaimer | Probably, with attribution |
| Aria-MIDI | CC-BY-NC-SA 4.0 | No |

### Cited Findings
- **POP909:** the repo shows an MIT license. Per song it has MELODY, BRIDGE and PIANO tracks, beat/chord/key annotations, and alternate versions. The audio is not included; the 909 songs are Mandarin pop — [POP909 GitHub](https://github.com/music-x-lab/POP909-Dataset); [Zhao et al. 2026](https://arxiv.org/html/2608.03050v1)
- **PSP (Pop2Piano):** only a CSV list and a downloader; no license file found — [pop2piano GitHub](https://github.com/sweetcocoa/pop2piano)
- **Aria-MIDI:** CC-BY-NC-SA 4.0 for the data, Apache-2.0 for the pipeline tools — [Aria-MIDI HF](https://huggingface.co/datasets/loubb/aria-midi)
- **PDMX:** described as CC-BY and "commercially viable" in one source and listed as CC-0/public domain in a table, so the label is inconsistent — [PDMX paper (ar5iv)](https://ar5iv.labs.arxiv.org/html/2409.10831)
- **GiantMIDI-Piano:** CC BY 4.0 per a mirror listing. Users must accept a disclaimer, and the source recordings are YouTube — [GiantMIDI mirror](https://sourceforge.net/mirror/giantmidi-piano)
- **PIAST:** 9,673 YouTube tracks with Kong-transcribed MIDI. No license statement was found — [PIAST](https://arxiv.org/html/2411.02551)
- **Pop1K7:** about 1,700 transcribed pop piano covers, used by PiCoGen. Its license was not checked — [PiCoGen](https://arxiv.org/html/2407.20883)
- **LOP:** about 200 orchestra/piano pairs — [Piano reduction BERT](https://arxiv.org/html/2512.21324v1)

### Inferences
- For a commercially usable product, the plausible clean sources are POP909 (MIT, but audio links point to copyrighted recordings) and PDMX (public-domain scores). Everything built from YouTube covers carries copyright risk in the underlying recordings regardless of the annotation license.

### Gaps
- I did not verify the licenses of Pop1K7, PIAST, Gangqinpu, ATEPP or PiJAMA, nor whether the Etude or AMT-APC datasets have been released.

## 5. Evaluation metrics and correlation with human judgment

### Takeaway
No paper reports correlation coefficients between objective metrics and listening tests. The data shows that MCA and similarity metrics such as ChromaCoverID Q_max and WPD do not track preference:
- AMT-APC is best on Q_max yet worst on human OVL.
- PiCoGen beats Pop2Piano in listening tests while having lower MCA.

Etude proposes rhythm metrics (RGC, IPE) and argues that closeness to human-cover values is the signal. That claim is supported only qualitatively.

### Cited Findings
- **AMT-APC, Q_max** (ChromaCoverID; lower = more similar; notes < 0.08 s removed; FluidSynth/FluidR3_GM render): human 0.038, AMT-APC 0.035, PiCoGen2 0.054, Pop2Piano 0.090 — [AMT-APC](https://arxiv.org/html/2409.14086v2)
- **Etude, Table 1 (WPD / RGC / IPE / OVL):**

  | System | WPD | RGC | IPE | OVL |
  |---|---|---|---|---|
  | Human | 0.49 | 0.042 | 10.13 | 3.92 |
  | Etude Default | 0.21 | 0.020 | 9.02 | 3.50 |
  | PiCoGen2 | 1.00 | 0.059 | 7.97 | 2.97 |
  | AMT-APC | 0.09 | 0.114 | 10.69 | 2.46 |
  | Music2MIDI | 0.18 | 0.160 | 8.94 | 2.27 |

  - The paper claims these metrics "correlate with human perception" but gives no coefficients, and its conclusion admits a gap between objective metrics and perception.
  - Rater expertise did not significantly change OVL (p = 0.070) — [Etude](https://arxiv.org/html/2509.16522v2)
- **PiCoGen2 (MCA / GS / H4):** Pop2Piano 0.42 / 0.86 / 2.46; PiCoGen2 0.17 / 0.84 / 2.46; human 0.16 / 0.81 / 2.59. Pop2Piano's MCA is far above the human cover's, yet it is rated far lower (OVL 2.71 vs 4.30) — [PiCoGen2](https://arxiv.org/html/2408.01551)
- **Pop2Piano:** MCA of human covers is 0.3493 (PSP) and 0.395 (POP909F). The authors warn that higher MCA does not mean a better cover — [Pop2Piano](https://ar5iv.labs.arxiv.org/html/2211.00895)
- **Zhao et al. 2026:** objective metrics are MCA, chord accuracy, groove pattern consistency, VCC and TA. The paper states that the subjective results "align with the objective evaluation" without any correlation analysis — [Zhao et al. 2026](https://arxiv.org/html/2608.03050v1)

### Inferences
- In Etude's table, rhythmic grid coherence (RGC) orders the systems roughly the same way as OVL among the models: Etude 0.020 → PiCoGen2 0.059 → AMT-APC 0.114 → Music2MIDI 0.160, with OVL 3.50 → 2.97 → 2.46 → 2.27. Similarity metrics (WPD, Q_max, MCA) do not. This is my own reading of 4–6 points, not a measured correlation.
- Practical implication: for model selection, use a beat-grid or rhythm-stability metric plus a small blind listening panel. Validation loss or similarity metrics alone are not enough, which matches the user's experience of validation loss improving while results got worse by ear.

### Gaps
- No published study reports Pearson or Spearman correlations between objective metrics and MOS for piano covers.
