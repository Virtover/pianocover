# Symbolic piano arrangers (stage 2 of audio -> lead sheet -> piano), state as of 2026-11-18

Notes on method: repo licenses and last-push dates were taken from the GitHub REST API (`api.github.com/repos/<owner>/<repo>`) on 2026-11-18. Items marked **[unverified]** come from background knowledge and could not be re-checked within the tool budget. "Older" means published before 2024.

## Q1. Which open or licensable symbolic arrangers exist, and what are their I/O, controllability, quality evidence, code/weights, licenses and GPU needs?

### Takeaway
No open arranger turns a lead sheet into a *playable, difficulty-controlled* pop piano cover, has commercially usable weights, and is backed by strong listening tests. The most relevant 2026 lead-sheet-to-piano models are D3PIA (KAIST, Feb 2026), BOSSA / "cross-modal bootstrapping" (Zhao et al., Aug 2026) and the older but MIT-licensed Fudan/NYU-Shanghai family (AccoMontage, Polyffusion, whole-song-gen, Structured Arrangement). All of them are trained on POP909, a roughly 900-song Mandarin-pop piano accompaniment dataset. Etude's decoder (MIT code, CC BY-NC-SA 4.0 weights) is the strongest "cover-style" generator, but it consumes dense AMT features rather than a lead sheet. Commercial tools (Band-in-a-Box, EZkeys 2, Scaler 3, Captain) are pattern or style engines, not learned arrangers.

### Cited Findings

**Lead-sheet -> piano accompaniment models (2024-2026)**
- **D3PIA** (Choi, Kim, Bang, Kwon, Nam, KAIST; arXiv 2602.03523, 3 Feb 2026). It is a discrete denoising diffusion model. The input is a lead-sheet piano roll (melody and chords in one channel, 16th-note grid) and the output is an 8-bar piano accompaniment piano roll. It has about 2.2M parameters and takes 1.7 s per 8 bars. — [arXiv](https://arxiv.org/html/2602.03523)
  - D3PIA results on the POP909 test set, as CA % / chord similarity % / groove similarity % (ground truth: 91.6 / 95.7 / 82.7):

    | Model | CA % | Chord similarity % | Groove similarity % |
    |---|---|---|---|
    | D3PIA | 80.1 | 93.6 | 82.1 |
    | Polyffusion | 37.5 | 54.0 | 79.9 |
    | Compose & Embellish, embellish stage | 58.1 | 70.6 | 80.8 |
    | WholeSongGen 4th stage (uses extra inputs) | 87.6 | 94.6 | 75.4 |
    | FGG | 62.0 | 77.3 | 78.9 |

    — [arXiv](https://arxiv.org/html/2602.03523)
  - The listening test was small: 13 trained listeners and 10 excerpts. The authors claim D3PIA beat all baselines, but the numbers appear only in a figure. WholeSongGen-4th had a higher CA but was perceived as more dissonant. — [arXiv](https://arxiv.org/html/2602.03523)
  - Limitations: no style control, no velocity modelling, and 8-bar windows. AccoMontage was excluded because it has no official training code. — [arXiv](https://arxiv.org/html/2602.03523)
  - The repo `jech2/D3PIA` has **no license file** (GitHub API shows license=None; last push 2026-02-10; 7 stars). The paper says the training code and samples are public, and it does not say whether weights are released. — [GitHub API](https://api.github.com/repos/jech2/D3PIA), [demo](https://jech2.github.io/D3PIA)
- **"Learning Music Style for Piano Arrangement Through Cross-Modal Bootstrapping"** (arXiv 2608.03050, 4 Aug 2026; demo page "bossa" by Zhao, so likely Jingwei Zhao et al.). This is the only arranger found that takes **a lead sheet plus reference audio**, which is exactly the stage-2 setting.
  - Architecture: a Q-Former links a frozen MusicGen-Large audio encoder to a frozen MuseCoco-xLarge symbolic LM, and a LoRA adapter conditions on the lead sheet. It generates in 4-bar windows. Training data is POP909 plus PIAST (about 8K piano recordings). — [arXiv](https://arxiv.org/html/2608.03050)
  - The evaluation is audio-in. Every system gets its lead sheet from Sheetsage (v1). Results as MCA / CA / time-signature accuracy:

    | Test set | This model | PiCoGen2 |
    |---|---|---|
    | POP909 | 32.0 / 33.3 / 83.6 | 39.3 / 40.0 / 74.4 |
    | Ballroom/GTZAN (out-of-distribution) | 17.8 / 16.3 / 79.4 | 17.2 / 15.5 / 57.6 |

    — [arXiv](https://arxiv.org/html/2608.03050)
  - The listening test had 21 participants. The authors report the model is significantly better than the baselines on coherence and musicality, and comparable to PiCoGen2 on naturalness. — [arXiv](https://arxiv.org/html/2608.03050)
  - The paper states no code or weights release; only a demo page exists. It is heavy, because it uses MusicGen-Large plus MuseCoco-xLarge. — [demo](https://zhaojw1998.github.io/bossa/)
- **Whole-Song Hierarchical Generation** (Wang, Min, Xia; ICLR 2024).
  - It is a cascade of four diffusion models: Form -> reduced lead sheet -> lead sheet -> accompaniment. The top level spans up to 256 bars; the lead-sheet and accompaniment levels work in 8-bar segments. The 4th stage can serve as a lead-sheet -> accompaniment module, but it needs extra hierarchical inputs. — [arXiv](https://arxiv.org/html/2405.09901v1), [ICLR](https://iclr.cc/virtual/2024/poster/17633)
  - Code: `ZZWaang/whole-song-gen`, **MIT**, last push 2024-04-03. Checkpoints are only partially released. — [GitHub](https://github.com/ZZWaang/whole-song-gen)
- **Polyffusion** (Min, Jiang, Xia, Zhao; ISMIR 2023, older). It is a diffusion piano-roll model conditioned on chords and/or texture, and supports inpainting.
  - Repo `aik2mlj/polyffusion`, **MIT**, last push 2024-07-16. — [GitHub API](https://api.github.com/repos/aik2mlj/polyffusion)
  - D3PIA's table shows weak chord adherence (CA 37.5%). — [D3PIA](https://arxiv.org/html/2602.03523)
- **AccoMontage family** (Zhao & Xia; older).
  - AccoMontage (ISMIR 2021) is a phrase-retrieval-plus-style-transfer piano arranger trained on POP909. Repo `zhaojw1998/AccoMontage` is **MIT** and was last pushed 2026-01-17. — [GitHub API](https://api.github.com/repos/zhaojw1998/AccoMontage)
  - AccoMontage2 (ISMIR 2022) adds chord harmonization and takes melody only. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.00353)
  - AccoMontage-3 is published as "Structured Multi-Track Accompaniment Arrangement via Style Prior Modelling" (NeurIPS 2023 per OpenReview link **[venue unverified]**). It is two stages: lead sheet -> piano (Stage 1, an AccoMontage-style arranger) -> multi-track orchestration. The repo was renamed to `zhaojw1998/Structured-Arrangement-Code` (**MIT**, last push 2026-01-17) and has a Colab demo. Its Python 3.8 / torch 1.10 environment is old. — [GitHub README](https://github.com/zhaojw1998/Structured-Arrangement-Code), [arXiv](https://arxiv.org/abs/2310.16334)
  - The AccoMontage2 repo path `zhaojw1998/AccoMontage2` returned Not Found, so its location and license are unverified.
- **Compose & Embellish** (Wu & Yang, ICASSP 2023, older). It is a two-stage Transformer: lead sheet first, then an "embellish" stage that produces the full piano. Repo `slSeanWU/Compose_and_Embellish` is **MIT** (last push 2024-09-16). In D3PIA's comparison its embellish stage reached only 58.1% CA, with 14.8% out-of-key notes. — [GitHub API](https://api.github.com/repos/slSeanWU/Compose_and_Embellish), [D3PIA](https://arxiv.org/html/2602.03523)
- **Pop Music Transformer / REMI** (Huang & Yang 2020, older). It is unconditional or prompt-based pop piano generation on REMI tokens, not a lead-sheet conditioner. Repo `YatingMusic/remi` is **GPL-3.0** and was last pushed 2022. — [GitHub API](https://api.github.com/repos/YatingMusic/remi)

**Cover-style decoders (inside audio -> piano systems)**
- **Etude decoder** (Chen & Joung, NTU; arXiv 2509.16522; ICASSP 2026 per search listing).
  - Inputs:
    - the AMT-APC-derived dense "feature" MIDI of each bar
    - up to 4 previous (feature, output) bar pairs
    - per-bar style bins: polyphony, rhythmic intensity and sustain, each low/medium/high
  - Output: Tiny-REMI piano tokens per bar.
  - Size and data: about 25.5M parameters (GPT-NeoX style), trained on 4,752 pop/cover pairs (about 500 h, mostly J-pop and K-pop).
  - There is no explicit difficulty control.

  — [arXiv](https://arxiv.org/html/2509.16522v2)
  - Licensing and hardware: repo `xiugapurin/Etude` uses a dual license. Code is **MIT**; pretrained checkpoints are **CC BY-NC-SA 4.0** ("you may NOT use the pre-trained models for commercial purposes"; commercial licensing by contacting the owner). The README asks for a GPU with ≥16 GB VRAM, or Apple Silicon with 16 GB+, and uses Spleeter for beat detection. — [LICENSE](https://github.com/xiugapurin/Etude/blob/main/LICENSE), [README](https://github.com/xiugapurin/Etude)
  - It could in principle be fed a lead sheet (melody plus chord voicings rendered as "features") instead of AMT output. That use is **untested**: the paper notes the decoder already struggles to identify the main melody. — [arXiv](https://arxiv.org/html/2509.16522v2)
- **PiCoGen v1 Performer** (Tan, Guan, Yang; ICMR 2024). It is the lead-sheet -> piano stage of a two-stage system: SheetSage lead sheet per bar, plus previous output, produces a piano bar. It lives in repo `tanchihpin0517/PiCoGen` (branch v1), which has **no license file** (API license=None; last push 2025-12-04). — [README](https://github.com/tanchihpin0517/PiCoGen), [project](https://tanchihpin0517.github.io/PiCoGen/picogen1.html)

**General symbolic foundation / text-to-MIDI models (not arrangers out of the box)**
- **Aria** (EleutherAI).
  - It is a LLaMA-3.2-1B-architecture model trained on about 60k hours of transcribed solo-piano MIDI. Checkpoints exist for the base model, piano continuation and embeddings. It is "best when continuing existing piano MIDI", so it is prompt continuation, not lead-sheet conditioning.
  - Code and weights are **Apache-2.0**; the Aria-MIDI dataset is CC BY-NC-SA 4.0.

  — [GitHub](https://github.com/EleutherAI/aria), [HF aria-medium](https://huggingface.co/loubb/aria-medium), [arXiv 2511.07268](https://arxiv.org/html/2511.07268v2)
  - It could be used for a "melody + chord voicing prompt -> continue/infill in piano style" approach, but **no evaluation of that** was found.
- **Anticipatory Music Transformer** (Thickstun et al., TMLR 2024 **[venue unverified]**). It supports infilling and accompaniment conditioned on given events, e.g. generating accompaniment for a fixed melody. Repo `jthickstun/anticipation` is **Apache-2.0** and was last pushed 2026-07-19. It was trained on Lakh MIDI (multi-instrument), so it is not piano-specific **[training data unverified]**. — [GitHub API](https://api.github.com/repos/jthickstun/anticipation)
- **Moonbeam**: a symbolic foundation model pretrained on 81.6K hours (~18B tokens) of MIDI; its code and pretrained model are said to be open. The license was **not found**. — [arXiv 2505.15559](https://www.arxiv.org/pdf/2505.15559)
- **MuseCoco** (Microsoft, 2023, older; text/attribute -> multi-track MIDI) is in `microsoft/muzic`, which is **MIT** (last push 2026-08-05). It is used as the frozen backbone in the 2026 cross-modal arranger above. — [GitHub API](https://api.github.com/repos/microsoft/muzic), [arXiv 2608.03050](https://arxiv.org/html/2608.03050)
- **NotaGen** (ABC-notation classical sheet generation, 2025). Repo `ElectricAlexis/NotaGen` is **MIT**, last push 2025-04-21. It is not lead-sheet conditioned, and its target is classical-style scores **[scope unverified]**. — [GitHub API](https://api.github.com/repos/ElectricAlexis/NotaGen)
- **MIDI-LLM** (Wu, Carlton, Mikayawa, Kim, Donahue, Huang; arXiv 2511.03942). It is text-to-MIDI built on Llama-3.2-1B, with weights at HF `slseanwu/MIDI-LLM_Llama-3.2-1B`. The GitHub license is NOASSERTION (no standard LICENSE file found), so effectively unlicensed / Llama-license terms **[unverified]**. — [README](https://github.com/slSeanWU/MIDI-LLM)
- **Text2midi** (AMAAI Lab, AAAI 2025 **[venue unverified]**): repo **MIT**, last push 2025-02-28. It is text-conditioned and not suitable for lead-sheet conditioning. — [GitHub API](https://api.github.com/repos/AMAAI-Lab/Text2midi)
- **SongComposer** (lyrics/melody LLM): repo **Apache-2.0**. It produces melody only, not piano arrangement. — [GitHub API](https://api.github.com/repos/pjlab-songcomposer/songcomposer)

**Piano reduction (multi-track -> piano)**
- AccoMontage-3 / Structured Arrangement goes the *opposite* way (piano -> band). — [README](https://github.com/zhaojw1998/Structured-Arrangement-Code)
- No 2024-2026 open piano-reduction model was verified in this pass. Older works include Crestel & Esling's "Live Orchestral Piano" (LOP, 2017, which actually projects piano -> orchestra) and rule/ML-based piano reduction papers by Takamori et al. and Nakamura & Sagayama (2015-2019) **[all unverified in this pass]**.

**Commercial tools**
- **Band-in-a-Box 2026** (PG Music) generates accompaniment from a chord chart with style engines (MIDI styles plus RealTracks audio). The 2026 manual says it can open a MIDI file and auto-interpret chords, export MusicXML with chords, and new "AI-notes" transcribes polyphonic audio to MIDI. It is a paid desktop product. Licence terms for redistributing generated output were not checked. — [BiaB 2026 manual ch.15](https://www.pgmusic.com/manuals/bbw2026full/chapter15.htm), [ch.3](https://www.pgmusic.com/manuals/bbw2026full/chapter3.htm)
- **Toontrack EZkeys 2**: its "Bandmate" takes audio or MIDI and suggests matching piano parts. "Replace MIDI" applies EZkeys playing-style MIDI patterns to your own chord sequence. The latest release note found is v2.1.1 (Oct 2024). It is a plugin with no batch API. — [product page](https://www.toontrack.com/product/ezkeys-2/), [release notes](https://toontrack.com/release-notes/release-notes-ezkeys-2)
- **Scaler 3** (Plugin Boutique, $99) offers chord/melody tools with performance patterns and External MIDI Out. It is pattern based, not a learned arranger. — [MusicTech review](https://musictech.com/reviews/plug-ins/scaler-3-review/), [Dubspot comparison](https://blog.dubspot.com/plugins/compare/ezkeys-2-vs-scaler-3)
- **Captain Plugins** (Mixed In Key) and **Orb Producer**: only older (≤2023) sources were found, and Orb's current status is unclear. — [AudioCipher](https://www.audiocipher.com/post/scaler-2-vs-captain-chords), [KVR](https://www.kvraudio.com/forum/viewtopic.php?p=7708433)

### Inferences
- For a *commercial-safe* stage 2, the realistic options are:
  - **MIT models trained on POP909**: AccoMontage, Structured-Arrangement Stage 1, Polyffusion, whole-song-gen, Compose & Embellish. Note that POP909's own license (reportedly MIT **[unverified]**) and the provenance of its arrangements should be checked.
  - **Apache-2.0 Aria / Anticipatory MT** used as infilling/continuation engines.
  - **Rule- or pattern-based arrangement**, which is what commercial tools do.
- D3PIA has the best published chord adherence and rhythm-likeness among lead-sheet -> piano models, is tiny (2.2M parameters) and would run trivially on a 3060 Ti. However, its repo has no license, so you would need to ask the authors before using it.
- Every academic arranger is trained on POP909 (Mandarin pop, accompaniment-only, the melody is a separate track). Outputs are therefore *accompaniment* textures. To get a solo piano cover you must merge the melody into the right hand yourself and handle register collisions. None models playability or difficulty explicitly; Etude's polyphony and rhythmic-intensity bins are the closest proxy.
- Etude's decoder is the strongest documented "cover" generator, but its weights are NC. For a commercial product you would retrain it on your own pairs (its code is MIT) or license the weights from the author.

### Gaps
- Weights availability and licenses for D3PIA, the cross-modal "BOSSA" model and PiCoGen v1 were not confirmed (no license files).
- I could not fetch exact listening-test means for D3PIA or BOSSA; they appear only in figures.
- I did not verify current (2026) features of BandLab SongStarter, Captain Plugins 2026, Orb Producer or Yamaha arrangement engines (e.g. Yamaha "Chord Tracker" or "Smart Pianist" audio-to-chord + style accompaniment). I found no evidence any of them exposes a lead-sheet -> solo-piano-arrangement export with a usable license.
- I found no 2024-2026 open orchestral/multi-track -> piano reduction model with released weights in this pass.
- POP909's dataset license and commercial status were not checked here.

## Q2. How good is audio -> lead-sheet extraction today, and which tools do it end to end?

### Takeaway
**SheetSage2** (m-a-p / HKUST / NYU, Oct 2026) is now clearly state of the art and end to end. From a single model it outputs vocal melody MIDI, chords, beats/downbeats, key, structure and ABC/PDF lead sheets. Vocal-melody note F1 is about 82.5% on RWC-Pop, up from 62.7% for SheetSage1. Chords reach 90% maj/min on osu2017 but 64.5% on jazz (JAAH). The weights are **CC BY-NC 4.0** (non-commercial). Permissive alternatives must be assembled from parts: beat_this (MIT), BTC (MIT), SOME (MIT), basic-pitch/MT3 (Apache-2.0); YourMT3 is GPL-3.0.

### Cited Findings
- SheetSage2 (Jiang, Yuan, Pan, Xue, Guo, Xia, LeCun; arXiv 2610.05336, 4 Oct 2026) is built on MERT-v2 with a 57.2M-parameter checkpoint. — [arXiv](https://arxiv.org/html/2610.05336v1), [HF](https://huggingface.co/m-a-p/SheetSage2)
  - Outputs: `melody_vocal.mid`, `melody_instrumental.mid`, `chords.mid`, `score.abc`, `events.json` / `.lab` (chords, beats, key, structure) and optional PDF/PNG. A `melody_only=True` mode is provided "for covers". — [HF](https://huggingface.co/m-a-p/SheetSage2)
  - Weights are **CC BY-NC 4.0**; access is gated (sign in, with rights to the MERT-v2 parent). — [HF](https://huggingface.co/m-a-p/SheetSage2)
  - Benchmarks for SheetSage2-AR, compared with the best listed baseline:

    | Task | Benchmark / metric | SheetSage2-AR | Best baseline |
    |---|---|---|---|
    | Melody | RWC-Pop vocal F1 | 82.51 | 62.71 (SheetSage1) |
    | Melody | RWC-Pop full F1 | 75.29 | – |
    | Melody | Rock Corpus vocal F1 | 67.08 | 49.19 (SheetSage1) |
    | Chords | osu2017 maj/min | 90.08 | 86.55 (ChordFormer) |
    | Chords | Chords1217 | 83.81 | 83.94 (ChordFormer) |
    | Chords | JAAH | 64.50 | 59.45 (ChordFormer) |
    | Downbeat | GTZAN F1 | 80.45 | 78.28 (Beat This!) |
    | Downbeat | osu2017 F1 | 92.90 | 85.90 (Beat This!) |
    | Beat | GTZAN F1 | 86.27 | 89.01 (Beat This!) |
    | Key | GiantSteps | 77.73 | 74.62 (madmom) |

    These are point estimates from the authors' own evaluation, with no significance tests. — [arXiv](https://arxiv.org/html/2610.05336v1)
  - SheetSage2-AR's training data: 441k pseudo-labelled recordings (28.4k h), mostly CC0 or synthetic, part of it licensed from Tokenwave.AI and partly private. — [arXiv](https://arxiv.org/html/2610.05336v1)
- SheetSage v1 (Donahue et al., 2022, older): **code MIT, models and dataset CC BY-NC-SA 3.0** because they are derived from HookTheory. — [LICENSE](https://github.com/chrisdonahue/sheetsage/blob/main/LICENSE)
- A related 2026 dataset, SheetSage-A2S (arXiv 2608.06165), evaluates audio-to-score lead sheets with symbol error rate. It notes that MV2H does not support the lead-sheet format. — [arXiv](https://arxiv.org/html/2608.06165v1)
- Licenses of component tools, from the GitHub API on 2026-11-18:

  | Tool | License | Last push |
  |---|---|---|
  | beat_this (CPJKU) | MIT | 2026-05-28 |
  | BTC-ISMIR19 (chords) | MIT | 2020, older |
  | SOME (openvpi, singing → MIDI) | MIT | 2026-03 |
  | MT3 | Apache-2.0 | 2026-10 |
  | basic-pitch | Apache-2.0 | – |
  | YourMT3 (mimbres) | GPL-3.0 | 2024-11 |

  — [beat_this](https://api.github.com/repos/CPJKU/beat_this), [BTC](https://api.github.com/repos/jayg996/BTC-ISMIR19), [SOME](https://api.github.com/repos/openvpi/SOME), [MT3](https://api.github.com/repos/magenta/mt3), [basic-pitch](https://api.github.com/repos/spotify/basic-pitch), [YourMT3](https://api.github.com/repos/mimbres/YourMT3)
- Band-in-a-Box 2026 can auto-interpret chords from MIDI and transcribe polyphonic audio to MIDI ("AI-notes"). It is a commercial end-to-end option of unknown accuracy. — [BiaB manual](https://www.pgmusic.com/manuals/bbw2026full/chapter3.htm)
- Songscription (a vendor blog) claims first place in its own Oct 2026 transcription benchmark. That benchmark is self-run, so the claim is weak evidence. — [Songscription blog](https://www.songscription.ai/blog/best-ai-sheet-music-generators)

### Inferences
- With SheetSage2, about 1 in 5 vocal notes is still wrong in pop and about 1 in 3 in rock. Chord errors are around 10% in pop and much worse for jazz or extended harmony. These errors propagate directly into any lead-sheet arranger, which is the main argument against a pure two-stage pipeline.
- For a commercial product: SheetSage2 / SheetSage1 weights are NC. A permissive chain would be Demucs (MIT **[unverified]**) -> SOME or basic-pitch for vocal melody, plus BTC or ChordFormer for chords and beat_this for beats. Accuracy would be clearly below SheetSage2.

### Gaps
- No head-to-head numbers for SOME / YourMT3 / ROSVOT vocal melody against SheetSage2 were found. SheetSage2 says external baselines (MuScriptor, YourMT3+, Demucs+ROSVOT) "score far below SheetSage1" on melody, but I did not extract their exact numbers.
- ChordFormer's code and license were not checked.
- SheetSage2's VRAM and speed are not stated.

## Q3. Has anyone evaluated a two-stage pipeline (lead sheet -> arranger) against end-to-end audio -> piano models?

### Takeaway
Only partially, and the evidence leans toward **richer intermediates beating a pure lead sheet**. No clean, controlled A/B of "lead sheet -> arranger" versus end-to-end on identical data was found.
- **PiCoGen v1 (lead sheet) vs Pop2Piano:** PiCoGen won subjectively.
- **PiCoGen2 (continuous SheetSage embeddings) vs Pop2Piano:** PiCoGen2 won clearly. Its authors abandoned the discrete lead sheet over suspected information loss, but never compared v1 against v2 empirically.
- **Etude (dense AMT features, not a lead sheet) vs PiCoGen2, AMT-APC and Music2MIDI:** Etude won, with OVL 3.50 against 2.97 for PiCoGen2 and 2.46 for AMT-APC.
- **The 2026 cross-modal arranger (Sheetsage lead sheet + audio style) vs PiCoGen2:** it was comparable or better subjectively, but worse on melody and chord accuracy in-domain.

### Cited Findings
- **PiCoGen v1** uses SheetSage to extract a lead sheet, then a Performer generates the piano. Compared with Pop2Piano across 10 genres, it had lower objective melody accuracy (MCA) but better subjective user ratings. The authors note performance is limited by lead-sheet extraction accuracy. — [arXiv 2407.20883](https://arxiv.org/html/2407.20883), [project](https://tanchihpin0517.github.io/PiCoGen/picogen1.html)
- **PiCoGen2** (arXiv 2408.01551, 2024) moved to an end-to-end design on frozen SheetSage *embeddings* "to avoid potential information loss from the two-stage process". It did **not** empirically compare against v1, only via demo examples. — [arXiv](https://arxiv.org/html/2408.01551v1), [README](https://github.com/tanchihpin0517/PiCoGen)
  - Listening test with 52 raters, MOS on a 1-5 scale:

    | System | OVL | SI | FL |
    |---|---|---|---|
    | PiCoGen2 | 3.48 | 3.55 | 3.66 |
    | Pop2Piano | 2.71 | 2.63 | 2.72 |
    | Human | 4.30 | 4.23 | 4.33 |

    Objective MCA was 0.17 for PiCoGen2, 0.42 for Pop2Piano and 0.16 for human covers, so the authors argue MCA is a poor metric. — [arXiv](https://arxiv.org/html/2408.01551v1)
- **Etude** (2025/26) ran a listening test with 101 raters on 100 test songs (each rater heard 4 songs). OVL ± SD:

  | System | OVL |
  |---|---|
  | Human | 3.92 ± 0.96 |
  | Etude-Default | 3.50 ± 0.99 |
  | Etude Extractor only (AMT-APC-like) | 3.33 |
  | PiCoGen2 | 2.97 |
  | AMT-APC | 2.46 |
  | Music2MIDI | 2.27 |

  — [arXiv](https://arxiv.org/html/2509.16522v2)
  - In an ablation, retraining PiCoGen2 on Etude's data raised its OVL only to 3.18, against 3.55 for Etude (p = 0.0007), so the gain comes from the architecture. — [arXiv](https://arxiv.org/html/2509.16522v2)
  - Etude explicitly calls PiCoGen's lead-sheet intermediate an "information bottleneck" and uses dense AMT features instead. Yet its decoder scored *lower Similarity* (3.16) than the extractor alone (3.41). The authors attribute this to the decoder failing to identify the main melody and propose multi-stream (e.g. separate melody) extractors as future work. — [arXiv](https://arxiv.org/html/2509.16522v2)
- **Cross-modal arranger** (arXiv 2608.03050, Aug 2026): this is a two-stage pipeline (Sheetsage lead sheet, then an arranger with audio style conditioning) compared directly against end-to-end PiCoGen2. — [arXiv](https://arxiv.org/html/2608.03050)
  - On POP909 it had lower melody and chord accuracy (MCA 32.0 vs 39.3), which the authors blame on Sheetsage transcription errors. — [arXiv](https://arxiv.org/html/2608.03050)
  - It had better style and time-signature metrics, and was slightly better than PiCoGen2 on out-of-distribution Ballroom/GTZAN. — [arXiv](https://arxiv.org/html/2608.03050)
  - It was subjectively significantly better on coherence and musicality (21 raters). — [arXiv](https://arxiv.org/html/2608.03050)

### Inferences
- The evidence suggests a hybrid design for this project:
  - Keep end-to-end or dense audio features (AMT-APC / Etude-style) for texture and similarity.
  - Inject an explicit, accurate melody line and chords (SheetSage2) as extra conditioning streams, which addresses Etude's "decoder can't find the melody" failure.
  - Use beat_this or SheetSage2 downbeats for bar structure.

  This is close to Etude's own proposed future work.
- A pure lead sheet -> POP909-trained arranger will probably sound "generic accompaniment + melody" and lose song-specific riffs, intros and textures. That is useful as a fallback or for a simplified/easy difficulty level, where a lead-sheet arranger plus playability rules may actually be preferable.
- Benchmarks are inconsistent: different test sets, metrics such as MCA treated as unreliable, and small listening panels (13 to 101 raters). Cross-paper OVL numbers are not directly comparable.

### Gaps
- I found no paper that runs SheetSage2 lead sheets through a modern arranger (D3PIA, AccoMontage, Etude decoder) and compares the result against Etude or AMT-APC in a listening test. That would be a novel and cheap experiment for this project.
- I found no evaluation of playability or difficulty of any of these outputs against human covers.
- I could not confirm whether any 2026 system (e.g. a PiCoGen3 or Etude v2) combines SheetSage2 features with dense AMT features.
