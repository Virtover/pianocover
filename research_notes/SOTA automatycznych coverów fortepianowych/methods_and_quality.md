# SOTA in automatic piano cover generation (audio -> piano), 2022 to Nov 2026: methods and quality

Research date: 2026-11-18. "Verified" means read in the paper's arXiv HTML or abstract page during this session. "Secondhand" means it came from a search-engine snippet or summary and was not read in the primary text.

## 1. Methods: architecture, representation, data, published quality evidence

### Takeaway
There is one main lineage: Pop2Piano (2022/23) → PiCoGen (2024) → PiCoGen2 (ISMIR 2024) / AMT-APC (2024) / Music2MIDI (MMM 2025) → Etude (arXiv Sep 2025, ICASSP 2026) → BOSSA-style cross-modal style conditioning (ISMIR 2026). Etude is the newest head-to-head audio-to-piano-cover paper with the largest listening test (101 listeners). It beats PiCoGen2, AMT-APC and Music2MIDI significantly on overall rating, but still sits well below human covers (OVL 3.50 vs 3.92).

### Cited Findings

**Pop2Piano: "Pop2Piano: Pop Audio-based Piano Cover Generation", Choi & Lee, arXiv 2211.00895 (v1 2 Nov 2022, v2 1 Apr 2023), ICASSP 2023 [older]**
- Architecture: T5-small encoder-decoder taken from MT3, about 59M parameters, with relative positional embedding and a learnable arranger-style embedding prepended to the spectrogram (verified). — [arXiv HTML](https://arxiv.org/html/2211.00895v2)
- Input: log-mel spectrogram (22.05 kHz, win 4096, hop 1024). There is no melody or chord extractor. Output tokens: pitch, note on/off, Beat Shift quantized to 8th-note beats, EOS/PAD (verified). — [arXiv HTML](https://arxiv.org/html/2211.00895v2)
- Data (PSP): 5,989 YouTube piano covers from 21 arrangers, leaving 4,989 tracks (~307 h) after filtering. Covers were transcribed with a piano AMT model, aligned with SyncToolbox, and their notes were remapped and quantized onto the pop song's beat grid (Essentia). Pairs were dropped if melody chroma accuracy ≤ 0.15 or the length differed by ≥ 20% (verified). — [arXiv HTML](https://arxiv.org/html/2211.00895v2)
- Results: MCA 0.402 on POP909F, vs human covers 0.395 (POP909F) / 0.349 (PSP). Generated note density tracks arranger density with R²=0.804. Listening test: 25 non-professional listeners, 10-s chorus clips, naturalness MOS of 3.771 for human, 3.216 for Pop2Piano-PSP and 2.856 for Pop2Piano-POP909. The text says 70% preference vs the POP909 model but Table 2 shows 64%, an internal inconsistency (verified). — [arXiv HTML](https://arxiv.org/html/2211.00895v2)
- Authors' own limitations: only a 4-beat input context, so melody and accompaniment consistency degrades over longer spans. 8th-note quantization rules out triplets, 16ths and trills (verified). — [arXiv HTML](https://arxiv.org/html/2211.00895v2)
- Venue ICASSP 2023 is secondhand, from Etude's reference list as quoted in a search snippet. — [Etude PDF](https://arxiv.org/pdf/2509.16522)
- Demo: [sweetcocoa.github.io/pop2piano_samples](https://sweetcocoa.github.io/pop2piano_samples); code: [github.com/sweetcocoa/pop2piano](https://github.com/sweetcocoa/pop2piano) (cited in PiCoGen). — [PiCoGen HTML](https://arxiv.org/html/2407.20883v1)

**PiCoGen: "PiCoGen: Generate Piano Covers with a Two-stage Approach", Tan, Guan, Yang (NTU / KKCompany), arXiv 2407.20883 (30 Jul 2024), ICMR 2024 [older]**
- Two stages. Stage 1 runs SheetSage (pretrained, frozen) to extract a lead sheet (melody and chords) from audio. Stage 2 is a CP-Transformer "Performer" (8 layers, 8 heads, 512-d) that generates piano bar by bar from the lead sheet. The Performer is trained on Pop1k7 (~1,700 piano covers) with lead sheets derived symbolically, so no paired audio is needed (verified). — [arXiv HTML](https://arxiv.org/html/2407.20883v1)
- Evaluation on GTZAN (883 songs). MCA: Pop2Piano 0.25 vs PiCoGen 0.17. Listening test with 30 volunteers on a 1–5 scale, Pop2Piano vs PiCoGen: Similarity 2.65 vs 3.20, Melody sim 2.85 vs 3.35, Chord sim 2.65 vs 3.05, Fluency 2.60 vs 3.20, Overall 2.55 vs 3.35. The difference was **not statistically significant** because of large standard deviations (verified). — [arXiv HTML](https://arxiv.org/html/2407.20883v1)
- Failure modes: poor on hip-hop because SheetSage gives bad lead sheets there. SheetSage assumes constant tempo, so 117 songs had to be excluded (verified). — [arXiv HTML](https://arxiv.org/html/2407.20883v1)

**PiCoGen2: "PiCoGen2: Piano cover generation with transfer learning approach and weakly aligned data", Tan, Ai, Chang, Guan, Yang, arXiv 2408.01551 (2 Aug 2024), ISMIR 2024 [older]**
- Architecture: frozen SheetSage audio encoder, then a 4-layer Transformer adapter, then a GPT-NeoX decoder (8 layers, 8 heads). About 39M trainable parameters. Output is modified REMI with 428 tokens (bar, position, chord, tempo, pitch, duration, velocity) (verified). — [arXiv HTML](https://arxiv.org/html/2408.01551v1)
- Training has two steps. It is first pre-trained on piano-only performances (100K steps), then fine-tuned on **weakly aligned** song–piano pairs: 5,844 web pairs collected with the Pop2Piano pipeline, leaving 5,503 after filtering (MCA < 0.05 or length difference > 15%). About 50% of bars were discarded as invalid. In the weak alignment, Beat Transformer finds beats in both audios, beats are mapped through the warping path, and piano note timing is **not** remapped. Each piano bar is paired with its matching song segment (verified). — [arXiv HTML](https://arxiv.org/html/2408.01551v1)
- Results, Table 2 (MCA / OVL / SI / FL):
  - PiCoGen2: 0.17 / 3.48 / 3.55 / 3.66
  - Pop2Piano: 0.42 / 2.71 / 2.63 / 2.72
  - Transcription baseline: 0.19 / 1.48 / 1.69 / 1.45
  - Human: 0.16 / 4.30 / 4.23 / 4.33
  - The test had 52 listeners, 3 songs each, 40-s clips. PiCoGen2 led every subjective criterion with p<0.05 (verified). — [arXiv HTML](https://arxiv.org/html/2408.01551v1)
- Demo and code: [tanchihpin0517.github.io/PiCoGen](https://tanchihpin0517.github.io/PiCoGen/). ISMIR 2024 poster: [ismir2024program poster_114](https://ismir2024program.ismir.net/poster_114.html).

**AMT-APC: "AMT-APC: Automatic Piano Cover by Fine-Tuning an Automatic Music Transcription Model", Komiya & Fukuhara, arXiv 2409.14086 (v1 21 Sep 2024; v2 4 Jul 2026, no changelog visible) [older]**
- Architecture: hFT-Transformer (a piano AMT model) pre-trained on transcription, then fine-tuned to output covers. Input is a mel-spectrogram. Output is piano-roll-like onset, frame and velocity matrices (512 frames × 88 pitches; velocity as a 128-way distribution). The context was extended from 128 to 512 frames (verified). — [arXiv HTML v2](https://arxiv.org/html/2409.14086v2)
- Style vector: 24 dimensions, made of quantized distributions of onset rate, velocity and pitch, injected through a gated FC layer into the encoder states. Pop2Piano instead uses discrete arranger IDs (verified). — [arXiv HTML v2](https://arxiv.org/html/2409.14086v2)
- Data: **small**. 332 songs and 1,267 YouTube covers, with songs chosen by Uta-Net artist ranking (J-pop). Unsuitable covers (different key, poor audio) were removed automatically and manually. Pairs were aligned with synctoolbox and cover MIDI was produced by AMT. Training took 5 epochs, about 3 h on 2×A100 (verified). — [arXiv HTML v2](https://arxiv.org/html/2409.14086v2)
- Results: Qmax cover-song similarity via ChromaCoverID (lower is better): AMT-APC 0.035, human re-synth 0.038, PiCoGen2 0.054, Pop2Piano 0.090. Ablation F1 (avg of onset, frame, velocity): full model 0.31; without AMT pre-training 0.27; without style vector 0.30; raw AMT without fine-tuning 0.12. **No listening test** (verified). — [arXiv HTML v2](https://arxiv.org/html/2409.14086v2)
- Authors' limitations: hard to keep consistency across the whole piece. The style vector captures only coarse features (density, velocity) and not accompaniment patterns or ornamentation (verified). — [arXiv HTML v2](https://arxiv.org/html/2409.14086v2)
- Demo and code: [310hz.github.io/amt-apc](https://310hz.github.io/amt-apc/).

**Music2MIDI: "Music2MIDI: Pop Music to MIDI Piano Cover Generation", Yip & Chau (CUHK), MMM 2025 (Springer LNCS, DOI 10.1007/978-981-96-2064-7_8)**
- Transformer that maps raw audio directly to piano MIDI. Its synchronized dataset has about 3,000 pairs (~180 h), filtered with two synchronization-quality metrics. The authors claim it beats Pop2Piano and PiCoGen on MCA and in subjective evaluation. All of this is **secondhand** (search snippet of the repo/abstract; the paper was not read). — [GitHub ytinyui/music2midi](https://github.com/ytinyui/music2midi); [Springer](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_8)

**Etude: "Etude: Piano Cover Generation with a Three-Stage Approach — Extract, strucTUralize, and DEcode", Tse-Yang Chen & Yuh-Jzer Joung (NTU), arXiv 2509.16522 (v1 Sep 2025, v2 23 Jan 2026)**
- Venue: listed as ICASSP 2026 accepted paper #4017 (**secondhand**, from a search snippet of the ICASSP 2026 accepted list). The project page still says "Preprint". — [ICASSP 2026 accepted papers](https://cmsworkshops.com/ICASSP2026/papers/accepted_papers.php); [project page](https://xiugapurin.github.io/Etude/)
- Stage 1, Extract: an Extractor adapted from **AMT-APC** (hFT-Transformer), retuned to output dense, transcription-like features rather than a playable cover (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Stage 2, strucTUralize: a pretrained Beat-Transformer on the source audio gives beats, downbeats, tempo and bar boundaries (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Stage 3, DEcode: a GPT-NeoX decoder (8 layers, 8 heads, 512-d, ~25.5M parameters) generates bar by bar. It interleaves source-feature bars and target bars, conditioning on up to 4 previous bar pairs (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- "Tiny-REMI" tokens: Bar, Pos (16th-note), Note (88), Duration (10 values), Grace. Tempo, chord and velocity tokens are dropped and timing is restored from the beat grid (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Per-bar style control: polyphony, rhythmic intensity and sustain, each binned into 3 levels (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Data: about 7,700 song/cover audio pairs (mostly J-pop and K-pop), 4,752 pairs (~500 h) after filtering. Pairs were dropped if the length differed by more than 30 s or the warping-path std exceeded 1.0. Alignment is a modified PiCoGen2 weak alignment: beats are detected on the source only, mapped to the cover with MrMsDTW, and aligned at measure level. The test set is 100 unseen songs, split evenly across C-pop, J-pop, K-pop and Western pop (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Objective results (WPD↓ structure distance, RGC↓ grid coherence, IPE rhythm entropy):
  - Human: 0.49 / 0.042 / 10.13
  - Etude Default: 0.21 / 0.020 / 9.02
  - Etude Extractor: 0.12 / 0.028 / 10.62
  - PiCoGen2: 1.00 / 0.059 / 7.97
  - AMT-APC: 0.09 / 0.114 / 10.69
  - Music2MIDI: 0.18 / 0.160 / 8.94
  - (verified) — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Listening test: **101 listeners** (30 amateur, 39 intermediate, 32 expert). Each heard 4 songs, with 7 anonymized versions per song, all rendered on the same synth, using the first 60 s. Ratings were 5-point SI / FL / DE / OVL:
  - Human: 3.75 / 4.03 / 3.79 / 3.92
  - Etude Default: 3.16 / 3.73 / 3.46 / **3.50**
  - Etude Prompted: 3.17 / 3.70 / 3.49 / 3.46
  - Etude Extractor: **3.41** / 3.31 / 3.35 / 3.33
  - PiCoGen2: 2.88 / 3.33 / 2.73 / 2.97
  - AMT-APC: 2.64 / 2.37 / 2.71 / 2.46
  - Music2MIDI: 2.56 / 2.29 / 2.24 / 2.27
  - All Etude variants beat PiCoGen2, AMT-APC and Music2MIDI on OVL (corrected p<0.001) (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Ablation (30 new listeners): PiCoGen2 retrained on Etude's data scored OVL 3.18 vs the original's 2.98 (not significant, p=0.066). Etude's 3.55 vs retrained PiCoGen2's 3.18 was significant (p=0.0007). The authors conclude the gain comes from the architecture, not just the data (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Pop2Piano was **not** included in Etude's evaluation tables (verified). — [arXiv HTML](https://arxiv.org/html/2509.16522)
- Code: [github.com/Xiugapurin/Etude](https://github.com/Xiugapurin/Etude). The project page has a 100-song model-comparison demo (Etude vs Music2MIDI, PiCoGen2, AMT-APC) and a style-control demo. Weights are not mentioned on the page (verified). — [project page](https://xiugapurin.github.io/Etude/)

**Cross-modal style bootstrapping (BOSSA): "Learning Music Style for Piano Arrangement Through Cross-Modal Bootstrapping", arXiv 2608.03050 (4 Aug 2026), ISMIR 2026 (acceptance is secondhand, from a search snippet)**
- Pipeline: audio goes to SheetSage to get a lead sheet. Style comes from a frozen MusicGen-Large audio LM (layer-25 hidden states), passed through a Q-Former (initialized from MusicBERT, 32 queries) into a frozen **MuseCoco-xLarge** symbolic decoder (1.2B parameters) with a LoRA (rank 16) for lead-sheet conditioning. A 10-s audio clip sets the style. Output is piano MIDI, generated in 4-bar windows. This is the closest thing found to "MuseCoco applied to piano arrangement" (verified). — [arXiv HTML](https://arxiv.org/html/2608.03050v1)
- Data: POP909 plus PIAST (~8K piano recordings with transcriptions), paired with loosely aligned audio (±1 s shift and 12-key transposition). There are no web cover pairs (verified). — [arXiv HTML](https://arxiv.org/html/2608.03050v1)
- Results on POP909, Ours vs PiCoGen2: MCA 32.0 vs **39.3**, CA 33.3 vs **40.0**, TA (style) **83.6** vs 74.4. On Ballroom/GTZAN Ours is slightly ahead (MCA 17.8 vs 17.2) with a much higher TA (79.4 vs 57.6). The listening test had only **21 participants** and 6 pieces. The method was comparable to PiCoGen2 on Naturalness and significantly better on Coherence and Musicality. Per-criterion numbers appear only in a figure (verified). Etude, AMT-APC and Pop2Piano were **not** compared. — [arXiv HTML](https://arxiv.org/html/2608.03050v1)
- Demo: [zhaojw1998.github.io/bossa](https://zhaojw1998.github.io/bossa/).

**Adjacent symbolic work (not audio-to-cover, but relevant for a reduction/arranger stage)**
- "Towards Practical Automatic Piano Reduction using BERT with Semi-supervised Learning", arXiv 2512.21324 (Dec 2025, rev. Jan 2026). MidiBERT does "simplification then harmonization" and predicts keep or drop per note of an orchestral score. It was pretrained on GiantMIDI-Piano and SOD and fine-tuned on the Live Orchestral Piano dataset (secondhand snippet, venue unclear). — [arXiv](https://arxiv.org/abs/2512.21324)
- D3PIA, a discrete denoising diffusion model for piano accompaniment from a lead sheet, from KAIST MAC lab at ICASSP 2026 (secondhand snippet). It would fit as a stage 2 after lead-sheet extraction. — [KAIST MAC pubs](https://mac.kaist.ac.kr/pubs.html)
- Anticipatory Music Transformer (Thickstun et al., TMLR 2024) does symbolic accompaniment infilling on Lakh MIDI. No audio-conditioned piano-cover application was found. — [arXiv 2306.08620](https://arxiv.org/abs/2306.08620)
- "Unlocking Potential in Pre-Trained Music Language Models for Versatile Multi-Track Music Arrangement", arXiv 2408.15176 (2024), includes piano reduction as one of its symbolic tasks (secondhand snippet). — [arXiv](https://arxiv.org/html/2408.15176v1)
- MPEcho, arXiv 2607.26698 (2026), does controllable cover-song generation (vocal/audio). It is not piano-specific (snippet only). — [arXiv](https://arxiv.org/pdf/2607.26698)

### Inferences
- The head-to-head chain, all from the beating paper's own tests:
  - PiCoGen beat Pop2Piano, but not significantly.
  - PiCoGen2 beat Pop2Piano (subjective, p<0.05).
  - AMT-APC beat PiCoGen2 and Pop2Piano on Qmax similarity only, with no listening test.
  - Music2MIDI claims to beat Pop2Piano and PiCoGen (secondhand).
  - Etude beat PiCoGen2, AMT-APC and Music2MIDI (subjective, p<0.001, n=101).
  - BOSSA beat PiCoGen2 on style and musicality but lost on MCA/CA on POP909.
- A defensible ranking for overall musicality is Etude > PiCoGen2 ≈ BOSSA (not compared directly to Etude) > AMT-APC ≈ Music2MIDI > Pop2Piano.
- For literal faithfulness, the ranking is AMT-APC / Etude-Extractor > Etude > PiCoGen2.

### Gaps
- Music2MIDI's architecture details and numbers were not read from the primary paper.
- No Etude-vs-BOSSA or Etude-vs-Pop2Piano comparison exists.
- No ISMIR 2025 piano-cover paper was found. The ISMIR 2026 and NeurIPS 2025/26 lists were not checked exhaustively.
- No diffusion or LLM model that works directly from audio to piano cover was found.

## 2. Which is best for faithfulness AND musicality/playability; disagreements

### Takeaway
By the most recent and largest study, Etude's Decoder gives the best balance (OVL 3.50, top fluency and dynamics). AMT-APC-style transcription-like models win on literal similarity metrics but lose badly on fluency. The papers disagree mainly because their metrics differ: MCA, Qmax and WPD all reward literal note overlap, while listeners reward structure and rhythm.

### Cited Findings
- AMT-APC's Qmax (0.035) is better than the re-synthesized human reference (0.038). Its abstract says it "reproduces original tracks more accurately than any existing models", but it compared only two baselines and ran no listening test. — [AMT-APC HTML](https://arxiv.org/html/2409.14086v2)
- In Etude, AMT-APC has the lowest WPD (0.09, the most structurally literal), yet the second-lowest overall rating (OVL 2.46, FL 2.37). The authors call its output overly literal and "chaotic" (high IPE) with no structural modelling. — [Etude HTML](https://arxiv.org/html/2509.16522)
- Etude's Extractor alone, essentially a retuned AMT-APC, had the best AI similarity score (SI 3.41) but lower fluency (3.31). The Decoder gained fluency (3.73) at some cost to similarity (3.16), which the authors attribute to difficulty identifying the main melody. — [Etude HTML](https://arxiv.org/html/2509.16522)
- Pop2Piano had the **highest MCA (0.42, above human 0.16)** in PiCoGen2's evaluation but the lowest similarity ratings (2.63). PiCoGen2's authors say MCA assumes a fixed time grid. In PiCoGen, Pop2Piano also won MCA (0.25 vs 0.17) yet lost on listener-rated similarity. — [PiCoGen2 HTML](https://arxiv.org/html/2408.01551v1); [PiCoGen HTML](https://arxiv.org/html/2407.20883v1)
- BOSSA had lower MCA/CA than PiCoGen2 on POP909 but higher musicality in a small test (n=21), and blamed SheetSage lead-sheet errors. — [BOSSA HTML](https://arxiv.org/html/2608.03050v1)
- Etude lists the divergence between objective metrics and human perception as an open limitation. — [Etude HTML](https://arxiv.org/html/2509.16522)

### Inferences
- Objective "similarity" metrics (MCA, Qmax, WPD) systematically favor transcription-like, over-dense outputs and should not be used alone to rank systems. Listening tests consistently rank structured, beat-grid generators (PiCoGen2, Etude) above them.
- No paper measures playability (hand span, polyphony per hand) directly. Etude's polyphony and sustain controls and its RGC metric are the closest proxies.

### Gaps
- No paper evaluates playability or sheet-music readability.
- None evaluates on Polish/European repertoire or on guitar-heavy songs.
- Listening tests use different synths, clip lengths and listener pools, so scores are not comparable across papers.

## 3. Known failure modes and the fixes authors propose

### Takeaway
The documented failures fall into a few groups. The two fixes that showed up repeatedly are a beat/bar-based token grid and a split between a dense extractor and a structured decoder.
- **Pop2Piano:** short context and coarse quantization, plus alignment artifacts from note remapping.
- **AMT-APC:** over-literal, dense, off-grid output.
- **PiCoGen and PiCoGen2:** a lead-sheet bottleneck, monotonous rhythm, and dependence on SheetSage and the beat tracker.
- **Music2MIDI:** off-beat notes.

### Cited Findings
- Pop2Piano:
  - A 4-beat context weakens long-range consistency, and 8th-note quantization rules out triplets, 16ths and trills (authors). — [Pop2Piano HTML](https://arxiv.org/html/2211.00895v2)
  - Note remapping distorts rhythm and loses piano information (PiCoGen2). — [PiCoGen2 HTML](https://arxiv.org/html/2408.01551v1)
  - Raw MIDI-like events are hard to learn and strong alignment induces artifacts (Etude). — [Etude HTML](https://arxiv.org/html/2509.16522)
- AMT-APC:
  - Hard to keep whole-piece consistency, and the style vector is coarse (authors). — [AMT-APC HTML](https://arxiv.org/html/2409.14086v2)
  - A time-based rather than beat-based framework gives high RGC (0.114, off-grid) and a chaotic rhythm entropy (Etude). — [Etude HTML](https://arxiv.org/html/2509.16522)
  - Fix (Etude): keep the AMT-APC-like model only as a dense feature extractor, then re-generate on a beat grid with a bar-wise decoder. — [Etude HTML](https://arxiv.org/html/2509.16522)
- PiCoGen and PiCoGen2:
  - The lead-sheet features are an information bottleneck, low IPE means monotonous rhythm, and quality is limited by beat-tracking accuracy (Etude). — [Etude HTML](https://arxiv.org/html/2509.16522)
  - SheetSage fails on hip-hop and assumes constant tempo (PiCoGen). — [PiCoGen HTML](https://arxiv.org/html/2407.20883v1)
  - A model trained only on piano data produces repeated short notes, and without piano pre-training it loses melodic contour (PiCoGen2 ablations). — [PiCoGen2 HTML](https://arxiv.org/html/2408.01551v1)
- Music2MIDI: highest RGC (0.160, off-beat) and lowest fluency and dynamics (Etude). — [Etude HTML](https://arxiv.org/html/2509.16522)
- Weak style control:
  - Pop2Piano uses arranger IDs (density R²=0.80), and AMT-APC's continuous vector is coarse.
  - Etude adds per-bar relative attributes, but its "Prompted" mode applied one vector to the whole song and gained nothing in listening scores (OVL 3.46 vs 3.50).
  - BOSSA conditions on reference audio through an audio LM and reports much higher style accuracy (TA 83.6 vs 74.4).
  - Sources: [Pop2Piano](https://arxiv.org/html/2211.00895v2); [AMT-APC](https://arxiv.org/html/2409.14086v2); [Etude](https://arxiv.org/html/2509.16522); [BOSSA](https://arxiv.org/html/2608.03050v1)
- Etude's own limitations: capped by beat-tracker precision, a single flattened feature stream loses information, and the melody is sometimes lost in the decoder. Proposed future work is multi-stream extraction and learned perceptual metrics. — [Etude HTML](https://arxiv.org/html/2509.16522)

### Inferences
- The "melody gets dropped" problem shows up both in Pop2Piano-family listening scores and in Etude's Decoder (SI drops from 3.41 to 3.16). An explicit melody stream, such as a separated-vocal f0 or lead-sheet melody fed alongside the dense features, is the obvious remedy. No paper has tested it yet; Etude lists "multi-stream extraction" as future work.

### Gaps
- No quantitative study isolates "melody dropped" rates per system.

## 4. Does fine-tuning AMT-APC/Pop2Piano on noisy web cover pairs hurt? What did others do instead?

### Takeaway
No paper reports exactly "we fine-tuned AMT-APC/Pop2Piano on more web pairs and it got worse." Several papers do report evidence of the underlying problem: web cover pairs are only loosely aligned, strong note-level remapping corrupts them, and data alone doesn't fix quality. The approaches that worked instead are weak bar-level alignment, aggressive filtering, piano-only pre-training, and moving the aligned dense model into a feature-extractor role.

### Cited Findings
- PiCoGen2's argument against strong alignment:
  - Covers are reinterpretations, not transcriptions, so forcing a note-by-note match to the song is inappropriate.
  - Measured duration, tempo and IOI deviation ratios (1.10 / 1.16 / 1.14) show covers and songs aren't tightly aligned.
  - About 50% of bars were invalid after alignment.
  - Their fix is weak bar-level alignment with no note remapping, plus pre-training on piano-only data.
  - Source: [PiCoGen2 HTML](https://arxiv.org/html/2408.01551v1)
- PiCoGen avoids paired data entirely: lead-sheet extraction, then a generator trained on piano-only Pop1k7. — [PiCoGen HTML](https://arxiv.org/html/2407.20883v1)
- AMT-APC itself was trained on only 1,267 manually curated covers, removing ones in a different key or with poor audio. Training ran for only 5 epochs, and AMT pre-training mattered (F1 0.31 vs 0.27 without it). — [AMT-APC HTML](https://arxiv.org/html/2409.14086v2)
- Etude's data work:
  - Filtered about 7,700 to 4,752 pairs by length difference and warping-path std (> 1.0 rejected).
  - Aligned at measure level using beats from the source only.
  - Retraining PiCoGen2 on this larger, cleaner data gave a non-significant gain (2.98 to 3.18, p=0.066), while the architectural change gave a significant one.
  - Source: [Etude HTML](https://arxiv.org/html/2509.16522)
- Etude deliberately retuned the AMT-APC-derived Extractor toward dense transcription-like output and left musicality to a separate bar-level decoder. — [Etude HTML](https://arxiv.org/html/2509.16522)
- Pop2Piano filtered by MCA > 0.15 and a length difference under 20%. PiCoGen2 used MCA ≥ 0.05 and a length difference ≤ 15%. — [Pop2Piano HTML](https://arxiv.org/html/2211.00895v2); [PiCoGen2 HTML](https://arxiv.org/html/2408.01551v1)
- BOSSA avoided web cover pairs, using POP909 and PIAST with loose ±1 s alignment as augmentation. — [BOSSA HTML](https://arxiv.org/html/2608.03050v1)

### Inferences
- Our result (AMT-APC fine-tuned on about 2.5k Transkun-aligned web pairs sounded worse) is consistent with this literature. AMT-APC's loss is frame-level and time-based, so DTW misalignment and arranger variance across many covers push it toward smeared, averaged, over-dense output. PiCoGen2 and Etude both moved away from frame-level supervision on web pairs to bar-level token targets on a beat grid.
- The practical path the literature points to is Etude's: use AMT-APC (or our fine-tune, possibly tuned denser) as Stage 1, Beat-Transformer as Stage 2, and train a small bar-wise decoder (~25M parameters) on weakly aligned pairs with WP-std filtering.

### Gaps
- No primary source directly ablates "AMT-APC + more noisy web pairs".
- No source tests Transkun as the cover transcriber, compared with the AMT models used by Pop2Piano, AMT-APC and Etude.

## 5. Demo pages and whether results sound musical

### Takeaway
Every major method has a public demo, and Etude's page has a 100-song side-by-side comparison of the four modern systems. The only musicality judgments available come from the authors' own listening tests. No independent third-party listening evaluation was found.

### Cited Findings
- Pop2Piano: [sweetcocoa.github.io/pop2piano_samples](https://sweetcocoa.github.io/pop2piano_samples). Authors' naturalness MOS 3.22 vs 3.77 for human (n=25, 10-s clips). — [Pop2Piano HTML](https://arxiv.org/html/2211.00895v2)
- PiCoGen and PiCoGen2: [tanchihpin0517.github.io/PiCoGen](https://tanchihpin0517.github.io/PiCoGen/), which hosts code, samples and song URLs. PiCoGen2 OVL 3.48 vs 4.30 for human (n=52). — [PiCoGen2 HTML](https://arxiv.org/html/2408.01551v1)
- AMT-APC: [310hz.github.io/amt-apc](https://310hz.github.io/amt-apc/), with code and demo audio. No author listening test; it was rated least fluent but one by Etude's listeners (FL 2.37). — [AMT-APC HTML](https://arxiv.org/html/2409.14086v2); [Etude HTML](https://arxiv.org/html/2509.16522)
- Music2MIDI: [github.com/ytinyui/music2midi](https://github.com/ytinyui/music2midi).
- Etude: [xiugapurin.github.io/Etude](https://xiugapurin.github.io/Etude/), with a model comparison across 100 songs and 4 genres plus a style-control demo. Code is at [github.com/Xiugapurin/Etude](https://github.com/Xiugapurin/Etude). Etude Default OVL 3.50 vs 3.92 for human (n=101). — [Etude HTML](https://arxiv.org/html/2509.16522)
- BOSSA: [zhaojw1998.github.io/bossa](https://zhaojw1998.github.io/bossa/). Judged musical in a small test (n=21). — [BOSSA HTML](https://arxiv.org/html/2608.03050v1)

### Inferences
- In every study, the best AI system still sits about 0.4–0.8 MOS points below human covers. Even the SOTA output is "decent but clearly machine-made."
- Etude's comparison page is the most efficient way to audit SOTA by ear before we invest.

### Gaps
- No third-party listening evaluations, blog reviews or community comparisons were found.
- The demo pages were not listened to.
- Whether Etude publishes trained weights is unconfirmed; the page mentions code only.
