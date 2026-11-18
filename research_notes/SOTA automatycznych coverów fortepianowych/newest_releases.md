# Newest releases (Sep–Nov 2026) relevant to automatic piano cover / arrangement from audio

Research date: 2026-11-18. Method: about 17 web searches and fetches (extended mode), plus direct fetches of the arXiv abstract pages, GitHub READMEs and HF model cards. Everything below was checked against the primary page unless it is marked "snippet only".

## What was released or announced in Sep–Nov 2026 (dates, IDs, repos, weights, license)?

### Takeaway
I found **no new piano-cover model in the last two weeks (about Nov 4–18, 2026)**. The only item in that window is a v2 revision of MuLaCover on Nov 4. The most relevant new and runnable release in the window is **tsumugi-piano-cover V2 (2026-10-20, sheet-music output 2026-10-21)**. It is an open-code (MIT), open-weights (CC BY-NC 4.0) pipeline from audio to piano MIDI to MusicXML. The most relevant new paper is **BOSSA (arXiv 2608.03050, ISMIR 2026)**, with code and checkpoints released. The other Sep–Nov items are cover-song systems that output **audio** (MuLaCover) or work in the reverse direction (Piano2Pop).

### Cited Findings

**A. Released with code and weights**

- **tsumugi-piano-cover V2** (anime-song/tsumugi-piano-cover; a fork was viewed at haveyouwantto/tsumugi-piano-cover). The README says it "Generate[s] piano covers of pop songs as MIDI". It lists a V2 release dated **2026-10-20** and a sheet-music feature (two-staff MusicXML, rendered with Verovio) dated **2026-10-21**. The GitHub topics page showed "Updated Oct 21, 2026" — [GitHub README](https://github.com/haveyouwantto/tsumugi-piano-cover)
  - Pipeline: tsumugi transcribes the audio to multi-instrument MIDI (stem separation, instrument refinement, velocity, beat/chord/key detection). An autoregressive piano model then generates the arrangement conditioned on that MIDI. Songs are handled in 2 s patches with a Global/Local transformer, 10 ms event resolution, and gated cross-attention to the source. An "Arrangement Planner" predicts velocity, density and arrangement curves. V2 uses SheetSage2-Refined for beats and melody — [GitHub](https://github.com/haveyouwantto/tsumugi-piano-cover)
  - Training data: the piano prior was pretrained on about 3,000 h of piano (transcribed acoustic performances, PiJAMA, Pop2Piano piano tracks, PIAST, MAESTRO). It was fine-tuned on piano covers paired with tsumugi transcriptions. The README gives two inconsistent figures: "~3,000 piano covers" and "~190 hours across ~780 songs" — [GitHub](https://github.com/haveyouwantto/tsumugi-piano-cover)
  - License: code is MIT. Weights (V1 and V2) are **CC BY-NC 4.0, non-commercial**, and SheetSage2-Refined is also non-commercial — [GitHub](https://github.com/haveyouwantto/tsumugi-piano-cover); [HF model card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
  - Size: piano_cover is 127.4M parameters, piano_ar 69.2M, piano_perf2score 110.6M, all fp32 safetensors. The model card showed 2,399 downloads in the last month — [HF](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
  - Tooling: a "Cover Studio" web UI (FastAPI + React) for iterative generation, comparing takes and regenerating from any point. There is also a Python API (`CoverModel.from_pretrained`, `generate_covers`, `events_to_midi`) and a CLI. Controls include source_cfg, density, dynamics, temperature and top_p. On Windows, torchcodec needs shared FFmpeg libraries. You can also upload your own MIDI in place of running tsumugi transcription — [GitHub](https://github.com/haveyouwantto/tsumugi-piano-cover)
- **BOSSA: "Learning Music Style for Piano Arrangement Through Cross-Modal Bootstrapping"** by Jingwei Zhao, Gus Xia, Ziyu Wang and Ye Wang (NUS / MBZUAI-related; Zhao now lists a songscription.ai email). arXiv **2608.03050**, v1 submitted **2026-08-04**, accepted to **ISMIR 2026** — [arXiv](https://arxiv.org/abs/2608.03050); [demo page](https://zhaojw1998.github.io/bossa/); [author CV](https://zhaojw1998.github.io/archive/pdf/CV_Jingwei.pdf)
  - Method: a Q-Former (in the style of BLIP-2) extracts a style vector from a pretrained audio LM and conditions a symbolic LM (MuseCoco-1B) that writes the piano arrangement. Training has two stages: contrastive alignment, then generation. Inputs are a lead sheet (melody + chords; for covers this is transcribed from audio by SheetSage) and a reference audio clip for style. It generates in 4-bar windows with a 2-bar hop — [arXiv](https://arxiv.org/abs/2608.03050); [repo](https://github.com/zhaojw1998/bossa-piano-cover-arrangement)
  - Code and checkpoints are released (`checkpoints.tar.gz` on Google Drive). **No license is stated** in the repo — [GitHub](https://github.com/zhaojw1998/bossa-piano-cover-arrangement)
- **MuLaCover** (HeartMuLa): the inference package was released **2026-09-11**, and code plus weights went public **2026-09-16**. The paper "MuLaCover: Flexible Cover Song Generation via Symbolic Tonal Control" is arXiv **2610.26684**, v1 **2026-10-19**, v2 **2026-11-04** — [GitHub](https://github.com/HeartMuLa/MuLaCover); [arXiv](https://arxiv.org/abs/2610.26684)
  - Code is Apache-2.0. Weights are **CC BY-NC 4.0 + MODEL_LICENSE**, and "model weights and generated outputs are restricted to noncommercial use" — [GitHub](https://github.com/HeartMuLa/MuLaCover)
  - Input is reference audio (converted to symbolic melody, harmony and optional drums) or melody/chord MIDI, plus lyrics and style tags. **Output is WAV audio.** The README has no piano-only mode or example — [GitHub](https://github.com/HeartMuLa/MuLaCover)

**B. Paper only (no code or weights found)**

- **Piano2Pop: Expanding Piano Demos into Pop Productions Without Aligned Supervision** by Yun-Chen Cheng, Chih-Pin Tan, Tzu-Hung Huang and Yi-Hsuan Yang. arXiv **2610.31510**, **2026-10-22**. It goes in the *reverse* direction (piano MIDI to pop audio, using a flow-matching model). For pseudo-supervision it uses an unnamed "pair-supervised piano-cover generator", and it was tested on 476 real piano-cover/pop pairs. No code link — [arXiv](https://arxiv.org/abs/2610.31510)
- **MIDI-REPA: Improving Text-to-Music Generation with Symbolic Representation Alignment**, arXiv **2610.20967**. The search snippet says it was submitted to ICASSP 2027. It is text-to-music, not piano covers (snippet only) — [arXiv](https://arxiv.org/abs/2610.20967)
- **MPEcho: A Melody and Phoneme-Aware Generative Framework for Controllable Cover Song Generation**, arXiv 2607.26698 (July 2026, just before the window). It is a vocal cover-song system (snippet only) — [arXiv](https://arxiv.org/pdf/2607.26698)

**C. Products / API**

- I found no product launch or new feature for piano covers in Sep–Nov 2026. These products already exist:
  - Songscription "piano cover creator" (audio or YouTube to solo piano) — [Songscription](https://www.songscription.ai/piano-cover-creator)
  - Klangio Piano2Notes "Arrangement Mode" (audio or YouTube to piano arrangement at Easy/Intermediate/Full; exports PDF/MIDI/MusicXML; blog dated 2025-06-11) — [Klangio blog](https://klang.io/blog/arrangement-update/)
  - PopPianoAI (DeepWave Muse Forge; full pop song to solo piano cover with chord symbols in about 3 min; snippet only) — [productdirs](https://productdirs.com/projects/poppianoai)
  - Suno Cover on V5.5 (style-transfer cover, audio output, no piano-specific mode found; third-party blog) — [apipass blog](https://apipass.dev/blogs/suno-cover-2026-guide)
  - ACE-Step 1.5 cover mode (keeps structure and duration, reimagines instrumentation; audio output) — [Medium](https://medium.com/@acemusic/meet-ace-step-1-5-an-on-device-music-model-that-beats-suno-on-common-eval-metrics-67ee61a1ce09); [ACE Music help](https://acemusic.ai/help/articles/create-an-ai-cover/)

**D. Relevant items just before the window (for context)**

- **MuScriptor** (Kyutai + Mirelo), **July 2026**. An open-weight decoder-only transformer for multi-instrument audio-to-MIDI transcription, with MIT inference code and three sizes (large is about 1.3B, HF-gated). It can restrict transcription to chosen instruments such as piano. The vendor reports Multi F1 48.2 vs 21.9 for YourMT3+ on its D_Test set — [MarkTechPost 2026-07-10](https://www.marktechpost.com/2026/07/10/kyutai-releases-muscriptor-an-open-weight-decoder-only-transformer-for-multi-instrument-music-transcription-to-midi/); [HF](https://huggingface.co/MuScriptor/muscriptor-large)
- **AMT-APC** repo: the search snippet showed "Updated Jul 4, 2026" with a beta model release. The arXiv paper 2409.14086 was revised in July 2026 (snippet only) — [GitHub](https://github.com/310hz/amt-apc)
- **Etude** (arXiv 2509.16522) was revised in Jan 2026 — [arXiv](https://arxiv.org/abs/2509.16522)
- A 2026 multi-instrument transcription benchmark paper (arXiv 2603.27528) has a winning entry that extends YourMT3+ with a MusicFM encoder. I found no new official MT3 or YourMT3 release in 2026 — [arXiv](https://arxiv.org/pdf/2603.27528)

### Inferences
- For this project, **tsumugi-piano-cover V2 is the release to evaluate first**. It is end-to-end from audio to piano MIDI (plus a score), small (about 300M parameters in total), has weights on HF and fits the existing AMT-APC comparison setup. Its CC BY-NC weights are a license constraint to record.
- BOSSA needs a separate lead-sheet extraction step (SheetSage) and a ~24 GB GPU. It is a research artifact with no license, so it is less practical.
- MuLaCover, Suno and ACE-Step produce audio. A piano-only result would require prompting for "solo piano" and then transcribing with a piano AMT model (e.g. MuScriptor piano-only, or an Aria/Kong-style piano transcriber). That adds two lossy stages, and I found no evidence that anyone has evaluated this route.

### Gaps
- I found no exact creation or last-commit date for the anime-song/tsumugi-piano-cover upstream repo or the HF upload date. Only the V2 changelog dates (Oct 20–21, 2026) are confirmed.
- The ISMIR 2026 conference dates and the full list of accepted papers were not retrieved. BOSSA is the only piano-arrangement paper I confirmed.
- I found no ICASSP 2027 preprint on piano cover generation. The arXiv Nov-2026 (2611.xxxxx) listings were not browsed directly, only via search, so a very recent preprint could be missing.
- New official releases from Google/Magenta, Meta, ByteDance, Tencent, Sony or Stable Audio that are specific to piano arrangement in Sep–Nov 2026 were not found. Searches did not surface any, but I did not check each company blog individually.

## Do they output symbolic MIDI or audio? Can audio output be transcribed to clean piano MIDI?

### Takeaway
Symbolic MIDI output: tsumugi-piano-cover (MIDI + MusicXML) and BOSSA (piano MIDI). Audio output: MuLaCover, Piano2Pop, Suno, ACE-Step. Turning audio into piano MIDI needs a separate AMT step, and I found no published evaluation of that route.

### Cited Findings
- tsumugi-piano-cover: `events_to_midi` produces MIDI, and V2 adds a two-staff MusicXML score — [GitHub](https://github.com/haveyouwantto/tsumugi-piano-cover)
- BOSSA outputs piano performances. The demo score is "manually engraved from MIDI" — [demo](https://zhaojw1998.github.io/bossa/)
- MuLaCover outputs a WAV file — [GitHub](https://github.com/HeartMuLa/MuLaCover)
- MuScriptor can transcribe audio to MIDI while restricting the output to chosen instruments (e.g. piano) — [HF](https://huggingface.co/MuScriptor/muscriptor-large)

### Inferences
- If an audio-to-audio "solo piano cover" from Suno, ACE-Step or MuLaCover sounds clean, transcribing it with a solo-piano AMT model would probably give usable MIDI. This is untested and no source reports it.

### Gaps
- No source compares "generate piano audio, then transcribe" against direct symbolic generation.

## Reported quality evidence (listening tests, comparisons to Pop2Piano / AMT-APC / Etude)

### Takeaway
The evidence is thin. No Sep–Nov 2026 release reports a head-to-head against Pop2Piano, AMT-APC or Etude that I could verify.

### Cited Findings
- tsumugi-piano-cover V2: on 12 held-out songs, the 2 s loudness-curve correlation with human covers is 0.51. An otherwise identical model without audio features scores 0.41, and two human covers of the same song correlate at 0.40. There are no listening tests or comparisons with other systems. Claims of "fewer rhythm breakdowns" and "closer to the original" are qualitative. Stated limitations: errors propagate from transcription, swing comes out straight, sustain pedal runs too long, and doublings carry into the score — [HF card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2); [GitHub](https://github.com/haveyouwantto/tsumugi-piano-cover)
- BOSSA: the demo compares against **PiCoGen2**, Audio2MIDI and an ablation without pre-training. It does not mention Pop2Piano, AMT-APC or Etude. Demos are best-of-3 samples per model. The abstract claims "substantial improvements in style-aware alignment and music quality" without numbers on the abstract page — [demo](https://zhaojw1998.github.io/bossa/); [arXiv](https://arxiv.org/abs/2608.03050)
- MuLaCover: objective evaluation across 6 systems and human evaluation across 4 systems (22 participants). It claims SOTA among open-source cover-song systems and parity with Suno-v5.5. These are vocal and full-song covers, not piano — [GitHub](https://github.com/HeartMuLa/MuLaCover); [arXiv](https://arxiv.org/abs/2610.26684)

### Inferences
- Running local A/B listening on the project's own worship and film test songs (tsumugi V2 vs AMT-APC vs Etude) is the only reliable way to rank these systems.

### Gaps
- I did not read BOSSA's full-paper numbers or subjective-test tables (only the abstract and demo page).

## Runnable locally on a 12 GB GPU (RTX 5070), or API-only?

### Takeaway
tsumugi-piano-cover very likely fits in 12 GB (small models, ~5 GB for the analysis step; not measured). BOSSA recommends ≥24 GB. MuLaCover is Linux-only and was tested on a B300, with no VRAM figure given. Songscription, Klangio, PopPianoAI and Suno are web or API products.

### Cited Findings
- tsumugi: "NVIDIA GPU recommended", CPU works but is much slower, Colab is supported. SheetSage2-Refined needs about 5 GB of GPU memory. Model parameters are 127M + 69M + 111M — [GitHub](https://github.com/haveyouwantto/tsumugi-piano-cover); [HF](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
- BOSSA: "≥24 GB recommended" because MusicGen-large and MuseCoco-1B are both kept in memory. It needs PyTorch 2.6 + **CUDA 11.8** and a compiled `pytorch-fast-transformers` (needs CUDA 11.x nvcc) — [GitHub](https://github.com/zhaojw1998/bossa-piano-cover-arrangement)
- MuLaCover: Linux + NVIDIA, Python 3.10, tested with PyTorch 2.10/CUDA 13.0 on a B300. Lazy loading reduces peak memory — [GitHub](https://github.com/HeartMuLa/MuLaCover)

### Inferences
- BOSSA's CUDA 11.8 requirement probably conflicts with an RTX 5070 (Blackwell, sm_120), which needs CUDA ≥12.8. Getting it running would need a port to newer CUDA or a rented GPU. This is my inference and was not tested.
- On a 12 GB card, MuLaCover would probably need WSL/Linux and may not fit. Not verified.

### Gaps
- None of these systems has a measured VRAM or runtime figure for 12 GB cards.

## ISMIR 2026 and ICASSP 2027 papers on piano arrangement or cover

### Takeaway
BOSSA (2608.03050) is the confirmed ISMIR 2026 piano-arrangement paper. I found no ICASSP 2027 piano-cover preprint. The only ICASSP 2027 submission I found is MIDI-REPA, which is text-to-music.

### Cited Findings
- The NUS Sound & Music Computing Lab announced BOSSA's ISMIR 2026 acceptance in July 2026 (per search summary) — [NUS SMC Lab](https://smcnus.comp.nus.edu.sg/); [arXiv](https://arxiv.org/abs/2608.03050)
- MIDI-REPA (arXiv 2610.20967) was submitted to ICASSP 2027 and covers text-to-music with symbolic alignment (snippet only) — [arXiv](https://arxiv.org/abs/2610.20967)

### Inferences
- More ICASSP 2027 preprints may appear on arXiv after the submission deadline. The project should check arXiv cs.SD again in a few weeks.

### Gaps
- I did not retrieve the ISMIR 2026 dates, location or full program.
