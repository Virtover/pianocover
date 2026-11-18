# Availability and licensing of automatic piano-cover / arrangement tools (for a paid Android app, server-side GPU)

Checked 2026-11-18. License fields come from the GitHub REST API (`api.github.com/repos/<repo>` → `license.spdx_id`), Hugging Face API (`huggingface.co/api/models/<id>` → `cardData.license`), and raw README/LICENSE files. "Unverified" means I did not read a primary page for it in this pass.

## 1. Open models: code license vs. weights license vs. training data

### Takeaway
None of the dedicated piano-**cover** models has a clean commercial grant on its weights. Etude and PiCoGen2 forbid commercial use outright (CC BY-NC-SA). Pop2Piano has no license at all, which legally means all rights reserved. AMT-APC is MIT in the repo, but the weights don't have their own license statement, and they descend from MAESTRO (CC BY-NC-SA 4.0) and from YouTube covers. The only cleanly licensed pieces are generic transcribers (Basic Pitch Apache-2.0, MT3 Apache-2.0, YourMT3 weights apache-2.0 with GPL-3.0 code, Transkun MIT), and even those were usually trained on MAESTRO or Slakh, whose licenses are non-commercial.

### Cited Findings

**Pop2Piano (sweetcocoa/pop2piano, HF `sweetcocoa/pop2piano`, transformers `Pop2PianoForConditionalGeneration`)**
- GitHub repo: the API returns `license: null`, so there is no LICENSE file. Last push was 2023-07-11. The README covers only the dataset download and preprocessing, with no license statement. — [GitHub API](https://api.github.com/repos/sweetcocoa/pop2piano), [README](https://github.com/sweetcocoa/pop2piano)
- The HF model card metadata block is empty (`{}`), and the license field is absent (API `cardData.license` = None). — [HF model card](https://huggingface.co/sweetcocoa/pop2piano)
- The HF card's usage section requires `essentia==2.1b6.dev1034` (plus pretty-midi and librosa). Essentia's repo license is AGPL-3.0. — [HF card](https://huggingface.co/sweetcocoa/pop2piano), [MTG/essentia GitHub API](https://api.github.com/repos/MTG/essentia)
- Training data: the repo's `train_dataset.csv` is a list of YouTube videos (pop originals paired with piano covers) for download. — [README](https://github.com/sweetcocoa/pop2piano)
- Hosted version: Replicate `m1guelpf/pop2piano` runs on an Nvidia A100 (80GB) at about $0.050/run, with typical runs finishing within 36 s. The page shows no license. — [Replicate](https://replicate.com/m1guelpf/pop2piano)
- I found no GitHub issue or statement from the authors about commercial use of the weights (searched). — [search via HF/transformers docs](https://huggingface.co/docs/transformers/model_doc/pop2piano.md)

**AMT-APC (310hz/amt-apc; the old misya11p/amt-apc URL redirects there)**
- Repo license is MIT, last pushed 2026-07-04. — [GitHub API](https://api.github.com/repos/310hz/amt-apc)
- The weights come as a GitHub release asset: `wget ... releases/download/beta/apc.pth`. The README has no separate weights license and no commercial-use remark (grep for licen/commercial found nothing). — [README](https://github.com/310hz/amt-apc)
- It fine-tunes an AMT model (paper arXiv 2409.14086), trains on a dataset of YouTube piano covers, and its inference accepts YouTube URLs via yt-dlp. — [README](https://github.com/310hz/amt-apc)
- The underlying AMT model (hFT-Transformer, per the paper) is trained on MAESTRO. MAESTRO is "made available by Google LLC under a Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0) license." — [MAESTRO page](https://magenta.withgoogle.com/datasets/maestro). (That AMT-APC's base is hFT-Transformer/MAESTRO comes from the paper and is not re-verified in this pass.)

**Etude (Xiugapurin/Etude, 2025: "Extract, strucTUralize, and DEcode")**
- Dual license. The source code is MIT. The README says: "All pre-trained model checkpoints (files ending in `.pth`, `.pkl`, etc.) ... are made available under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**." It also says: "Use of the pre-trained models for commercial purposes is strictly prohibited under this license. For inquiries about commercial licensing, please contact the project owner." — [LICENSE](https://github.com/Xiugapurin/Etude/blob/main/LICENSE), [README](https://github.com/Xiugapurin/Etude)
- Hardware stated in the README: "GPU with at least **16GB of VRAM** (CUDA) or Apple Silicon with 16GB+ unified memory." — [README](https://github.com/Xiugapurin/Etude)
- The training data format follows Pop2Piano: CSV pairs of YouTube IDs (`pop_ids`, `piano_ids`). Users can train their own decoder with the MIT code. — [README](https://github.com/Xiugapurin/Etude)

**PiCoGen / PiCoGen2 (tanchihpin0517/PiCoGen, branch `v2`)**
- The GitHub API reports no SPDX license for the repo. The LICENSE file on the v2 branch says: "All *code* in this repository is MIT-licensed, but dataset and trained models are distributed under CC BY-NC-SA 3.0, as they are derived from user contributions on HookTheory." — [v2 LICENSE](https://github.com/tanchihpin0517/PiCoGen/blob/v2/LICENSE)
- PiCoGen2 depends on SheetSage. SheetSage's README states: "the 'Sheet Sage' system as a whole contains numerous additional licensing considerations that especially affect commercial use ... transcription models ... are released under CC BY-NC-SA 3.0. Moreover, Sheet Sage makes use of madmom, Jukebox, and Melisma, which all have additional licensing terms affecting commercial use." — [SheetSage README](https://github.com/chrisdonahue/sheetsage)
- Hardware: "Ensure you are running Linux with GPU memory larger than 16GB." Distributed as Docker `tanchihpin0517/picogen2:latest-full`, with checkpoints for the beat trackers, SheetSage, and the decoder. SheetSage's Jukebox features alone need a GPU with at least 12GB and about 10GB of downloads. — [PiCoGen v2 README](https://github.com/tanchihpin0517/PiCoGen/tree/v2), [SheetSage README](https://github.com/chrisdonahue/sheetsage)

**madmom (beat/downbeat tracking, used by SheetSage/PiCoGen and by many cover pipelines)**
- Code is BSD. "all model and data files are distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. If you want to include any of these files ... or technology which utilises them in a commercial product, please contact Gerhard Widmer." — [madmom README](https://github.com/CPJKU/madmom)

**Transkun (Yujia-Yan/Transkun)**
- Repo license is MIT, last pushed 2024-11-22. The checkpoints (shipped in the pip package and on Google Drive) are trained on "Maestro V3". The README has no separate weights license. — [GitHub API](https://api.github.com/repos/Yujia-Yan/Transkun), [README](https://github.com/Yujia-Yan/Transkun)
- MAESTRO is CC BY-NC-SA 4.0 (see above). — [MAESTRO](https://magenta.withgoogle.com/datasets/maestro)

**YourMT3 / YourMT3+ (mimbres/YourMT3)**
- The code repo is GPL-3.0. The HF checkpoint repo `mimbres/YourMT3` says `license: apache-2.0` and "We use this repository solely to host the model checkpoints of YourMT3+." — [GitHub API](https://api.github.com/repos/mimbres/YourMT3), [HF card](https://huggingface.co/mimbres/YourMT3)

**MT3 (magenta/mt3)**
- Repo license is Apache-2.0, last pushed 2026-10-27. — [GitHub API](https://api.github.com/repos/magenta/mt3)

**Basic Pitch (spotify/basic-pitch)**
- Licensed Apache-2.0, and the HF `spotify/basic-pitch` card is apache-2.0. It is "instrument-agnostic and supports polyphonic instruments ... works best on one instrument at a time." — [README](https://github.com/spotify/basic-pitch), [HF API](https://huggingface.co/api/models/spotify/basic-pitch)

**Generative audio models (they output audio, not MIDI arrangements)**
- MusicGen: the audiocraft code is MIT, but the `facebook/musicgen-large` weights are `cc-by-nc-4.0`. — [audiocraft GitHub API](https://api.github.com/repos/facebookresearch/audiocraft), [HF API](https://huggingface.co/api/models/facebook/musicgen-large)
- ACE-Step: code Apache-2.0, `ACE-Step/ACE-Step-v1-3.5B` weights apache-2.0. Newer v1.5 repos (`ACE-Step/Ace-Step1.5`, `acestep-v15-xl-*`) exist on HF, but I did not check their license. — [GitHub API](https://api.github.com/repos/ace-step/ACE-Step), [HF API](https://huggingface.co/api/models/ACE-Step/ACE-Step-v1-3.5B)
- Stable Audio Open 1.0: "Stability AI Community License". The card points to https://stability.ai/license for commercial terms; I did not read the revenue threshold. Training used 486,492 CC0/CC-BY/CC Sampling+ recordings (Freesound + FMA). The card says "It is better at sound effects and field recordings than at music" and should not be used downstream "without further risk evaluation and mitigation." The tools code (stable-audio-tools) is MIT. — [HF card](https://huggingface.co/stabilityai/stable-audio-open-1.0), [GitHub API](https://api.github.com/repos/Stability-AI/stable-audio-tools)

### Inferences
- **Pop2Piano:** without any license, we have no right to redistribute or commercially use the weights or code (default copyright). The transformers integration (Apache-2.0 code) doesn't change the status of the weights. On top of that, its processor depends on AGPL-3.0 essentia. Server-side AGPL use is generally fine if unmodified, but it is a compliance burden. Treat it as not usable commercially without written permission from the authors (Jongho Choi / Kyogu Lee, SNU).
- **Etude and PiCoGen2:** explicitly non-commercial, so they are out unless we get a commercial license (Etude invites inquiries). Their MIT code could be retrained on our own data, but the PiCoGen2 pipeline also depends on SheetSage, madmom, and Jukebox, which carry their own non-commercial terms.
- **AMT-APC and Transkun:** they are the "least bad". Both have MIT repos that ship weights with no separate restriction. But the weights are derived from MAESTRO (CC BY-NC-SA 4.0), and for AMT-APC also from YouTube covers. Whether NC dataset terms "infect" trained weights is legally unsettled. This is a risk to document, not a clear prohibition. Our existing stack (AMT-APC + Transkun) is therefore the most commercially defensible open option, but it has that residual risk.
- **Any pipeline using madmom's pretrained beat models inherits CC BY-NC-SA 4.0.** Check our own code for madmom/essentia imports.
- MusicGen, Stable Audio Open, and ACE-Step generate audio. They are not piano-cover MIDI generators, so they are irrelevant as a "gotowiec" (ready-made solution). Only ACE-Step (apache-2.0) is commercially clean among them.

### Gaps
- No primary statement from the AMT-APC authors on weights licensing or commercial use. It's worth an issue or email to 310hz.
- Pop2Piano: no statement from the authors found. Unknown whether they would grant a license.
- The Stability AI Community License revenue threshold was not read on stability.ai/license.
- I did not check the licenses of newer HF piano-arrangement models beyond Pop2Piano mirrors (`susnato/pop2piano_dev`, a tiny test model). A lead I could not verify: a Kyutai "muscriptor" blog post dated 2026-07-10 surfaced in search ([HN link](https://hn.svelte.dev/item/49389846)); its content and license are unknown.
- Magenta's newer models (e.g. Magenta RT) were not checked.

## 2. Hardware fit: RTX 5070 (12 GB, sm_120, PyTorch 2.7+/CUDA 12.8)

### Takeaway
The stated VRAM needs of Etude (16GB) and PiCoGen2 (>16GB) exceed 12 GB, though both are non-commercial anyway. Pop2Piano (T5-based), AMT-APC, Transkun, Basic Pitch, and MT3/YourMT3 are small transcription-scale models that should fit. Our repo already has measured 5070 results for AMT-APC (commit 28100c3).

### Cited Findings
- Etude: "GPU with at least 16GB of VRAM". — [Etude README](https://github.com/Xiugapurin/Etude)
- PiCoGen2: "GPU with more than 16GB memory", Ubuntu only (tested). — [PiCoGen v2 README](https://github.com/tanchihpin0517/PiCoGen/tree/v2)
- SheetSage with Jukebox: "a GPU w/ at least 12GB memory", "will likely take several minutes to complete". — [SheetSage README](https://github.com/chrisdonahue/sheetsage)
- Pop2Piano on Replicate (A100 80GB) typically finishes within 36 s per prediction, at about $0.05/run. — [Replicate](https://replicate.com/m1guelpf/pop2piano)
- AMT-APC install is plain `pip install torch torchaudio soundfile pretty-midi tqdm` on Python 3.10, with no pinned old CUDA. — [README](https://github.com/310hz/amt-apc)

### Inferences
- Models that pin only `torch` (AMT-APC, Transkun) should work with PyTorch 2.7+cu128 on sm_120. Docker images with old CUDA (PiCoGen2) or old TensorFlow (Basic Pitch's TF/CoreML/ONNX backends; MT3's JAX/T5X) may need rebuilding for Blackwell. Basic Pitch also runs on CPU via ONNX/TFLite, so this matters less for it.
- Etude might fit in 12 GB with fp16 or shorter segments, but this is unverified.

### Gaps
- Per-song inference times on 5070-class GPUs were not reported by any of these authors except Replicate's A100 figure. Use our own benchmarks (repo commit 28100c3).

## 3. Commercial products / APIs

### Takeaway
Klangio is the only vendor found with a public, priced self-serve music-transcription API, and it offers transcription (piano/guitar/bass/vocal) rather than "make a piano cover of a full band song." Songscription has a dedicated piano-arrangement or cover feature, but its API is enterprise-only on request. Neither publishes clear terms on redistributing outputs in a third-party paid app; that needs a contract.

### Cited Findings
- **Klangio API** (klang.io/api) offers: Music Transcription ("Convert audio files into MIDI, MusicXML, PDF, or GP5 formats ... Available for piano, guitar, bass, and vocal recordings"), Source Separation, Beat Tracking, Strum Recognition, and Chord Recognition. The page mentions no piano-arrangement endpoint. Pricing per month:
  - Basic: $0, 50 requests, max 15 s audio
  - Startup: $99, 500 requests, $0.20 per extra, max 300 s
  - Business: $499, 3,000 requests, $0.17 per extra, max 300 s, 5 req/s
  - Enterprise: 10,000 requests, $0.10 per extra, max 600 s

  The page states no terms on commercial use or redistribution of outputs. The terms-of-use URL I tried (klang.io/terms-of-use/) returned 404. — [Klangio API](https://klang.io/api/)
- **Songscription:** product pages for a "Piano Cover Creator" and "Piano Arrangement Generator" (a solo piano arrangement with separate left- and right-hand parts). — [Piano cover creator](https://www.songscription.ai/piano-cover-creator), [Arrangement generator](https://www.songscription.ai/piano-arrangement-generator). Its plans:
  - Free: first 30 s of each song
  - Plus: 60 min of transcription and 600 credits/month
  - Pro: 300 min and 3,000 credits/month
  - Enterprise: custom; "API access" is listed only here

  Prices weren't rendered. Exports are MIDI, MusicXML, PDF, and Guitar Pro. The pricing page doesn't mention commercial use. — [Songscription pricing](https://songscription.ai/pricing)
- A MusicRadar review headline says "Humans will be doing all the serious music transcription for the foreseeable future". Per the search snippet, it said the piano arrangement feature was in alpha (the review may be dated). — [MusicRadar](https://www.musicradar.com/music-tech/humans-will-be-doing-all-the-serious-music-transcription-for-the-foreseeable-future-songscription-review)
- Songscription received a MIDI Association innovation award listing. — [midi.org](https://midi.org/innovation-award/songscription-ai)
- Replicate-hosted Pop2Piano exists as a pay-per-run API (about $0.05/run). It does not cure Pop2Piano's missing weights license. — [Replicate](https://replicate.com/m1guelpf/pop2piano)
- Other consumer tools surfaced in search but were not examined: Super Simple Piano (transcribes existing YouTube piano covers into MIDI/MusicXML), Wondera (piano accompaniment for vocals), musicmake.ai/musci.io (audio-to-MIDI). — [supersimplepiano](https://www.supersimplepiano.com/youtube), [wondera](https://wondera.ai/tools/en/ai-piano-version-remake-tools), [musci.io](https://musci.io/ai-midi-generator)

### Inferences
- For a paid app, the realistic "gotowiec" commercial path is a B2B contract: Songscription Enterprise (arrangement + API) or Klangio Business/Enterprise (transcription only, so we'd still need our own arrangement step). Get explicit written terms on output ownership and redistribution. Also, the song copyright of user-uploaded music remains our problem regardless of vendor.
- At Klangio's $0.17–0.20 per request and a 300 s cap on non-enterprise plans, a full song often needs the Enterprise tier (600 s). Server-side open models on our own 5070 are far cheaper per song.

### Gaps
- Not checked in this pass (no primary pages read): AnthemScore (desktop licensing), Melody Scanner, Samplab, Moises (has a developer API, "Music AI", per prior knowledge; pricing and terms unverified), LALAL.AI (stem-separation API, unverified), ScoreCloud, Tunebat, MuseScore features. None of these is known to offer a piano-*cover/arrangement* API. Moises, LALAL, and Samplab are separation or transcription tools.
- Klangio's and Songscription's terms of service on output rights were not read: the Klangio URL returned 404, and Songscription's terms page wasn't fetched.
- No independent quality comparison (Songscription arrangement vs. AMT-APC/Pop2Piano/Etude) was found.

## 4. Which options are realistic ready-made solutions for commercial piano-cover MIDI today?

### Takeaway
No open piano-cover model is both high-quality and cleanly licensed for commercial use. In order of legal defensibility:
1. Our current stack (AMT-APC MIT + Transkun MIT, with the MAESTRO/YouTube-data residual risk).
2. A commercial contract with Songscription (Enterprise API, arrangement feature) or with Klangio (transcription API, plus our own arrangement).
3. Seeking a commercial license from the Etude authors, who explicitly invite inquiries.

Pop2Piano (no license), PiCoGen2, and Etude as shipped (CC BY-NC-SA) should not be shipped in a paid app.

### Cited Findings
- Etude: "For inquiries about commercial licensing, please contact the project owner." — [Etude README](https://github.com/Xiugapurin/Etude)
- PiCoGen2/SheetSage NC terms come from HookTheory user contributions. — [PiCoGen v2 LICENSE](https://github.com/tanchihpin0517/PiCoGen/blob/v2/LICENSE), [SheetSage README](https://github.com/chrisdonahue/sheetsage)
- Songscription Enterprise includes "API access". — [Songscription pricing](https://songscription.ai/pricing)
- Klangio has a priced self-serve API, transcription only. — [Klangio API](https://klang.io/api/)

### Inferences
- The Etude and PiCoGen2 codebases are MIT, so their *methods* (extract → structuralize → decode) can be re-implemented or retrained on data we have rights to. Their YouTube-pair training data has the same copyright issue as our web-cover scraping, though.
- Basic Pitch (Apache-2.0) and MT3 (Apache-2.0) / YourMT3 weights (apache-2.0) are the cleanest licensed transcription components. They could serve as a melody/stem-transcription front end feeding a rule-based or self-trained arranger.

### Gaps
- Legal question of whether models trained on CC BY-NC-SA data (MAESTRO) are themselves bound by NC: no authoritative source consulted. Needs legal advice.
