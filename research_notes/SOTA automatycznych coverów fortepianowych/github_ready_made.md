# Ready-made open-source piano cover generators on GitHub / Gitee (state as of 2026-11-18)

Method: GitHub REST search API (`/search/repositories`, sorted by stars; exact `pushed_at`/`created_at` dates come from the API on 2026-11-18). Queries: "piano cover", pop2piano, "piano arrangement", "audio to piano", "piano reduction", "song to piano", "piano cover generation", "piano accompaniment generation", AMT-APC, PiCoGen, "piano transcription arrangement", "youtube to piano sheet", topic:piano-cover, topic:music-arrangement, plus Chinese, Japanese and Korean keywords. Then I read the README, issues, releases and HF model cards of the relevant repos. "pushed" means the date of the last push to any branch, which is close to the last commit date but not identical. Star counts are as of 2026-11-18.

## Per-repo inventory: URL, stars, dates, weights, license, I/O, GPU

### Takeaway
There are six real audio-to-piano-cover generators with downloadable weights: Pop2Piano, AMT-APC, Etude, PiCoGen2, Music2MIDI and tsumugi-piano-cover. BOSSA also exists, but its input is a lead sheet plus reference audio, not a raw song. The only weights licensed for commercial use are Music2MIDI's: the repo is MIT-licensed and the weights ship with no separate license, but this is unverified (see Gaps). Pop2Piano's weights have **no license at all**. AMT-APC's weights have no license either, and they are almost certainly derived from Sony's hFT-Transformer, whose checkpoints are CC BY-NC-SA. Etude's weights are CC BY-NC-SA 4.0, PiCoGen2's are CC BY-NC-SA 3.0, and tsumugi-piano-cover's are CC BY-NC 4.0. So every one of these is non-commercial or legally unclear for a paid app.

### Cited findings

**A. Audio -> piano cover generators (core)**

| Repo | Stars | Last push / created | Code license | Weights + license | I/O | GPU | Inference / demo |
|---|---|---|---|---|---|---|---|
| [sweetcocoa/pop2piano](https://github.com/sweetcocoa/pop2piano) | 497 | push **2023-07-11** | none in API (no LICENSE detected); Colab is Apache-2.0 | HF [sweetcocoa/pop2piano](https://huggingface.co/sweetcocoa/pop2piano), **no license on the card**, updated 2023-08-29, 6,037 downloads/month | audio -> MIDI (T5/MT3 backbone, "composer" = style token) | small (T5-small class; not stated) | in HF `transformers` (Pop2PianoForConditionalGeneration), Colab, HF Spaces |
| [310hz/amt-apc](https://github.com/310hz/amt-apc) | 87 | push **2026-07-04** | MIT | GitHub Release tag `beta` ("22 Sep", year not shown, so likely 2024): `apc.pth` and `amt.pth`, **no license stated** | WAV / YouTube URL -> MIDI; `--style level1/2/3` | not stated | `python infer input.wav`; demo at 310hz.github.io/amt-apc |
| [Xiugapurin/Etude](https://github.com/Xiugapurin/Etude) | 20 | push **2026-05-02** | MIT per README (API reports NOASSERTION) | GitHub release zip, **CC BY-NC-SA 4.0, commercial use prohibited, contact author for a commercial license** | audio / YouTube -> MIDI; controls `--polyphony/--rhythm/--sustain` 0-2 | **>=16 GB VRAM** | `python infer.py --input`; requires a separate Py3.8 env for Spleeter; demo at xiugapurin.github.io/Etude |
| [tanchihpin0517/PiCoGen](https://github.com/tanchihpin0517/PiCoGen) (branches v1 and v2) | 55 | push **2025-12-04** | MIT (code) | **"dataset and trained models: CC BY-NC-SA 3.0"**; checkpoints inside Docker `tanchihpin0517/picogen2:latest-full` | YouTube URL / audio -> piano cover (MIDI) | **>16 GB** VRAM, Ubuntu | `./infer --input_url ... --docker_image ...`; SheetSage + Beat This; project page |
| [ytinyui/music2midi](https://github.com/ytinyui/music2midi) | 21 | push **2025-07-01**, created 2024-10-11 | MIT | Release `0.1.0` "Dataset and model checkpoint" (date "15 Oct", year not shown), no separate weights license | audio -> MIDI piano cover | not stated | conda env, web UI and `demo.ipynb`; MMM 2025 paper |
| [anime-song/tsumugi-piano-cover](https://github.com/anime-song/tsumugi-piano-cover) | 16 | push **2026-10-21**, created 2026-06-10 | MIT | HF `anime-song/tsumugi-piano-cover-v2` (default) and v1, **CC BY-NC 4.0 (non-commercial)**; also uses SheetSage2-Refined, which is non-commercial | audio (via tsumugi transcription) or multitrack MIDI -> piano MIDI, WAV and **MusicXML** (V2) | NVIDIA recommended; VRAM not stated | `python -m cover_studio` web UI on :8100, CLI, Python API, Colab (EN/JP); V2 released 2026-10-20, sheet music 2026-10-21 |
| [zhaojw1998/bossa-piano-cover-arrangement](https://github.com/zhaojw1998/bossa-piano-cover-arrangement) | 9 | push **2026-09-20**, created 2026-03-17 | none stated | Google Drive `checkpoints.tar.gz` (MuseCoco-1B backbone + stage1/2), **no license stated** | **lead-sheet MIDI (melody + chords) + reference audio for style** -> piano MIDI | **>=24 GB** recommended | notebook `test_stage2_generate.ipynb`; torch 2.6 / CUDA 11.8 + pytorch-fast-transformers build; ISMIR 2026, arXiv 2608.03050 |

Sources for the table: the READMEs linked in the first column; [Pop2Piano HF card](https://huggingface.co/sweetcocoa/pop2piano); [HF model search "pop2piano"](https://huggingface.co/models?search=pop2piano); [PiCoGen v2 branch](https://github.com/tanchihpin0517/PiCoGen/tree/v2); [PiCoGen LICENSE](https://github.com/tanchihpin0517/PiCoGen/blob/v2/LICENSE); [tsumugi-piano-cover-v2 HF card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2); [amt-apc releases](https://github.com/310hz/amt-apc/releases); [music2midi releases](https://github.com/ytinyui/music2midi/releases).

Further details:
- tsumugi-piano-cover V2 has three components: piano_cover (127.4M parameters), piano_ar (69.2M) and piano_perf2score (110.6M). It was fine-tuned on about 6,200 covers of about 2,200 songs and got 2,399 downloads in the last month. The architecture is a Global plus Local Transformer over 2-second patches, with 10 ms resolution. — [HF card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2), [repo](https://github.com/anime-song/tsumugi-piano-cover)
- Upstream tsumugi (the transcriber that tsumugi-piano-cover uses): [anime-song/tsumugi](https://github.com/anime-song/tsumugi). It has 105 stars, MIT code, and transcribes any instrument to MIDI. It recommends at least 12 GB VRAM and requires PyTorch 2.13. Initial release 2026-05-03, latest changelog 2026-10-17. The README does not state the checkpoint license, but another project, zeikar/pianoize, describes tsumugi as "code and weights are MIT" (secondary, unverified). — [tsumugi](https://github.com/anime-song/tsumugi), [pianoize](https://github.com/zeikar/pianoize)
- Etude's Extract stage uses a pretrained model "based on AMT-APC", and its checkpoints include `hft_transformer`. Etude's author reproduced AMT-APC on 2,858 JPOP and KPOP pairs and released Etude's code and checkpoints on 2025-09-07. — [amt-apc issue #7](https://github.com/310hz/amt-apc/issues/7), [Etude](https://github.com/Xiugapurin/Etude)
- Sony hFT-Transformer, which AMT-APC fine-tunes, has MIT code, but its **checkpoints are CC BY-NC-SA 4.0** because they were trained on MAESTRO v3. — [sony/hFT-Transformer](https://github.com/sony/hFT-Transformer)

**B. End-to-end apps and pipelines (song or YouTube -> piano MIDI / sheet / video)**

| Repo | Stars | Last push | What it uses | License | Notes |
|---|---|---|---|---|---|
| [Oh-Sheet-Team/oh-sheet](https://github.com/Oh-Sheet-Team/oh-sheet) | 400 | **2026-05-18** (created 2026-04-06) | YouTube / audio / MIDI -> PDF, MusicXML, MIDI. Basic Pitch (ONNX, CPU, about 10 s/song), optional Demucs, beat and key detection, rule-based arrangement, music21 + LilyPond | sidebar says MIT, README badge says "Proprietary" (**conflict**); the engraver service `oh-sheet-ml-pipeline` is **not open source** | Flutter frontend, `docker compose` for production. Not a learned cover model. |
| [zeikar/pianoize](https://github.com/zeikar/pianoize) | 3 | **2026-11-13** | Claude Code plugin: tsumugi + tsumugi-piano-cover + chordotomy; Claude writes the two-hand arrangement; FluidSynth; MP4 falling notes | MIT (code); inherits CC BY-NC for the cover weights | about 1.6 GB of weights; developed on an Apple Silicon Mac |
| [anosatsuk124/piakura](https://github.com/anosatsuk124/piakura) | 2 | **2026-10-17** | agent skill (Codex / Claude Code): YourMT3+, Basic Pitch, MuScriptor, SheetSage2, Demucs, Beat This, and others -> MIDI + SakuraMML per difficulty | Apache-2.0 (code); many dependencies are NC | experimental Docker with NVIDIA toolkit; about 20 GB disk |
| [confusedjpeg/amt-piano-visualizer](https://github.com/confusedjpeg/amt-piano-visualizer) | 2 | **2026-07-31** | Demucs htdemucs_6s, Basic Pitch on vocals, ByteDance piano transcription, chord-based algorithmic accompaniment -> two-hand MIDI + Synthesia MP4 | "personal and educational use" | no Docker |
| [yoda-jm/ai-arranger](https://github.com/yoda-jm/ai-arranger) | 1 | 2026-10-16 | Claude Code skills + Python toolkit: stems, transcription, rearrangement | none | listed by API only, README not read |
| [hd11/piano-sheet-extractor](https://github.com/hd11/piano-sheet-extractor) | 0 | 2026-03-10 | MP3 / YouTube -> melody -> MusicXML | none | trivial, melody only |
| [SleepyLGod/oh-my-score](https://github.com/SleepyLGod/oh-my-score) | 20 | 2026-10-25 | browser piano studio: audio-to-MIDI with engine comparison and score cleanup | MPL-2.0 | transcription, not cover generation (README not read) |
| [MoveBricksWorker/music_to_playable_piano_score](https://github.com/MoveBricksWorker/music_to_playable_piano_score) | 4 | 2026-10-11 (created 2026-10-10) | Audio -> playable piano score (MusicXML, MIDI, PDF): Demucs + Transkun + SheetSage2 with dual-lattice quantization | NOASSERTION | description from GitHub API only, README not read; SheetSage2 is NC |
| [tunglambk/piano-to-sheet-local](https://github.com/tunglambk/piano-to-sheet-local) | 0 | 2026-11-03 | **piano** audio/video -> two-staff sheet (MusicXML, PDF, MIDI), offline | **PolyForm Noncommercial 1.0.0** (per description) | transcription of a piano recording, not a cover |
| [kishin123/2midi4lin](https://github.com/kishin123/2midi4lin) | 12 | 2026-08-03 | Chinese tool: audio/video -> piano MIDI for the miHoYo desktop app "林离" (the only Chinese-language hit, found through the pop2piano in:readme query, so it probably uses Pop2Piano; unverified) | MIT | README not read |
| [masakiaota/pop2piano_WebUI](https://github.com/masakiaota/pop2piano_WebUI) | 0 | 2023-12-23 | Gradio UI for Pop2Piano | none | stale |
| [m1guelpf/cog-pop2piano](https://github.com/m1guelpf/cog-pop2piano) | 6 | 2022-11-22 | Replicate Cog container for Pop2Piano (Docker-like) | none | stale |

Source for each row: the README linked in that row, plus GitHub API search metadata.

**C. Adjacent tools: transcription or other tasks, not cover generation**
- [Yujia-Yan/Transkun](https://github.com/Yujia-Yan/Transkun): 529 stars, MIT, last push 2024-11-22, piano audio -> MIDI. — GitHub API
- [HemantKArya/Melodfy](https://github.com/HemantKArya/Melodfy): 133 stars, MIT, last push 2025-04-04, piano audio -> MIDI app. — GitHub API
- [wei-zeng98/piano-a2s](https://github.com/wei-zeng98/piano-a2s): 42 stars, Apache-2.0, last push 2024-09-17, piano audio -> score (IJCAI 2024). — GitHub API
- [jech2/D3PIA](https://github.com/jech2/D3PIA): 7 stars, last push 2026-02-10, diffusion piano accompaniment from lead sheet. [OlyMarco/D3EMO](https://github.com/OlyMarco/D3EMO): last push 2026-08-22, emotion-driven variant. — GitHub API
- [asigalov61/POP-Melody-Transformer](https://github.com/asigalov61/POP-Melody-Transformer): 8 stars, Apache-2.0, 2023, melody extraction / arrangement from symbolic input. — GitHub API
- [gerasmark/GreekSong2Piano-Dataset](https://github.com/gerasmark/GreekSong2Piano-Dataset): MIT, 659 Greek songs with piano covers (audio + MIDI), last push 2025-09-15. This is a non-English dataset, not a model. — GitHub API
- The piano-reduction repos (orchestral score -> piano) are all 0-4 stars student or thesis projects, for example [tpmmthomas/piano-reduction-with-midibert](https://github.com/tpmmthomas/piano-reduction-with-midibert) and [ivjiyeon/Automatic-Piano-Reduction](https://github.com/ivjiyeon/Automatic-Piano-Reduction). They are symbolic-only and not usable for audio. — GitHub API

### Inferences
- For "try immediately", the strongest current candidates are tsumugi-piano-cover V2, Etude and AMT-APC. Of the three, tsumugi-piano-cover is the most actively developed (a V2 and sheet-music export in Oct 2026). Etude needs at least 16 GB VRAM according to its README, which is above the RTX 5070's 12 GB, so it may need fp16 or chunking. PiCoGen2 (over 16 GB) and BOSSA (24 GB) are in the same position. AMT-APC and Pop2Piano should fit in 12 GB comfortably, but this is not stated by the authors.
- None of the generators is cleanly usable commercially. Pop2Piano's weights carry no license, which means all rights are reserved by default. The project already uses AMT-APC (commit 28100c3); its weights likely inherit hFT-Transformer's CC BY-NC-SA 4.0. This is an inference, because AMT-APC's README does not state it.

## Maintenance and breakage in 2025-2026

### Takeaway
Actively maintained in 2025-2026: tsumugi-piano-cover (pushed 2026-10-21), BOSSA (2026-09-20), AMT-APC (2026-07-04), Etude (2026-05-02), PiCoGen (2025-12-04) and Music2MIDI (2025-07-01, now quiet). The original Pop2Piano repo is dead (last push 2023-07-11). It survives only through HF `transformers`, where the usual failures are essentia and other missing dependencies.

### Cited findings
- The Pop2Piano repo has 3 open issues, none answered: #20 "colab fails to download result" (2023-09-02), #11 "essentia issue" (2023-02-04) and #5 "License" (2022-12-03). The License issue asks whether the code is open source and has no reply. — [issues](https://github.com/sweetcocoa/pop2piano/issues), [#5](https://github.com/sweetcocoa/pop2piano/issues/5)
- The Pop2Piano HF card says to install `transformers` from git and pins `pretty-midi==0.2.9` and `essentia==2.1b6.dev1034`. — [HF card](https://huggingface.co/sweetcocoa/pop2piano)
- Several HF Spaces running Pop2Piano show "runtime error", including "Pop2PianoFeatureExtractor requires essentia library". The processor requires the essentia, librosa, pretty_midi, scipy and torch backends. — [space mzr](https://huggingface.co/spaces/mzr/sweetcocoa-pop2piano), [space susnato/pop2piano_dev](https://huggingface.co/spaces/susnato/pop2piano_dev), [transformers docs](https://huggingface.co/docs/transformers/model_doc/pop2piano)
- AMT-APC issues: #7 "Issues with Fine-Tuning the AMT Model" was opened 2025-03-07 and is still open with 11 comments. The owner fixed a bug where training skipped the style vector (2025-03-16) and suggested lowering the θ thresholds. #8 "what does style actually do" was closed 2026-01-17. No inference breakage was reported. — [issue list](https://github.com/310hz/amt-apc/issues?q=is%3Aissue), [#7](https://github.com/310hz/amt-apc/issues/7)
- Etude issues: #1 "Availability of Pre-Trained Checkpoints" was closed 2025-09-20. #2 "Running on 6GB VRAM" was opened 2025-09-07 and is open. — [Etude issues](https://github.com/Xiugapurin/Etude/issues?q=is%3Aissue)
- Etude depends on Spleeter in a separate Python 3.8 conda env. The README warns that Demucs + Beat-Transformer gives worse beats. — [Etude](https://github.com/Xiugapurin/Etude)
- BOSSA requires building `pytorch-fast-transformers` with a matching nvcc, which takes 15-25 minutes, and pins torch 2.6.0 cu118. cu118 has no Blackwell (sm_120) kernels, so on an RTX 5070 this needs a newer torch/CUDA (my inference). — [BOSSA](https://github.com/zhaojw1998/bossa-piano-cover-arrangement)
- tsumugi requires PyTorch 2.13 and Python 3.10-3.14, so it is on a modern stack. tsumugi-piano-cover uses Python 3.12. — [tsumugi](https://github.com/anime-song/tsumugi), [tsumugi-piano-cover](https://github.com/anime-song/tsumugi-piano-cover)

### Inferences
- Pop2Piano via `transformers` is the easiest thing to run, but its quality is the 2022 baseline. Its essentia dependency is the most common breakage point; essentia wheels are Linux and macOS only, so it should run in a Linux container on the server.
- BOSSA's torch 2.6/cu118 pin and PiCoGen2's Docker image (age unknown) are the most likely to break on an RTX 5070 (Blackwell needs CUDA 12.8+). tsumugi on torch 2.13 is the safest for Blackwell.

### Gaps
- I did not test-run anything.
- I could not see the contents of open issues in PiCoGen (2 open), Music2MIDI or tsumugi-piano-cover.
- The release years for AMT-APC `beta` ("22 Sep") and Music2MIDI `0.1.0` ("15 Oct") were not shown on the page. They are probably 2024, but this is unverified.

## Forks and improved versions of Pop2Piano / AMT-APC

### Takeaway
I found no community fork of Pop2Piano or AMT-APC with bigger checkpoints, multilingual data or added style control. The "improvements" are new projects by other authors. Etude was built on AMT-APC's extractor by the person who reproduced AMT-APC on JPOP and KPOP. tsumugi-piano-cover uses Pop2Piano's piano tracks in pretraining. Music2MIDI is a separate model. Pop2Piano forks are just copies.

### Cited findings
- The forks named pop2piano are [AgentHitmanFaris/pop2piano](https://github.com/AgentHitmanFaris/pop2piano) (0 stars, pushed 2025-12-19, same description as upstream), [vincefav/pop2piano-colab](https://github.com/vincefav/pop2piano-colab) (0 stars, 2026-07-31), [StevenSchuerstedt/pop2piano_linux](https://github.com/StevenSchuerstedt/pop2piano_linux) (2023) and [xiangze/pop2piano2magenta](https://github.com/xiangze/pop2piano2magenta) (MIT, 2022). — GitHub API search
- On HF, the only Pop2Piano checkpoints are `sweetcocoa/pop2piano`, `susnato/pop2piano_dev` (the port during transformers integration, 2023-08-28) and a tiny-random test model. There is no larger or retrained Pop2Piano. — [HF search](https://huggingface.co/models?search=pop2piano)
- Etude's author reproduced AMT-APC on 2,858 JPOP and KPOP pairs, with weak alignment taken from PiCoGen2. That led to Etude, with controllable polyphony, rhythm and sustain. — [amt-apc #7](https://github.com/310hz/amt-apc/issues/7), [Etude](https://github.com/Xiugapurin/Etude)
- tsumugi-piano-cover pretrains on about 3,000 h of solo piano, including "Pop2Piano's piano tracks", PiJAMA, PIAST and MAESTRO. V2 is fine-tuned on about 6,200 covers of about 2,200 songs, the largest cover fine-tuning set among these repos. — [repo](https://github.com/anime-song/tsumugi-piano-cover), [HF](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)

### Inferences
- If "better than the originals" means newer, larger data and style control, the candidates are tsumugi-piano-cover V2 (Oct 2026) and Etude (Sep 2025). Both carry NC weights.

### Gaps
- I did not walk the full GitHub fork network graphs of pop2piano or amt-apc; I only searched names and descriptions. A forked repo with renamed weights could exist unnoticed.
- No published quality comparison exists between tsumugi-piano-cover and Etude or AMT-APC. The tsumugi README does not compare against them.

## End-to-end packaged pipelines, Docker, and CJK / Gitee sweep

### Takeaway
The packaged end-to-end options with YouTube input are PiCoGen2 (an official Docker image with checkpoints), AMT-APC and Etude (both CLIs accept YouTube URLs), Oh Sheet (a full app producing a PDF score, but transcription plus rules rather than a cover model, and its engraver is closed) and tsumugi-piano-cover Cover Studio (web UI, MusicXML out). The new 2026 trend is LLM-agent pipelines: pianoize, piakura and ai-arranger, which are Claude Code or Codex skills. Searches in Chinese, Japanese and Korean on GitHub returned **zero** repositories, and Gitee search could not be read.

### Cited findings
- PiCoGen2 Docker: `tanchihpin0517/picogen2:latest-full` includes the beat-tracker, SheetSage and decoder checkpoints, run with `./infer --input_url "<youtube>"`. — [PiCoGen v2](https://github.com/tanchihpin0517/PiCoGen/tree/v2)
- AMT-APC: `python infer 'https://www.youtube.com/watch?v=...'` (needs yt-dlp). Etude: `python infer.py --input <YouTube URL>`. — [amt-apc](https://github.com/310hz/amt-apc), [Etude](https://github.com/Xiugapurin/Etude)
- Oh Sheet: YouTube or audio in; PDF, MusicXML and humanized MIDI out; `docker compose up -d` for production; MIT vs "Proprietary" license conflict; engraver not open source. — [oh-sheet](https://github.com/Oh-Sheet-Team/oh-sheet)
- tsumugi-piano-cover Cover Studio: web UI, MIDI out, WAV rendering, MusicXML (V2). — [repo](https://github.com/anime-song/tsumugi-piano-cover)
- The GitHub API returned 0 results for "钢琴改编", "钢琴 伴奏 生成", "钢琴 翻弹", "ピアノ アレンジ", "ピアノカバー", "피아노", "피아노 편곡" (all sorted by stars). The only Chinese-language hit came from the "pop2piano in:readme" query: kishin123/2midi4lin. That query also surfaced the awesome-lists [affige/genmusic_demo_list](https://github.com/affige/genmusic_demo_list) (802 stars, pushed 2026-11-11) and [ad-si/awesome-music-production](https://github.com/ad-si/awesome-music-production) (1,565 stars), which mention Pop2Piano; they were not read in full. Web searches in Japanese and Korean surfaced only the known projects (Pop2Piano, PiCoGen, AMT-APC, Etude) plus a pointer to `jerryzhou196/pianoify` from a Hacker News comment, which I did not verify. — [JA search result: HN item](https://hn.svelte.dev/item/49389846), [KO: uithub misya11p/amt-apc mirror](https://uithub.com/misya11p/amt-apc)
- AMT-APC appears to have been previously hosted as misya11p/amt-apc, with its project page at misya11p.github.io/amt-apc. — [uithub mirror](https://uithub.com/misya11p/amt-apc)
- Gitee: `gitee.com/search` redirects to so.gitee.com, which renders client-side; the fetch returned no results. — [so.gitee.com](https://so.gitee.com/?q=pop2piano)

### Inferences
- To get from YouTube to a sheet: a cover model producing MIDI (any of the above), then quantization and engraving. tsumugi-piano-cover V2 (piano_perf2score) and Oh Sheet are the only ones that produce sheet music out of the box. For your stack, engraving could be done with music21 + LilyPond/MuseScore, as Oh Sheet does.
- Licensing for a paid app: every learned cover model found is NC or unlicensed. A commercially clean path requires your own trained weights; the MIT code from AMT-APC, Etude, tsumugi-piano-cover or Music2MIDI can be reused. Upstream components also need checking: Spleeter (MIT code), Demucs (MIT code, weights license not found), SheetSage (NC) and Basic Pitch (Apache-2.0).

### Gaps
- Gitee could not be searched because the page is JS-rendered. Chinese repos there are **unverified**.
- `jerryzhou196/pianoify` was not checked.
- I found no Docker image for AMT-APC or Etude beyond `pip`/conda instructions.
- Papers with Code (shut down / redirected to HF in 2025) and the awesome-lists were not swept separately in this pass.
- I did not verify the Music2MIDI checkpoint's VRAM, quality or license beyond the repo's MIT.
