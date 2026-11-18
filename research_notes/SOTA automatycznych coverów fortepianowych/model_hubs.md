# Piano cover models on model hubs (HF, ModelScope, Replicate, fal.ai), as of 2026-11-18

Method: Hugging Face numbers (downloads = HF's rolling ~30-day count, likes, lastModified, license tag) were read straight from the HF REST API (`huggingface.co/api/models|spaces|datasets?search=...`) on 2026-11-18. ModelScope was queried through its public search API. Replicate and fal.ai were checked by web search and page fetch. Search terms: pop2piano, piano cover, piano-cover, pianocover, amt-apc, etude, picogen, piano arrangement, arrang, piano, pianist, audio2piano, song2piano, plus Chinese terms on ModelScope.

## Hugging Face: models and Spaces that turn a song into a piano arrangement

### Takeaway
The hub has very few real "song to piano cover" models: Pop2Piano (2023, no license), tsumugi-piano-cover V1/V2 (Oct 2026, CC BY-NC 4.0, the newest and most complete pipeline), PiCoGen2 (CC BY-NC-SA 3.0 mirror plus a working Space) and one unlicensed hobby checkpoint. None of them can be used commercially as published. The only Space that runs well and covers several engines is StrikeKhan's free ZeroGPU demo (Pop2Piano vs. ByteDance transcription). No fine-tunes of Pop2Piano on more data (multilingual, J-pop, worship) exist on HF.

### Cited Findings
**Models**
- **anime-song/tsumugi-piano-cover-v2**: 2,399 downloads, 1 like, created 2026-10-20, lastModified 2026-10-21. License field `cc-by-nc-4.0`, `library_name: pytorch`. Three models: `piano_cover` CoverModel (127.4 M params), `piano_ar` (69.2 M), `piano_perf2score` (110.6 M, writes MusicXML) — [HF API](https://huggingface.co/api/models/anime-song/tsumugi-piano-cover-v2), [card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
  - Input is the original recording plus a multi-instrument MIDI transcription made by tsumugi (stems, instruments, velocity, beats, chords, key). V2 adds beats and melody from SheetSage2-Refined and MERT audio features. It generates one bar at a time on the song's beat grid. Output is MIDI, which perf2score turns into a two-staff MusicXML score — [card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
  - Training per the card: pretraining on about 46,000 solo-piano MIDI pieces, then fine-tuning on "about 6,200 piano covers of about 2,200 original songs". There are 1,916 anonymous performer indices to pick from. The card does not name the data source — [card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
  - Quality claim: on 12 held-out songs, the loudness-curve correlation with human covers is 0.51. That is higher than 0.41 without audio features and higher than 0.40 between two human covers of the same song. No listening test is reported — [card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
  - SheetSage2-Refined needs about 5 GB of GPU memory. The card says the score model "writes what was played", so its scores can be harder to read than a hand-made arrangement — [card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
  - License text, quoted: "The weights are released under CC BY-NC 4.0 (non-commercial use only). SheetSage2-Refined, which computes the inputs, is also non-commercial." The code repo, github.com/anime-song/tsumugi-piano-cover, is MIT, has 16 stars and was last pushed 2026-10-21 — [card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2), [GitHub API](https://api.github.com/repos/anime-song/tsumugi-piano-cover)
  - It includes a local web UI (`python -m cover_studio`) that runs the whole pipeline from audio: transcription, beats, generation and score — [card](https://huggingface.co/anime-song/tsumugi-piano-cover-v2)
- **anime-song/tsumugi-piano-cover (V1)**: 117 downloads, lastModified 2026-10-20, `cc-by-nc-4.0`, 127.2 M params. Works on 2-second patches with a 10 ms grid and runs without the original audio (MIDI from tsumugi only) — [card](https://huggingface.co/anime-song/tsumugi-piano-cover)
- **anime-song/SheetSage2-Refined** (dependency of V2): 8,111 downloads, lastModified 2026-11-05, `cc-by-nc-4.0`, `base_model: m-a-p/SheetSage2`. The card says "The training songs are mostly Japanese pop and anime songs". The weights come from SheetSage2 and MERT-v2-FullSong, both CC BY-NC 4.0. The inference code is MIT — [card](https://huggingface.co/anime-song/SheetSage2-Refined)
- **sweetcocoa/pop2piano**: 6,037 downloads, 17 likes, created 2022-11-05, lastModified 2023-08-29. The card metadata is empty (`{}`), so there is **no license field**. The GitHub repo (497 stars, last push 2023-07-11) also has no license. The card says it was "mainly trained on Korean Pop music". Usage is through `transformers` `Pop2PianoForConditionalGeneration` with a `composer` parameter (style) — [HF API](https://huggingface.co/api/models/sweetcocoa/pop2piano), [card](https://huggingface.co/sweetcocoa/pop2piano), [GitHub API](https://api.github.com/repos/sweetcocoa/pop2piano)
- **susnato/pop2piano_dev**: 186 downloads, 2023-08-28, no license. This is the porting checkpoint used during the transformers integration — [HF](https://huggingface.co/susnato/pop2piano_dev)
- **teamup-tech/PiCoGen2-weights**: 46 downloads, created and modified 2026-10-22, license field `cc-by-nc-sa-3.0`. It mirrors Zenodo record 13380452 (`model_ft_00070000`). Card, quoted: "The Zenodo record licenses the checkpoint CC-BY-4.0, while the repository's LICENSE distributes trained models under CC-BY-NC-SA-3.0, since they derive from user contributions on HookTheory. The card declares the more restrictive of the two." — [card](https://huggingface.co/teamup-tech/PiCoGen2-weights)
- **zerostratos/pop-to-piano-llama-v1**: 10 downloads, 2025-06-18. No README and no license. Only `model.safetensors` and `training_args.bin`. Unverified what it does — [HF](https://huggingface.co/zerostratos/pop-to-piano-llama-v1)
- **LongshenOu/m2m_arranger** (629 downloads, 2024-08-14) and **m2m_pianist** (5 downloads): auto-generated empty cards with no license. These are symbolic MIDI-to-MIDI arrangers (the author's "m2m" work), not audio-to-piano — [HF](https://huggingface.co/LongshenOu/m2m_arranger)
- **AMT-APC is not on HF**: author `misya11p` on HF has only GPT-2 Japanese models, and no `310hz` account turned up. The weights are only in the GitHub release github.com/310hz/amt-apc (MIT; the project already reviewed it in docs/MODEL_LICENSES.md) — [HF API author=misya11p](https://huggingface.co/api/models?author=misya11p)
- **Etude is not on HF**: author `Xiugapurin` has only 2023 NLP models — [HF API author=Xiugapurin](https://huggingface.co/api/models?author=Xiugapurin)
- Related, but not covers: `cstr/piano-transcription-GGUF` (380 downloads, apache-2.0, 2026-08-02); `tjpurdy/Piano-Separation-Model-small` (292 downloads, cc-by-nc-4.0); `yhj137/pianist-transformer-rendering` (678 downloads, apache-2.0, expressive MIDI rendering); `SyMuPe/PianoFlow-base` (932 downloads, cc-by-nc-sa-4.0) — [HF API search "piano"](https://huggingface.co/api/models?search=piano&sort=downloads)
- Searches for "piano arrangement", "pianocover", "audio2piano", "song2piano" and "amt-apc" returned **0 models** — [HF API](https://huggingface.co/api/models?search=piano%20arrangement)

**Spaces**
- **sweetcocoa/pop2piano** (official demo): 124 likes, lastModified 2025-12-11, runtime stage **BUILD_ERROR** (requested t4-small). It does not work at the moment — [HF API](https://huggingface.co/api/spaces/sweetcocoa/pop2piano)
- **StrikeKhan/ai-piano-cover** ("AI Piano Cover Studio v5.0"): created 2026-10-25, modified 2026-11-01, **RUNNING** on zero-a10g, 1 like, license `mit` (covers the Space code only). It compares two engines side by side: (1) HTDemucs + ByteDance High-Resolution Piano Transcription + FluidSynth, and (2) Pop2Piano — [HF API](https://huggingface.co/api/spaces/StrikeKhan/ai-piano-cover), [Space](https://huggingface.co/spaces/StrikeKhan/ai-piano-cover)
- **teamup-tech/PiCoGen** ("PiCoGen2"): created 2026-07-10, modified 2026-11-03, **RUNNING** on zero-a10g, license `cc-by-nc-sa-3.0`, tags `input:audio`, `output:midi`, `piano-cover`. It is a HARP deployment (it also runs as a DAW plugin through HARP) using Beat-This and PiCoGen2 weights — [HF API](https://huggingface.co/api/spaces/teamup-tech/PiCoGen), [Space](https://huggingface.co/spaces/teamup-tech/PiCoGen)
- Other Pop2Piano Spaces: KennethCaple/pop2piano (9 likes, CONFIG_ERROR), hantnam, SapSapphireFR, jacobm13, promiseX, mzr (0–1 likes each). `mzr/picogen_test` (2024-06-22) — [HF API spaces search](https://huggingface.co/api/spaces?search=pop2piano)
- Transcription Spaces (useful as a baseline, not arrangers): asigalov61/ByteDance-Solo-Piano-Audio-to-MIDI-Transcription (80 likes, 2026-04-24) and Genius-Society/piano_trans (42 likes, 2026-11-02) — [HF API spaces search](https://huggingface.co/api/spaces?search=piano%20trans)
- `mediumedium1998/piano-forte-covers-galore` is a static DeepSite page with no model — [HF](https://huggingface.co/spaces/mediumedium1998/piano-forte-covers-galore)

### Inferences
- The best ready-made option ("gotowiec") to listen to right away is **tsumugi-piano-cover-v2 with Cover Studio** run locally. It is the newest work (Oct 2026), trained on far more data than Pop2Piano (about 6.2k covers vs. about 300 h of K-pop for Pop2Piano), and it outputs both MIDI and a score. A 5070 12 GB card should be enough: about 5 GB for SheetSage plus a 127 M model. This is an estimate, not a measurement. It is non-commercial at every layer (cover weights, SheetSage2-Refined, MERT), so it can only be used for evaluation and as a quality reference, not shipped in the paid app.
- To compare Pop2Piano against simple transcription with no setup, use the StrikeKhan Space. For PiCoGen2, use the teamup-tech Space.
- Its training on Japanese and anime pop may favour J-pop. It is not clear how well it handles Polish, worship or guitar-led songs.

### Gaps
- No fine-tunes of Pop2Piano or the other models trained on more data (multilingual, worship and so on) were found on HF.
- HF has no listening scores or independent comparisons for tsumugi-v2. The data source of its 6,200 covers is not named, likely YouTube (unverified).
- What zerostratos/pop-to-piano-llama-v1 does is unknown (no card).

## Is Pop2Piano in transformers still maintained; are there newer pipelines in transformers/diffusers?

### Takeaway
Pop2Piano is still in transformers (latest release v5.23.0, 2026-11-13) and is maintained, but only mechanically: refactors across the whole library keep it alive, with no functional work. There are no newer piano-cover pipelines in transformers or diffusers.

### Cited Findings
- Recent commits touching `src/transformers/models/pop2piano`: 2026-11-16 "Remove no_grad decorator on all _init_weights (#50831)", 2026-11-13 "Rename past_key_values/cache_params to cache (#50585)", 2026-11-04, 2026-10-30, 2026-09-28, 2026-08-20 "Always tie embeddings for LongT5 and Pop2Piano (#47620)" — [GitHub API commits](https://api.github.com/repos/huggingface/transformers/commits?path=src/transformers/models/pop2piano)
- Latest transformers release: v5.23.0, published 2026-11-13 — [GitHub releases](https://github.com/huggingface/transformers/releases)
- PR #50610 "Slim down `[audio]` to torchaudio + soundfile + soxr (drop librosa and phonemizer)" was closed on 2026-11-12. This matters because the Pop2Piano processor needs librosa and essentia, which now have to be installed by hand (closed ≠ merged; the merge status was not checked) — [PR #50610](https://github.com/huggingface/transformers/pull/50610)
- The search found no other piano-cover classes in transformers or diffusers. The only related music audio-to-audio route found is the ACE-Step v1.5 piano LoRA in DiffSynth-Studio (ModelScope, see below), which is text-to-music and not a cover of a given song — [ModelScope](https://modelscope.cn/models/DiffSynth-Studio/acestep15xlsft-lora-piano)

### Inferences
- `transformers` is a convenient way to load Pop2Piano, but it does not fix the license problem: the weights have no license.

### Gaps
- Whether diffusers supports any audio-to-audio "cover" mode (for example ACE-Step's cover or repaint) with a piano prompt was not checked.

## ModelScope and other Chinese hubs

### Takeaway
ModelScope has no original piano-cover model. It has a copy of Pop2Piano with no license and an interesting MIT Core ML port of AMT-APC for on-device iOS/macOS inference, which is useful as an on-device reference.

### Cited Findings
- **huluxiaohuowa/wa-score-amt-apc-coreml**: License `mit`, created about 2026-09-24 (unix 1790226284), 0 downloads. Description: "AMT-APC piano cover models for offline Core ML inference in WA Score". Card: a "conversion of the official `apc.pth`, not a newly trained model", from AMT-APC commit 78c1afb, 5.75 M params, float32, iOS 17 / macOS 14, "No Python or network inference is required inside the app", including style vectors level1/2/3 — [ModelScope](https://modelscope.cn/models/huluxiaohuowa/wa-score-amt-apc-coreml)
- **JeffreyZhou02/pop2piano**: a copy of the sweetcocoa card, License field empty, created about 2026-06-19, 35 downloads — [ModelScope](https://modelscope.cn/models/JeffreyZhou02/pop2piano)
- **Genius-Society/piano_trans**: "钢琴转谱模型镜像" (a mirror of a piano transcription model), License `LGPL-3.0`, 1,102 downloads. Transcription, not arrangement — [ModelScope](https://modelscope.cn/models/Genius-Society/piano_trans)
- **DiffSynth-Studio/acestep15xlsft-lora-piano**: Apache License 2.0, about 2026-05-27, 90 downloads. A LoRA for ACE-Step-v1.5-XL-sft that "增强生成音乐中的钢琴" (strengthens the piano in generated music). It is text/lyrics-to-music, not a cover of a given song — [ModelScope](https://modelscope.cn/models/DiffSynth-Studio/acestep15xlsft-lora-piano)
- The search for "piano" returned 25 models in total, mostly LLMs named "piano" or MIDI tools. "amt-apc" gave 1 result (the Core ML port). Searches for 钢琴 / 钢琴改编 / 扒谱 / 翻弹 ignored the name filter and returned the general ranking, so they gave no result — [ModelScope API](https://www.modelscope.cn/api/v1/dolphin/models)

### Inferences
- The AMT-APC Core ML port shows that an MIT-licensed APC can run entirely on device (iOS). An Android analogue (TFLite/ONNX) is something we could do ourselves, but the project rejected AMT-APC on quality.

### Gaps
- No other Chinese hubs (WiseModel, OpenXLab, Gitee AI) were checked. The ModelScope search for Chinese keywords did not work through the API, so there may be models with Chinese-only names (unverified).

## Replicate / fal.ai / other inference marketplaces

### Takeaway
The only hosted piano-cover API found is Replicate's community wrapper `m1guelpf/pop2piano`, which costs a few cents per run, has no license and has no commercial-use terms for the output. fal.ai has no piano-cover model.

### Cited Findings
- **replicate.com/m1guelpf/pop2piano**: A100 80GB, "about $0.050 per run, or 20 runs per $1" with typical time about 36 s per the page. A search-engine snippet gave $0.035 per run and 279 runs, so the price changes over time. Neither the page nor the snippet gives a license or commercial terms. The README sections are empty placeholders. "Model created over 1 year ago" — [Replicate](https://replicate.com/m1guelpf/pop2piano); snippet: [Replicate API page](https://replicate.com/m1guelpf/pop2piano/api)
- fal.ai: no audio-to-piano-cover endpoint. The closest are `fal-ai/sam-audio/separate` (separating a "piano" source by text prompt) and Stable Audio 2.5 / 3 audio-to-audio (style variation from text, not a faithful cover) — [fal SAM Audio](https://fal.ai/models/fal-ai/sam-audio/separate/llms.txt), [fal Stable Audio 3 a2a](https://fal.ai/models/fal-ai/stable-audio-3/small/music/audio-to-audio/api)
- Replicate also hosts U-MusT (malerlab), which goes from a sheet-music image to piano audio and is research-only under CC BY-NC-SA 4.0 per the search summary (unverified) — [Replicate](https://replicate.com/malerlab/u-must-image-to-audio)

### Inferences
- Replicate's Pop2Piano is fine for a quick test without a GPU, but it adds nothing over running the model locally on the 3060 Ti / 5070, and it inherits the license problem of the weights.

### Gaps
- Replicate's search page could not be fetched (empty response), so there may be other community piano-cover models (for example AMT-APC or PiCoGen wrappers) that were not found. Civitai-style music hubs: nothing was found or checked for piano covers.

## Datasets of song + piano cover pairs on the hubs

### Takeaway
No public song-and-piano-cover pair dataset exists on HF or ModelScope. Only small test sets and related collections exist, mostly with NC licenses.

### Cited Findings
- `sweetcocoa/pop2piano_ci`: mit, n<1K, 71 downloads, 2023-06-19. A few CI examples, not a training set — [HF](https://huggingface.co/datasets/sweetcocoa/pop2piano_ci)
- `audio-tau/pop909-with-songs`: mit tag, 100K<n<1M, gated "auto", 2025-04-16. POP909 piano arrangements with the songs. The song audio itself is copyrighted, so the MIT tag does not cover the audio (inference) — [HF](https://huggingface.co/datasets/audio-tau/pop909-with-songs)
- `Hayeonbang/PIAST`: mit, 361 downloads, 2024-11-14. Piano music with text descriptions, not cover pairs — [HF](https://huggingface.co/datasets/Hayeonbang/PIAST)
- `SyMuPe/PianoCoRe` (cc-by-nc-sa-4.0, 1,024 downloads), `SKY-Piano/SKY-Piano` (cc-by-nc-nd-4.0, gated manual, updated 2026-11-18), `PianoVAM/PianoVAM_v1` (cc-by-nc-4.0, 4,869 downloads): piano performance data, not covers — [HF API datasets search "piano"](https://huggingface.co/api/datasets?search=piano&sort=downloads)
- Searches for "piano-cover", "piano cover", "pianocover", "picogen", "piano_cover" and "cover_song" returned **0** relevant datasets — [HF API](https://huggingface.co/api/datasets?search=piano%20cover)

### Inferences
- The 6,200-cover set behind tsumugi-v2 and the Pop2Piano set (YouTube) are not published, so we still have to build our own licensed pair set.

### Gaps
- No ModelScope datasets were searched.
