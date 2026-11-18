# Music foundation models and audio-language models for audio to symbolic piano covers (state as of 2026-11-18)

Legend: **[V]** means verified this session against the cited page. **[UNVERIFIED]** means it comes from prior knowledge or a secondary aggregator and was not re-checked. Treat those items as leads to confirm.

## Q1. Which models take audio in and emit symbolic music, and how good are they at transcription or arrangement?

### Takeaway
No general audio-language "foundation model" (Qwen-Omni, MuMu-LLaMA, LLark, M2UGen and similar) reliably emits symbolic music from audio. The ones that do take audio in and put symbolic out are specialised seq2seq transcription or arrangement models. The strongest new one is Kyutai/Mirelo **MuScriptor** (Jul 2026), a multi-instrument transcriber. The newest piano-arrangement paper (Aug 2026, "BOSSA") wires frozen foundation models (MusicGen + MuseCoco) together with a Q-Former. Its outputs are symbolic, but it needs a lead sheet as input.

### Cited findings

**Dedicated audio-to-symbolic models (real candidates)**
- **MuScriptor** (Kyutai + Mirelo; arXiv 2607.08168, v1 9 Jul 2026, ISMIR 2026 camera-ready) [V]:
  - Architecture: a decoder-only Transformer. It reads a mel spectrogram of a 5 s excerpt plus an optional list of instruments, and writes MT3-style MIDI tokens covering 36 instrument groups.
  - Training: pre-training on about 1.45M synthetic MIDIs, fine-tuning on about 170k real recordings (over 11,000 h), then GRPO-like RL post-training on 300 curated tracks.
  - Results for the 1.3B model: onset F1 60.4 and multi F1 48.2 on their test set. Dagstuhl ChoirSet frame F1 rose from 51.0 for YourMT3+ to 80.7. No MAESTRO or piano results are reported. — [HF paper page](https://huggingface.co/papers/2607.08168.md); [arXiv](https://arxiv.org/abs/2607.08168)
  - Another source reports multi F1 48.2 versus 21.9 for YourMT3+ on their test set. [UNVERIFIED secondary] — [search summary / alphaxiv](https://www.alphaxiv.org/abs/2607.08168.md)
  - Limitation: the tokenization cannot represent two overlapping notes of the same pitch on the same instrument. — [HF paper page](https://huggingface.co/papers/2607.08168.md)
- **YourMT3+** (arXiv 2407.04822, 2024) is the multi-instrument MT3 successor. The paper promised to release weights and code; the repo is at github.com/mimbres/YourMT3. — [arXiv v1](https://arxiv.org/html/2407.04822v1). License not verified this session [UNVERIFIED].
- **Pop2Piano** (2022) is a T5 model that maps pop audio spectrograms directly to piano MIDI. It was trained on 300 h of synchronized pop/piano-cover pairs and uses arranger tokens. — [arXiv](https://arxiv.org/html/2211.00895v2)
- **AMT-APC** (arXiv 2409.14086, revised 2026) fine-tunes an hFT-Transformer piano AMT model for piano cover generation. — [arXiv](https://arxiv.org/pdf/2409.14086)
- **Etude** (Chen & Joung; arXiv 2509.16522, v2 23 Jan 2026) is a 3-stage system: Extract (beats/downbeats), strucTUralize, DEcode with simplified REMI. The authors claim it "substantially outperforms prior models" in subjective tests and is "comparable to human composers". The abstract page shows no code or weights. — [arXiv](https://arxiv.org/abs/2509.16522)
- **PiCoGen / PiCoGen2** (2024): PiCoGen2 is a Transformer LM on Sheetsage (Jukebox-derived) hidden states. — [PiCoGen2](https://arxiv.org/pdf/2408.01551), [PiCoGen](https://arxiv.org/pdf/2407.20883)
- **"Learning Music Style for Piano Arrangement Through Cross-Modal Bootstrapping"** (arXiv 2608.03050, 4 Aug 2026; demo zhaojw1998.github.io/bossa) [V]:
  - Inputs and output: takes a lead-sheet MIDI plus reference audio (for style) and outputs a symbolic piano arrangement.
  - Architecture: a Q-Former initialised from MusicBERT-Base (about 186M) pulls style from frozen MusicGen-Large layer-25 states (about 1.7B). A frozen MuseCoco-xLarge decoder (about 1.2B) with a rank-16 LoRA does the writing.
  - Data: POP909 + PIAST (about 8k piano recordings).
  - Results on POP909: PiCoGen2 is better on melody chroma accuracy (39.3 vs 32.0) and chord accuracy (40.0 vs 33.3). The new model is better on groove (79.2 vs 78.4), velocity (76.6 vs 73.2) and tempo (83.6 vs 74.4), and is significantly preferred in a 21-listener test.
  - It does **not** compare against AMT-APC, Pop2Piano or Etude. No code or weights license is stated. — [arXiv HTML](https://arxiv.org/html/2608.03050v1)
- **SheetSage-A2S** (Cummins et al., arXiv 2608.06165, ACM MM '26) does audio-to-score transcription for popular music using pretrained features and a new dataset. Details are [UNVERIFIED]; it surfaced via search only. — [alphaxiv](https://www.alphaxiv.org/abs/2608.06165)
- **Sori** (ISMIR 2025 LBD) does real-time general audio-to-MIDI with a controllable trade-off between accordance and musicality. — [ISMIR LBD](https://ismir2025program.ismir.net/lbd_444.html)
- **VocalParse** (arXiv 2605.04613) extends the Qwen3-ASR vocabulary with 128 MIDI-pitch tokens for singing transcription, which makes it an LALM that emits notes. It covers vocals only. — [takara summary](https://tldr.takara.ai/p/2605.04613), [HF](https://huggingface.co/ngoctham/VocalParse)

**General audio-language models: audio in, text out, no reliable symbolic output**
- **CMI-Bench** (Ma et al., Jun 2025) tested Qwen2-Audio, Qwen-Audio, SALMONN, MU-LLaMA, MusiLingo, LTU, Audio-Flamingo, GAMA and Pengi. It found "significant performance gaps between LLMs and supervised models". Sequential, structured tasks such as melody extraction and beat tracking "remain challenging". — [arXiv](https://arxiv.org/pdf/2506.12285)
- A third-party summary says the best LLM scored about 8% on key detection versus more than 74% for MIR baselines. [UNVERIFIED secondary] — [emergentmind](https://www.emergentmind.com/topics/cmi-bench)
- A 2025 NeurIPS-workshop evaluation (Carone et al.) found LLMs near ceiling on MIDI input but with accuracy drops on audio input. [secondary search snippet] — [arXiv 2510.22455](https://arxiv.org/abs/2510.22455v1)
- **Qwen3-Omni**: I found no official ABC or MIDI output feature. The Captioner variant (Qwen3-Omni-30B-A3B-Captioner) writes descriptive text only. — [Qwen docs](https://docs.platform.qianwenai.com/developer-guides/text-generation/audio-understanding)
- **MuMu-LLaMA** (arXiv 2412.06660) generates music as audio through AudioLDM 2 / MusicGen. It has no MIDI path. — [arXiv](https://arxiv.org/pdf/2412.06660)
- **M2UGen** has the same audio-decoder design. — [arXiv](https://arxiv.org/pdf/2311.11255)
- [UNVERIFIED, prior knowledge] **LLark** (Spotify, 2023) is a Jukebox-encoder + Llama 2 model that outputs text and does not emit notes. **MusiLingo** and **MU-LLaMA** output text Q&A and captions. **ChatMusician** (7B Llama 2) works in text/ABC only and has no audio input. **YuE**, **SongGen**, **Mustango**, **Google Lyria/MusicFX** and Meta **AudioCraft/MusicGen** output audio, not symbolic music.

**Symbolic-only foundation models (usable as decoders, but no audio input)**
- **Seed-Music** (ByteDance, Sep 2024) uses a "lead sheet token" codec. Its symbolic features for training come from in-house MIR models. Its pipelines are Lyrics2Leadsheet and Leadsheet2Song (audio). It has no public weights or audio-to-symbolic API. — [arXiv](https://arxiv.org/html/2409.09214v1), [project](https://team.doubao.com/seed-music/lyrics-to-leadsheet)
- **MIDI-LLM** (Wu, Carlton, Miyakawa, Kim, Donahue, Huang; arXiv 2511.03942, ISMIR 2026) is Llama 3.2 1B with a vocabulary extended by MIDI tokens. It does text-to-MIDI and lead-sheet generation/infilling, not audio. — [arXiv](https://arxiv.org/abs/2511.03942)
- **Aria** (EleutherAI) is a piano-MIDI LM of about 650M–700M params (LLaMA 3.2-style), trained on about 60k h of transcribed solo piano, Apache-2.0. — [GitHub](https://github.com/EleutherAI/aria), [HF](https://huggingface.co/loubb/aria-medium)
- **aria-amt**: EleutherAI's seq2seq piano transcriber. License not confirmed. — [mirror of repo](https://backiee.wasmer.app/http_github_com/EleutherAI/aria-amt)
- **Moonbeam** (Guo & Dixon, QMUL, arXiv 2505.15559) is a MIDI foundation model at about 309M and about 839M params, trained on 81.6k h of MIDI. — [arXiv](https://arxiv.org/html/2505.15559v1)
- **NotaGen** outputs ABC, at 110M / 244M / 516M params, pre-trained on 1.6M pieces with classical fine-tuning. — [HF](https://huggingface.co/ElectricAlexis/NotaGen), [arXiv](https://arxiv.org/abs/2502.18008v3)
- **text2midi** (Bhandari et al.) conditions a decoder on an LLM text encoder. — [search result, arXiv 2412.16526](https://arxiv.org/abs/2412.16526v1)
- **MuseCoco** (Microsoft) is text-to-attribute-to-MIDI; its xLarge decoder is about 1.2B per the BOSSA paper. — [2608.03050](https://arxiv.org/html/2608.03050v1)
- **MuLaCover** (Yi et al., arXiv 2610.26684, v1 19 Oct 2026, v2 4 Nov 2026) conditions an unnamed pretrained text-to-song foundation model on a symbolic lead sheet. Its output is audio songs. The paper is CC BY-NC-SA. — [arXiv](https://arxiv.org/abs/2610.26684)

### Inferences
- There is no ready-made foundation-model piano arranger working from audio. The proven recipe is still a specialised encoder (spectrogram, MERT or Jukebox features) feeding a symbolic decoder, trained on paired data.
- MuScriptor is the most relevant new asset. It could serve as a front-end (a full-mix transcription that a symbolic piano-reduction model then rewrites), or its pretrained decoder could be fine-tuned to emit a piano cover instead of a transcription, in the same spirit as AMT-APC fine-tuning hFT-Transformer. That is speculation; nobody has published it.

### Gaps
- There are no head-to-head numbers between AMT-APC, Etude, Pop2Piano, PiCoGen2 and BOSSA on one benchmark. Each paper uses different baselines and metrics.
- I couldn't confirm whether Etude or BOSSA released code or weights.
- No piano (MAESTRO) result was found for MuScriptor.

## Q2. Open weights, licenses, sizes, and whether they fit an RTX 5070 (12 GB)

### Takeaway
The useful open models are small, roughly 0.1–1.4B. All of them fit on 12 GB for inference, and LoRA or even full fine-tuning is plausible at up to about 0.5B. The licensing catch is that MuScriptor weights are CC BY-NC 4.0 (non-commercial), while Aria is Apache-2.0.

### Cited findings
- **MuScriptor** [V]:
  - Code is MIT; weights are CC BY-NC 4.0.
  - Sizes: small 103M (14 layers, dim 768), medium 307M (default; 24 layers, dim 1024), large 1.4B (48 layers, dim 1536).
  - Weights are on HF at MuScriptor/muscriptor-{small,medium,large}.
  - No VRAM figures are given; small "suits CPU", large "really wants a GPU", and Windows needs `--torch-backend=cu128`.
  - — [GitHub](https://github.com/muscriptor/muscriptor). The PyPI package is at 0.4.0 (20 Oct 2026). — [PyPI](https://pypi.org/project/muscriptor/)
  - Mirelo runs an "improved version" in its free Studio tool, which is closed. — [Mirelo blog](https://mirelo.ai/blog/turning-audio-to-midi)
- **Aria**: Apache-2.0 for models and tooling, about 0.65–0.7B (sources differ). — [GitHub](https://github.com/EleutherAI/aria), [HF card](https://huggingface.co/loubb/aria-medium)
- **NotaGen**: up to 516M. The license is conflicting: an HF metadata PR says MIT, while an aggregator says the README states none. — [HF](https://huggingface.co/ElectricAlexis/NotaGen/discussions/1/files), [SourcePulse](https://www.sourcepulse.org/projects/1843142)
- **Moonbeam**: 309M / 839M. One aggregator lists CC BY-SA 4.0 [UNVERIFIED]. — [arXiv](https://arxiv.org/html/2505.15559v1)
- **MIDI-LLM**: 1B (Llama 3.2). The weights license was not stated on the abstract page; the Llama 3.2 Community License would presumably apply [UNVERIFIED]. — [arXiv](https://arxiv.org/abs/2511.03942)
- **MuMu-LLaMA**: no license in the README per an aggregator; it is derived from MU-LLaMA. — [SourcePulse](https://www.sourcepulse.org/projects/1842228)
- [UNVERIFIED, prior knowledge] Qwen2-Audio-7B and Qwen3-Omni-30B-A3B are Apache-2.0. Qwen3-Omni (about 30B total / 3B active MoE) does not fit 12 GB in bf16 and would need 4-bit quantization. Pop2Piano (in HF transformers) is a small T5, roughly tens of millions of params. YourMT3+ is Apache-2.0. ChatMusician is 7B.

### Inferences
- For a commercial project, MuScriptor weights cannot be used commercially without a separate license; check with Kyutai/Mirelo. Aria (Apache-2.0) is the cleanest permissively licensed large piano-symbolic decoder.
- Rough VRAM estimates (not measured):
  - MuScriptor-large at 1.4B: about 3 GB bf16 weights, so inference fits comfortably. A full fine-tune with AdamW would need about 22 GB or more, so 12 GB means LoRA or 8-bit optimizers.
  - MuScriptor-medium (307M) full fine-tuning looks feasible on 12 GB with bf16 + gradient checkpointing.
  - A 7B audio-LLM LoRA in 4-bit (QLoRA) fits but is slow, and the evidence (CMI-Bench) says these models are weak at dense temporal output.

### Gaps
- No measured VRAM or throughput for any of these on an RTX 5070.
- YourMT3+ and aria-amt weight licenses not verified.

## Q3. Would fine-tuning one of them on ~2.5k song/piano-cover pairs beat AMT-APC? Has anyone done exactly this?

### Takeaway
I found no paper that fine-tunes a large audio foundation model or audio-LLM into an audio-to-piano-cover generator, and none that beats AMT-APC on a shared benchmark. The closest analogues are AMT-APC itself (fine-tuning an AMT model), PiCoGen2 (transfer from Jukebox/Sheetsage features) and BOSSA (frozen MusicGen + MuseCoco + Q-Former/LoRA). The most plausible route to beating AMT-APC is fine-tuning a strong pretrained transcriber such as MuScriptor, not a chat-style audio LLM. That is speculation.

### Cited findings
- AMT-APC's recipe is to pre-train on AMT, then fine-tune for piano covers. — [arXiv](https://arxiv.org/pdf/2409.14086)
- PiCoGen2 uses transfer learning with weakly aligned data. — [arXiv](https://arxiv.org/pdf/2408.01551)
- BOSSA trains only a Q-Former + LoRA on frozen foundation models, using POP909 + PIAST (about 9k pieces). It beats PiCoGen2 on style metrics but loses on melody and chord accuracy. — [arXiv](https://arxiv.org/html/2608.03050v1)
- MuScriptor's own ablation shows that pre-training matters most when real data is scarce: with 1% of real data, offset F1 is 33.4 with pre-training versus 9.9 without. — [HF paper page](https://huggingface.co/papers/2607.08168.md)
- Audio-LLMs lag supervised MIR models on structured temporal outputs. — [CMI-Bench](https://arxiv.org/pdf/2506.12285)

### Inferences (speculative)
- 2.5k pairs is far more than BOSSA's style-only training needs, and it is about the size of a Pop2Piano-scale dataset, which was 300 h. Fine-tuning MuScriptor-medium or MuScriptor-large (which already "hears" full mixes and outputs MIDI tokens) to output a piano-cover token stream is the closest analogue to AMT-APC's successful recipe, with a much stronger, mix-trained starting point. Caveats:
  - The 5 s context is short for arrangement-level structure.
  - The weights are non-commercial.
  - The output vocabulary is MT3-style, so the piano program alone is easy to keep.
- A chat audio-LLM (Qwen2-Audio / Qwen3-Omni) fine-tuned with LoRA to emit MIDI tokens is unlikely to beat AMT-APC given the CMI-Bench evidence, the 12 GB constraint and the dataset size.
- A hybrid is also plausible: MuScriptor (or YourMT3+) transcription plus beat tracking, fed into a symbolic piano-arranger decoder such as Aria or Moonbeam fine-tuned on the 2.5k pairs (transcription to cover). This mirrors Etude's and PiCoGen's multi-stage designs.

### Gaps
- No published comparison of any foundation-model approach against AMT-APC.
- No public evaluation of MuScriptor fine-tuned for arrangement.

## Q4. Brand-new releases (Sep–Nov 2026)

### Takeaway
I found no major new audio-to-symbolic foundation model released in Sep–Nov 2026. Relevant items from that window are MuScriptor package 0.4.0 (20 Oct 2026) and MuLaCover (Oct/Nov 2026, symbolic in, audio out). In the months before: MuScriptor (Jul 2026), BOSSA piano-arrangement (Aug 2026), SheetSage-A2S (Aug 2026), and MIDI-LLM v2 / ISMIR 2026 (Aug 2026).

### Cited findings
- MuScriptor launched 9 Jul 2026; PyPI 0.4.0 is dated 20 Oct 2026. — [LinkedIn (Kyutai)](https://www.linkedin.com/posts/kyutai-labs_today-were-releasing-muscriptor-the-best-activity-7481305950000185344-0y__), [PyPI](https://pypi.org/project/muscriptor/), [VP Land](https://www.vp-land.com/stories/mirelo-and-kyutai-open-source-muscriptor-an-audio-to-midi-model-that-transcribes-full-mixes)
- MuLaCover: v1 19 Oct 2026, v2 4 Nov 2026. — [arXiv](https://arxiv.org/abs/2610.26684)
- BOSSA piano arrangement paper: 4 Aug 2026. — [arXiv](https://arxiv.org/html/2608.03050v1)

### Inferences
- The field's momentum is in strong open transcribers (MuScriptor) and in "foundation model + adapter" designs (BOSSA, MuLaCover), not in an end-to-end audio-LLM arranger.

### Gaps
- I didn't find November 2026 ISMIR proceedings listings or recent Tencent/NetEase or Google Lyria symbolic features. My searches surfaced nothing, but coverage of Chinese-lab releases is weak, so this is not conclusive.
- Qwen3.5-Omni or other late-2026 omni models were not checked.
