# Audio route to piano covers: generative audio model -> solo-piano audio -> piano AMT -> MIDI (state as of 2026-11-18)

Research scope: Can ready-made audio generators (Suno, Udio, ACE-Step, Stable Audio, MusicGen, ElevenLabs, Lyria, timbre transfer) turn an arbitrary song into a faithful solo-piano rendition, how well does piano transcription then recover MIDI, is it commercially usable, does it run on an RTX 5070 12 GB, and has anyone published the two-step pipeline? About 15 tool calls; many sources are vendor or blog pages, and these are marked.

## 1. Which generators can take a reference song and output solo piano that follows its melody and harmony? Evidence of faithfulness

### Takeaway
Several systems offer a "cover" or restyle mode that keeps the melody skeleton and swaps the instrumentation: Suno Covers, ACE-Step 1.5 Cover, Stable Audio 2.5/3 inpainting, MusicGen-Melody, MusicGen-Style and JASCO. None publishes a quantitative evaluation of melody or harmony faithfulness for instrument-to-solo-piano conversion. The best published melody-control figure for open audio diffusion is about 61% melody accuracy (MuseControlLite, ICML 2025). That is far below what a usable note-level piano cover needs. Faithfulness is a soft, stochastic constraint, not a guarantee.

### Cited Findings
**Suno (commercial SaaS)**
- Suno's Covers feature (launched Sept 2024) "keeps the original melody" and changes the style, arrangement and instrumentation. Guides describe using it to swap instrumentation, such as guitar to piano. — [AlternativeTo news, 2024-09](https://alternativeto.net/news/2024/9/suno-ai-s-new-covers-feature-lets-users-reimagine-music-in-any-different-style); [bmushsoftmusic guide](https://bmushsoftmusic.com/suno-cover/) (third-party)
- A third-party guide says the Covers feature is intended for covering Suno-generated tracks or the user's own uploads, not commercial recordings. — [Jack Righteous rights guide, 2026](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/can-you-monetize-suno-remix-cover-extend-rights) (third-party; unverified against Suno's own help docs)
- Suno Studio 2.0 (Premier tier) reportedly lets you drag audio onto a MIDI track to transcribe it, and MIDI clips can drive an audio cover generation. Suno Studio can also convert single stems to MIDI. — [suno.skin blog, 2026](https://suno.skin/en/blog/suno-studio-2-guide-2026/); [Songscription blog](https://www.songscription.ai/blog/i-made-a-song-in-suno-now-what) (third-party; unverified)
- I found no published metric (melody accuracy, chord accuracy) for Suno Covers' faithfulness to the source.

**Udio**
- After the UMG settlement (2025-10-29), Udio disabled downloads. A new licensed "walled garden" platform was planned for 2026. — [RouteNote](https://routenote.com/blog/udio-stops-user-downloads-after-umg-deal-heres-why/); [Digital Music News, 2025-11-20](https://www.digitalmusicnews.com/2025/11/20/will-udio-still-exist-in-2026/); [WebProNews](https://www.webpronews.com/udio-halts-ai-song-downloads-after-copyright-settlement-with-umg-warner)
- I could not confirm whether the 2026 platform has launched or what its terms are. As of the sources found, Udio output cannot be exported, so it cannot feed a transcription pipeline.

**ACE-Step 1.5 (open weights)**
- ACE-Step 1.5 (arXiv 2602.00744, Jan/Feb 2026) supports Cover and Repaint (DiT turbo/SFT variants), reference audio, and "vocal-to-BGM conversion". Extract, lego and complete are base-model only. — [HF model card](https://huggingface.co/ACE-Step/Ace-Step1.5); [arXiv 2602.00744](https://arxiv.org/abs/2602.00744)
- How cover works: source audio is encoded into 5 Hz FSQ-quantized "Source Latent" codes (codebook about 64k). Cover is described as "re-synthesizing timbre while retaining melodic skeletons via quantized latents". The paper claims "high-fidelity preservation of musical elements like melody and rhythm" only qualitatively. **No quantitative cover/repaint faithfulness evaluation is reported.** — [arXiv 2602.00744v2 HTML](https://arxiv.org/html/2602.00744v2)
- In ComfyUI, the AceStep 1.5 Cover node exposes `melody_strength` (default 0.5), which "dictates to what extent the original melody should resemble the reference". Higher strength behaves as a cover and lower strength as a restyle. — [ComfyAI docs](https://comfyai.run/documentation/AceStep15CoverConditioning); [okmiku blog](https://okmiku.com/blog/ai-cover-vs-remix) (blog)
- Instrumental outputs can come back with humming, syllables or a buried vocal line, so prompts need to name the instruments explicitly. — [okmiku blog](https://okmiku.com/blog/ace-step-instrumental-prompts) (blog)
- A LoRA can be trained "from a few songs" in about 1 hour on an RTX 3090. A solo-piano-style LoRA is therefore plausible, but I found none published for this purpose. — [ACE-Step 1.5 docs](https://ace-step.github.io/ACE-Step-1.5/en/)

**Meta AudioCraft (MusicGen-Melody, MusicGen-Style, JASCO)**
- The AudioCraft repo lists MusicGen, MAGNeT, **MusicGen Style** (text-and-style-to-music) and **JASCO** (text-to-music conditioned on chords, melodies and drum tracks). — [GitHub facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- MusicGen-Melody conditions on a chromagram of the reference (Copet et al. 2023). It keeps a rough melodic contour, but the chroma is folded to one octave and computed after Demucs drum removal. — [AudioCraft README FAQ mentions musicgen-melody relies on Demucs](https://github.com/facebookresearch/audiocraft) (the chroma details come from the MusicGen paper, which I did not re-fetch this session, so treat them as unverified)

**Research melody-control (audio diffusion)**
- MuseControlLite (Tsai, Wu, ..., Yi-Hsuan Yang; ICML 2025; arXiv 2506.18729) adds melody, rhythm and dynamics control to Stable Audio Open. Melody accuracy is **61.1%** with RoPE (56.6% without). It reports improved controllability over Stable Audio Open ControlNet and compares against MusicGen-Melody. Code and checkpoints are at musecontrollite.github.io/web/. — [arXiv 2506.18729](https://arxiv.org/abs/2506.18729); [PDF](https://arxiv.org/pdf/2506.18729)
- Cover-song generation research in 2026 includes SongEcho (arXiv 2602.19976) and MPEcho, a "melody and phoneme-aware generative framework for controllable cover song generation" (arXiv 2607.26698). Both target vocal covers, not piano. — [SongEcho](https://arxiv.org/html/2602.19976); [MPEcho](https://www.alphaxiv.org/abs/2607.26698v1) (abstracts only, not read in depth)

**Timbre transfer to piano**
- "Research on Piano Timbre Transformation System Based on Diffusion Model" (Hsu et al., arXiv 2601.09333, 2026-01-14) converts performances on other instruments to piano timbre. A diffusion decoder is conditioned on pitch-encoder and loudness-encoder features. Evaluation is "case analysis" of pitch accuracy and timbral similarity across classical, jazz and pop, with no numeric metrics in the abstract and no code linked. — [arXiv 2601.09333](https://arxiv.org/abs/2601.09333)
- DDSP and RAVE: I did not find new sources this session. From prior knowledge (unverified here): DDSP timbre transfer is monophonic (f0 plus loudness), and RAVE is single-domain, so neither handles a full polyphonic pop mix to solo piano.

**Stable Audio**
- Stable Audio 2.5 (Sept 2025) supports audio inpainting: "users can input their own audio, select where they want it to start". It is available via StableAudio.com, the API, partners, or an on-prem enterprise license. Stability says it is trained on a fully licensed dataset. — [Stability AI news](https://stability.ai/news/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale)
- Stable Audio 3 (technical report May 2026) is reported as a family of small, medium and large latent diffusion models for generation and editing (inpainting). Small and medium weights are on HF, and Large is under an enterprise license. — [blog summary](https://thedailypac.blogspot.com/2026/05/stability-ai-releases-stable-audio-3.html) (secondary; unverified against Stability's primary page)
- Inpainting or continuation is not the same as melody-preserving instrument conversion. I found no evidence that Stable Audio produces a faithful piano rendition of a reference song.

**ElevenLabs Music**
- A third-party guide describes Eleven Music v2 "Audio Reference" as about 30 s of uploaded audio that guides mood, production style, instrumentation and tempo, plus improved inpainting. These are style cues, not melody-preserving conversion. — [Jack Righteous ElevenMusic guide 2026](https://jackrighteous.com/blogs/music-creation-process-guide/elevenmusic-guide-2026-features-royalties-rights-and-plans) (third-party)

**Symbolic-domain competitors (context)**
- Melody-conditioned piano cover research mostly works in the symbolic domain: PiCoGen (arXiv 2407.20883) uses audio -> lead sheet -> symbolic piano, and Etude (arXiv 2509.16522) uses a three-stage extract/structuralize/decode approach. — [PiCoGen](https://arxiv.org/html/2407.20883v1); [Etude](https://www.researchgate.net/publication/395724360_Etude_Piano_Cover_Generation_with_a_Three-Stage_Approach_--_Extract_strucTUralize_and_DEcode)
- "Learning Music Style for Piano Arrangement Through Cross-Modal Bootstrapping" (arXiv 2608.03050, 2026-08-04) uses frozen MusicGen-Large only as an audio feature extractor (via a Q-Former), feeding MuseCoco to generate a symbolic piano arrangement. Against PiCoGen2 on POP909: MCA/CA 32.0/33.3 vs 39.3/40.0, so PiCoGen2 preserves content better, while the new model wins on style metrics. It does not discuss generating audio covers. — [arXiv 2608.03050v1](https://arxiv.org/html/2608.03050v1)

### Inferences
- The only ready-made systems with a credible "same song, now on piano" mode are **Suno Covers** (closed) and **ACE-Step 1.5 Cover** (open, MIT). Both preserve a "melodic skeleton" from low-rate latents (ACE-Step's codes run at 5 Hz, i.e. 200 ms). That rate is coarser than note-level detail, so fast passages, exact voicings and bass lines are likely to be reinterpreted rather than copied. This follows from the 5 Hz design and is not measured.
- Even MuseControlLite, an explicit melody-control model, reaches only about 61% melody accuracy. Expect the melody to drift, especially at low cover strength. High strength preserves more but may leak non-piano timbres from the source mix.
- The generator's output is a re-imagined arrangement, not a transcription. Harmony errors in generation compound with AMT errors downstream.

### Gaps
- No public objective evaluation (melody accuracy, chord accuracy) of Suno Covers or ACE-Step Cover for song-to-solo-piano was found. A controlled test would be needed, e.g. run 20 pop songs, transcribe, and compare to a reference with chord/melody metrics.
- Google Lyria 2 / Lyria RealTime / MusicFX DJ: not researched in depth this session. To my knowledge (unverified) they are text- and prompt-steered with no melody-preserving audio-to-audio cover mode.
- I found no dedicated commercial "piano cover audio generator" with published faithfulness. The vendor pages found (e.g. MiniMax Music Cover, [okmiku review](https://okmiku.com/blog/minimax-music-cover-review)) were not verified.
- Sony and ByteDance MusicControlNet-style models: no 2026 piano-specific results found.

## 2. How good is the subsequent piano transcription on generated or synthetic piano audio?

### Takeaway
No study benchmarks Transkun, hFT-Transformer, YourMT3+, Kong or Aria-AMT on AI-generated piano audio. Evidence on out-of-distribution robustness suggests a clear accuracy drop on non-MAESTRO acoustics, with Transkun the most robust under noise and reverb. Neural-codec and diffusion artifacts (smeared onsets, inconsistent pedal or timbre) are an untested risk.

### Cited Findings
- Edwards et al. (arXiv 2402.01424, 2024) show that strong piano AMT models overfit to the acoustics of their training data, tested on a re-performed Disklavier MAESTRO. Pitch-shift and reverb augmentation consistently improves OOD performance. — [arXiv 2402.01424](https://arxiv.org/abs/2402.01424)
- A 2026 study (arXiv 2607.28876) tested Transkun, Kong, Edwards and hFT on ASAP audio with added noise and reverb. "Transkun comes out as the most robust model, topping its competitors in almost all measures". Velocity estimation is especially sensitive to recording conditions. — [arXiv 2607.28876](https://arxiv.org/pdf/2607.28876)
- A MAESTRO-vs-Disklavier evaluation found statistically significant performance differences for all three models analysed. — [arXiv 2406.08454](https://arxiv.org/pdf/2406.08454)
- Phase-based onset enhancement (arXiv 2610.29222, 2026) compares Transkun, Kong and YourMT3+ on piano-in-ensemble audio and finds that environmental-sound augmentation improves OOD robustness. It also notes that systems like YourMT3+ "are only trained on synthetic recordings of pop music, and perform poorly on commercially-produced songs". — [arXiv 2610.29222](https://arxiv.org/html/2610.29222v1)
- Aria-AMT ("Musically aware automatic piano transcription using synthetic pretraining") uses synthetic pretraining for robustness. — [Spangher et al. PDF](https://www.alexander-spangher.com/papers/aria_amt.pdf) (not read in full this session)
- Practitioner guides for Suno -> MIDI say: use a piano-specific transcriber, watch sustain-pedal handling (the main failure point), and expect a rough draft that needs manual cleanup. — [Melumimi](https://melumimi.com/en/suno-to-midi); [NotationAI](https://notationai.app/guides/suno-to-sheet-music); [Rebel Music Distribution, 2026-08-11](https://rebelmusicdistribution.com/2026/08/11/ai-piano-to-midi-studio-quality-piano/) (vendor blogs)

### Inferences
- A clean solo-piano generation is acoustically closer to AMT training data than a full mix, so AMT on it should do better than AMT on the original song. But generated "piano" is often not a physically consistent piano: it can contain pad or string bleed, smeared transients and pitch-unstable tails. Expect onset F1 below the MAESTRO-level figures of about 0.97 or more. The size of the drop is unknown.
- Transkun (most robust in the 2026 noise/reverb study) and Aria-AMT (synthetic pretraining) are the best candidates for this input. hFT and Kong are more MAESTRO-specific.
- Velocity and pedal from generated audio will be unreliable (velocity sensitivity is documented). A downstream MIDI cleanup or quantization step would be needed.

### Gaps
- No paper or benchmark measures AMT F1 on Suno, ACE-Step or MusicGen generated piano. This is a clear gap that could be measured locally, e.g. by generating piano from known MIDI renders and comparing.
- Licenses of the AMT models (Transkun, YourMT3+, Aria-AMT, hFT) were not re-verified in this sub-task.

## 3. Licensing: commercial use of generated audio and derived MIDI; uploading copyrighted songs as reference

### Takeaway
Commercial SaaS terms all require that you own or have rights to any uploaded audio. Uploading a copyrighted commercial song to make a "cover" breaches Suno's warranty, and the output would still be a derivative of the composition. The cleanest legal profile is ACE-Step 1.5 (MIT code and weights; the card says outputs may be used commercially), but the composition-copyright problem of covering someone else's song remains whatever tool is used. MusicGen weights are CC-BY-NC 4.0, which is non-commercial.

### Cited Findings
- **Suno ToS** (Date of Last Revision 2026-08-10, Effective 2026-09-03):
  - Pro and Premier: "Suno hereby assigns to you all of its right, title and interest in and to any Output owned by Suno", subject to commercial-use restrictions. "Suno makes no representation or warranty to you that any copyright will vest in any Output."
  - Free and Basic: "lawful, personal and non-commercial purposes" only.
  - Commercial use requires a permitted Download, and "Obtaining a copy of an Output by any means other than a download channel made available by Suno is prohibited."
  - Uploads: the user warrants they "have, or have obtained, all rights, licenses, consents, permissions", and uploading material that "infringes, violates or misappropriates any intellectual property" is prohibited. Remixes of others' output are "personal and non-commercial" only.
  — [suno.com/terms](https://suno.com/terms)
- A third-party guide argues that a Suno cover does not grant the composition (mechanical) license needed to release a cover of someone else's song commercially. — [Jack Righteous](https://jackrighteous.com/en-us/blogs/guides-using-suno-ai-music-creation/can-you-monetize-suno-remix-cover-extend-rights) (opinion)
- **Udio**: post-settlement terms removed downloads. Pre-settlement downloads may keep their original rights, but this is disputed. — [RouteNote](https://routenote.com/blog/udio-stops-user-downloads-after-umg-deal-heres-why/); [Digital Music News 2025-11-03](https://www.digitalmusicnews.com/2025/11/03/umg-udio-vexing-questions-swirl/)
- **ElevenLabs Music Terms**:
  - Output use follows the "Media Rights for your plan" (Exhibit A / Music Commercial Rights Table). Some plans allow all online and offline commercial use except film, TV, radio and studio games.
  - "Customer may not sell, resell, ... license, sublicense, or otherwise commercially exploit any Output" standalone, and may not build a commercial music library.
  - Under the archived v1 terms (to 2025-11-21), the free plan was non-commercial only.
  — [elevenlabs.io/music-terms](https://elevenlabs.io/music-terms); [model-specific terms](https://elevenlabs.io/eleven-music-model-specific-terms); [archived v1 terms](https://elevenlabs.io/archived-eleven-music-v1-terms)
- **ACE-Step 1.5**: License **MIT** (Copyright 2025-present ACE-Step Team). The model card says "You can strictly use the generated music for commercial purposes", and training data is described as licensed, royalty-free or public domain, and synthetic. Neither the card nor the paper addresses copyrighted reference audio. — [HF card](https://huggingface.co/ACE-Step/Ace-Step1.5); [docs](https://ace-step.github.io/ACE-Step-1.5/en/); [arXiv 2602.00744v2](https://arxiv.org/html/2602.00744v2)
- **Meta AudioCraft**: code under the MIT license, and "The models weights in this repository are released under the CC-BY-NC 4.0 license" (LICENSE_weights). This covers MusicGen (incl. Melody and Style), JASCO and MAGNeT. — [GitHub audiocraft](https://github.com/facebookresearch/audiocraft)
- **Stable Audio Open**: Stability AI Community License, which allows non-commercial use and commercial use for individuals and organizations with up to $1M annual revenue. Trained on CC0, CC-BY and CC-Sampling+ audio (Freesound and FMA, 7,300 h). — [Stability AI news](https://stability.ai/news/stable-audio-open-research-paper); [arXiv 2407.14358](https://arxiv.org/html/2407.14358v2)
- **Stable Audio 3**: licensing is conflicting. One secondary source says the open models use the Community License, a GitHub mirror says MIT, and a wiki claims CreativeML Open RAIL-M. **Unverified.** — [blog](https://thedailypac.blogspot.com/2026/05/stability-ai-releases-stable-audio-3.html); [ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fstability-ai%2Fstable-audio-3)

### Inferences
- For a product that ships piano MIDI of copyrighted songs, the generator's license is a secondary issue. The primary issue is the composition copyright, which needs a mechanical or print/arrangement license, and that applies equally to Pop2Piano/AMT-APC MIDI.
- The audio route adds an extra layer. With SaaS tools (Suno, ElevenLabs), uploading a copyrighted commercial recording breaches the terms' rights warranty, so it is unusable for a product built on arbitrary songs. With local ACE-Step (MIT), no ToS bars it, but the output is still a derivative of the input recording and composition.
- MusicGen-Melody and MusicGen-Style weights (CC-BY-NC 4.0) are research-only for this project.
- Derived MIDI inherits whatever restrictions apply to the audio. The Suno ToS's "no representation... that any copyright will vest" also means you may not own protectable rights in AI-generated parts.

### Gaps
- I did not fetch Udio's current 2026 terms or Google Lyria / Vertex AI terms.
- There is no case law or official statement specifically on audio-reference uploads used to create covers. Ongoing litigation (labels vs Suno) was not re-checked this session.

## 4. Local feasibility on an RTX 5070 12 GB and generation time per song

### Takeaway
ACE-Step 1.5 is clearly feasible locally. Official figures are under 4 GB VRAM for the standard models and under 10 s per full song on an RTX 3090. The 4B "XL" LM variants want 12 GB or more with offloading. Add about 10-60 s for piano AMT per song, which is an estimate. Suno, ElevenLabs and Stable Audio 2.5 are cloud-only, and MusicGen is local but non-commercial.

### Cited Findings
- ACE-Step 1.5: "Runs locally with less than 4GB of VRAM". "Generates a full song in under 2 seconds on an A100 and under 10 seconds on an RTX 3090". Lengths run from 10 s to 10 min. LoRA training takes about 1 h on an RTX 3090. — [HF card](https://huggingface.co/ACE-Step/Ace-Step1.5); [docs](https://ace-step.github.io/ACE-Step-1.5/en/)
- Third-party sources report higher practical requirements: 12 GB or more with offload for the 4B XL models, CPU offload auto-enabled in the HF Gradio UI below 16 GB, and 6-8 GB recommended for "2B turbo + 0.6B LM". — [Clore docs](https://docs.clore.ai/guides/music-generation/ace-step); [HF Space README](https://huggingface.co/spaces/ACE-Step/Ace-Step-v1.5/blob/main/README.md); [Boppy blog](https://boppy.me/blog/best-open-source-ai-music-models) (conflicting; the official "<4GB" applies to the smaller configs)
- Stable Audio 3 small and medium have open weights on HF. — [secondary blog](https://thedailypac.blogspot.com/2026/05/stability-ai-releases-stable-audio-3.html) (VRAM not found)

### Inferences
- On an RTX 5070 (12 GB, Blackwell), the ACE-Step 1.5 turbo plus 0.6B/1.7B LM should fit without offload, at an estimated roughly 5-15 s per 3-4 min song. This is not measured, and Blackwell needs a PyTorch build with sm_120 support. Several cover takes per song (sampling for the best melody match) are cheap.
- The pipeline cost per song is dominated by quality control, not compute. Generation might need N samples plus automatic scoring, e.g. chroma or melody similarity between the source and the generated piano.

### Gaps
- No measured ACE-Step Cover-mode VRAM or time on a 12 GB Blackwell card. This should be benchmarked locally (the project already has a benchmarking runbook).

## 5. Has anyone published or demonstrated the two-step pipeline (generate piano audio -> transcribe) for piano covers, and how does it compare to Pop2Piano / AMT-APC?

### Takeaway
I found no academic paper that evaluates "audio generator -> solo-piano audio -> AMT -> MIDI" as a piano-cover method. It exists only as informal practitioner and vendor workflows (Suno -> piano-to-MIDI tool, and Suno Studio's built-in audio-to-MIDI). Research piano-cover systems (Pop2Piano, AMT-APC, PiCoGen/PiCoGen2, Etude, the 2026 cross-modal bootstrapping model) all output MIDI directly, symbolically.

### Cited Findings
- The practitioner workflow is: generate or export piano in Suno, transcribe with a piano-focused audio-to-MIDI tool, then clean up timing, durations, velocity and pedal manually. Authors call the auto-MIDI "a rough draft". — [Rebel Music Distribution, 2026-08-11](https://rebelmusicdistribution.com/2026/08/11/ai-piano-to-midi-studio-quality-piano/); [Songscription](https://www.songscription.ai/blog/i-made-a-song-in-suno-now-what); [NotationAI](https://notationai.app/guides/suno-to-sheet-music); [latouchemusicale](https://latouchemusicale.com/en/tools/suno-to-sheet-music-midi/) (vendor blogs; these concern Suno originals, not covers of external songs)
- Suno Studio 2.0 integrates audio-to-MIDI and MIDI-driven cover generation (Premier). — [suno.skin](https://suno.skin/en/blog/suno-studio-2-guide-2026/) (unofficial; unverified)
- The 2026 cross-modal bootstrapping paper (arXiv 2608.03050) compares against PiCoGen2 and an Audio-to-MIDI baseline, but does not consider generating audio and then transcribing it. — [arXiv 2608.03050v1](https://arxiv.org/html/2608.03050v1)
- The piano timbre-transformation diffusion system (arXiv 2601.09333) is the closest "audio-domain piano conversion" research. It has no transcription stage and no released code. — [arXiv 2601.09333](https://arxiv.org/abs/2601.09333)

### Inferences
- Compared with Pop2Piano and AMT-APC, which go directly from audio to piano MIDI (AMT-APC fine-tunes an AMT model, hFT-based, to output cover MIDI):
  - **Pros of the audio route:** it can draw on large commercial or open generators' musicality (richer arrangements, natural pianistic textures), and a single generator can be steered by prompt or LoRA toward a style.
  - **Cons:**
    - Two lossy stages compound errors (generation drift plus AMT errors on OOD audio).
    - Melody faithfulness is stochastic and unmeasured, with a 5 Hz semantic bottleneck in ACE-Step.
    - The MIDI is not guaranteed playable or pianistic after AMT.
    - Velocity and pedal are noisy.
    - It is non-deterministic.
    - SaaS terms forbid uploading copyrighted songs.
    - It is slower and harder to evaluate.
  - Pop2Piano and AMT-APC are trained explicitly on (song, piano cover) pairs, so their alignment to the source is structural.
- The audio route looks better suited as (a) a data-augmentation source, e.g. generating extra piano covers of songs you have rights to, transcribing them, and adding the pairs to training data with careful filtering, or (b) a style-exploration prototype. It does not look like a direct replacement for AMT-APC in a commercial product. This is an inference, untested.
- A cheap local experiment would be: ACE-Step 1.5 Cover (high melody_strength, "solo piano" prompt, optionally a piano LoRA), then Transkun or Aria-AMT, then comparison against AMT-APC output on the same songs, using melody chroma accuracy and chord accuracy against the source.

### Gaps
- No published side-by-side comparison of the audio route against Pop2Piano or AMT-APC exists, to my search.
- No user reports with quantitative fidelity for "Suno cover -> solo piano" of popular songs were found. Reddit and forum threads were not surfaced by search.
