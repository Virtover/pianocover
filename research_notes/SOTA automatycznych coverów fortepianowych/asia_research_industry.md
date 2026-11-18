# East Asian research and industry on piano cover / arrangement generation (2024 to Nov 2026)

Research date: 2026-11-18. I ran about 25 searches and fetches in English, Chinese, Japanese and Korean. "UNVERIFIED" marks claims I could not check against a primary source.

## 1. What methods and products exist, what they output, quality evidence, availability and licenses

### Takeaway
There are two significant items. The first is **Yamaha Piano Sheet Converter**, a consumer app released as a beta on 2026-09-11. It turns audio into piano scores at 4 difficulty levels and exports PDF/MIDI/MusicXML. It has no API or weights, and its terms ban commercial use. The second is **Etude**, from National Taiwan University (arXiv Sep 2025, v2 Jan 2026). In an independent listening test it beat PiCoGen2, AMT-APC and Music2MIDI. Its code is MIT, but its weights are CC BY-NC-SA. For the large Chinese platforms (Tencent Music/QQ/Kugou, NetEase, ByteDance, Kuaishou, Alibaba, Baidu), I found no public piano-cover feature or model. The Chinese and Korean tools I did find are AMT/"扒谱" (transcription) services, not arrangement generators.

### Cited Findings

**Japan: Yamaha Piano Sheet Converter (β), released Sep 2026**
- Yamaha announced it on 2026-09-11 as a Windows/macOS beta. It loads an audio file and auto-generates a piano score ("音源ファイルを読み込むだけでピアノ譜を自動作成できるツール", "a tool that automatically creates a piano score just by loading an audio file"). A standard song takes about 3 minutes. — [窓の杜, 2026-09-17](https://forest.watch.impress.co.jp/docs/news/2141191.html); [ITmedia NEWS, 2026-09-14](https://www.itmedia.co.jp/news/article/2609/14/2000001452/)
- Options:
  - 4 difficulty levels: 入門/初級/中級/上級 (beginner/elementary/intermediate/advanced).
  - Key: automatic, easy key or original key.
  - Length: full song or 1 chorus.
  - Style: solo score (ソロ譜) or accompaniment score (伴奏譜).
  - Playback with a piano tone; in-app editing on the paid plan.
  - Source: [窓の杜](https://forest.watch.impress.co.jp/docs/news/2141191.html)
- Exports: PDF, MIDI and MusicXML, available on the ADVANCED plan. — [窓の杜](https://forest.watch.impress.co.jp/docs/news/2141191.html); [data-litmus](https://data-litmus.com/articles/ai/2026-09-14-yamaha-piano-sheet-converter-beta-price/)
- Pricing:
  - FREE plan: no registration, 5 conversions per month.
  - ADVANCED plan: ¥1,980/month, unlimited conversions, editing and export.
  - All ADVANCED features were free during the beta, which ran to the end of October 2026.
  - Sources: [窓の杜](https://forest.watch.impress.co.jp/docs/news/2141191.html); [data-litmus](https://data-litmus.com/articles/ai/2026-09-14-yamaha-piano-sheet-converter-beta-price/)
- Terms:
  - Commercial use of generated scores is prohibited ("生成した楽譜の商用利用は禁止されている"). — [窓の杜](https://forest.watch.impress.co.jp/docs/news/2141191.html)
  - Not available in the EEA or other GDPR regions, per 窓の杜. That matters for a user in Poland. — [窓の杜](https://forest.watch.impress.co.jp/docs/news/2141191.html)
  - data-litmus mentions no regional limits, so the two sources differ on this point. — [data-litmus](https://data-litmus.com/articles/ai/2026-09-14-yamaha-piano-sheet-converter-beta-price/)
- Accuracy: Yamaha says accuracy is highest for pop songs with vocals; other genres are future targets. I found no published metrics or technical paper. — [data-litmus](https://data-litmus.com/articles/ai/2026-09-14-yamaha-piano-sheet-converter-beta-price/)
- Roadmap: a full Windows/macOS release in autumn 2026, then iOS/Android in spring 2027. Official beta site: https://piano-sheet-converter.beta.yamaha.com/ — [ITmedia](https://www.itmedia.co.jp/news/article/2609/14/2000001452/)
- Reception on social media: users called it "やばい" / "凄すぎ" (roughly "insane" / "amazing"), but some said accuracy was uneven. This is anecdotal. — [Yahoo!リアルタイム検索 まとめ, 2026-09-14](https://search.yahoo.co.jp/realtime/search/matome/c543594463fd49e283dbb695bb5e0b1d-1789341000)
- A secondary article says Yamaha is also researching AI that analyzes scores and generates new scores at different difficulty levels. UNVERIFIED: the Yamaha R&D news page (https://www.yamaha.com/ja/tech-design/research/news/25_senjyu/) returned 403. — [ITmedia search snippet](https://www.itmedia.co.jp/news/article/2609/14/2000001452/)

**Taiwan: Etude (National Taiwan University, Dept. of Information Management)**
- Authors and venue: Tse-Yang Chen and Yuh-Jzer Joung. arXiv 2509.16522, submitted 2025-09-20, revised 2026-01-23 (v2). Preprint only; no venue is stated. — [arXiv abs](https://arxiv.org/abs/2509.16522); [arXiv HTML v2](https://arxiv.org/html/2509.16522v2)
- Pipeline:
  - **Extract:** an AMT-APC-based extractor.
  - **Structuralize:** a beat detector built on Spleeter and Beat-Transformer.
  - **Decode:** a GPT-NeoX decoder (8 layers, hidden size 512, about 25.5M parameters) over a simplified REMI tokenization.
  - The decoder has style controls for polyphony, rhythm and sustain.
  - Sources: [arXiv HTML v2](https://arxiv.org/html/2509.16522v2); [GitHub](https://github.com/Xiugapurin/Etude)
- Data: about 7,700 pop/piano-cover audio pairs, mostly J-pop and K-pop. After filtering and alignment, 4,752 pairs (about 500 h) remain. The test set is 100 songs: 25 each of C-pop, J-pop, K-pop and Western pop. — [arXiv HTML v2](https://arxiv.org/html/2509.16522v2)
  - Discrepancy: one search snippet describes "3,000 pairs, ~180 h", probably from v1. Treat v2 as authoritative.
- Objective results (WPD ↓ / RGC ↓ / IPE): — [arXiv HTML v2](https://arxiv.org/html/2509.16522v2)

  | Model | WPD | RGC | IPE |
  |---|---|---|---|
  | Human | 0.49 | 0.042 | 10.13 |
  | Etude-Default | 0.21 | 0.020 | 9.02 |
  | PiCoGen2 | 1.00 | 0.059 | 7.97 |
  | AMT-APC | 0.09 | 0.114 | 10.69 |
  | Music2MIDI | 0.18 | 0.160 | 8.94 |

- Subjective results: 101 listeners rated 60 s clips, with all models rendered through the same synth. Overall scores: — [arXiv HTML v2](https://arxiv.org/html/2509.16522v2)
  - Human 3.92
  - Etude 3.50
  - PiCoGen2 2.97
  - AMT-APC 2.46
  - Music2MIDI 2.27
  - Etude beat every baseline on overall quality (p < 0.001). Pop2Piano was not among the baselines.
- Availability:
  - Code: MIT license. — [GitHub](https://github.com/Xiugapurin/Etude)
  - Checkpoints: CC BY-NC-SA 4.0; the README says "commercial purposes … strictly prohibited". — [GitHub](https://github.com/Xiugapurin/Etude)
  - Running it: `infer.py` accepts a YouTube URL or audio file and writes a .mid. It needs a GPU with 16 GB+ VRAM and a separate Python 3.8 env for Spleeter. — [GitHub](https://github.com/Xiugapurin/Etude)
  - Demos: [project page](https://xiugapurin.github.io/Etude/)

**Taiwan: PiCoGen / PiCoGen2 (Yi-Hsuan Yang's group; Chih-Pin Tan)**
- PiCoGen2 was published at ISMIR 2024 by Tan, Ai, Chang, Guan and Yang. It pre-trains piano-to-piano and fine-tunes on weakly aligned song–piano pairs. Yang said it is "Open source w/ docker support". — [arXiv 2408.01551](https://arxiv.org/abs/2408.01551); [Yang on X](https://x.com/affige_yang/status/1822889316160590315)
- PiCoGen (ICMR 2024) uses two stages: lead sheet transcription, then piano generation. — [arXiv 2407.20883](https://arxiv.org/html/2407.20883v1)

**Singapore/NUS-affiliated (Gus Xia, Ziyu Wang, Ye Wang), relevant because of POP909 (Chinese pop)**
- Paper: "Learning Music Style for Piano Arrangement Through Cross-Modal Bootstrapping" by Jingwei Zhao, Gus Xia, Ziyu Wang and Ye Wang. arXiv 2608.03050, 2026-08-04, accepted to ISMIR 2026. — [arXiv abs](https://arxiv.org/abs/2608.03050)
- Method: a BLIP-2-style Q-Former extracts style from a pretrained audio LM and conditions a symbolic LM. Inputs are a lead sheet (content) and reference audio (style). For audio covers, the lead sheet comes from Sheetsage. — [arXiv HTML](https://arxiv.org/html/2608.03050v1)
- Data: trained on POP909 and PIAST (about 8K piano recordings with transcriptions). — [arXiv HTML](https://arxiv.org/html/2608.03050v1)
- Results versus PiCoGen2:
  - On the POP909 test set, PiCoGen2 is better on melody and chord accuracy (MCA 39.3 vs 32.0; CA 40.0 vs 33.3). The new model is better on the style metrics: GPC 79.2 vs 78.4, VCC 76.6 vs 73.2, TA 83.6 vs 74.4.
  - Out of distribution (Ballroom/GTZAN), the new model leads on all three reported metrics: MCA 17.8 vs 17.2, CA 16.3 vs 15.5, TA 79.4 vs 57.6.
  - Source: [arXiv HTML](https://arxiv.org/html/2608.03050v1)
- Availability: a demo page exists at https://zhaojw1998.github.io/bossa/. Neither the paper nor the abstract page mentions released code or weights. — [arXiv HTML](https://arxiv.org/html/2608.03050v1)

**Korea: KAIST (Juhan Nam lab)**
- PianoBind (ISMIR 2025) by Hayeon Bang, Eunjin Choi, Seungheon Doh and Juhan Nam. It is a joint audio/symbolic/text embedding for pop-piano music, used for retrieval. It is not a generator, but it is relevant as an evaluation or embedding tool for pop-piano. — [arXiv 2509.04215](https://arxiv.org/abs/2509.04215)
- PIAST is a KAIST pop-piano dataset with about 8K recordings and transcriptions. It is used as training data in 2608.03050. — [arXiv 2608.03050 HTML](https://arxiv.org/html/2608.03050v1)
  - UNVERIFIED: that PIAST's authors are from KAIST, and the details of its license.

**Korea: consumer services (transcription or "AI sheet" tools; marketing claims only)**
- LaLaNote (라라노트): AI analyzes melody, chords and rhythm from MP3/WAV/YouTube and generates an editable score. — [lalanote.kr](https://lalanote.kr/)
- JUICE (리라): uploads an MP3 and produces an editable digital score. — [juice.co.kr](https://juice.co.kr/ko/services)
- 나도애드립 (NadoAdlib): software for ad-lib and arrangement, aimed at jazz and K-pop ad-libs. It is not an audio-to-piano-cover tool. — [nadoadlib.com](https://www.nadoadlib.com/arrangement/)
- UNVERIFIED for all three: whether they generate a true piano arrangement rather than a lead sheet or transcription.

**China: consumer "AI扒谱" (transcription) tools**
- A Tencent News comparison from 2026-07-09 covers 爱扒谱, 反谱, StemSplit, AudioBatchPro and 音谱小助手. The leaders combine source separation with transcription and export PDF/MIDI/MusicXML/简谱/tab. — [腾讯新闻 2026-07-09](https://news.qq.com/rain/a/20260709A03NE300)
  - 爱扒谱 claims "正确率可达 98% 以上" ("accuracy can exceed 98%"). This is vendor marketing and the method is unknown.
  - None of the five is described as a piano-cover arranger.
- Sites: 爱扒谱 [aibapu.cn](https://aibapu.cn/); 反谱 [miaogaichong.com](https://miaogaichong.com/)
- One article says some tool offers a "Piano Cover模式" focused on producing piano adaptations of mixed-instrument audio. UNVERIFIED: which tool, and how it works. — [smzdm](https://post.smzdm.com/p/a0v8pw7z/)
- Tencent Music's AI activity is in AI-generated songs. The 2026 Q1 report says AI songs make up a growing share of new releases, and Tencent has an AI-music app called 未音. NetEase runs AI songwriting contests. Neither is a piano-arrangement product. — [tmtpost](https://www.tmtpost.com/7944172.html); [扬子晚报 2026-04](https://www.yangtse.com/news/zntt/202604/t20260417_343255.html)

**Japan: academic background (pre-2024, for context)**
- Takamori, Fukayama, Goto and Morishima (AIST/Waseda), IPSJ SIG-MUS 2017/2018: audio-to-piano arrangement. The right hand plays melody plus chord tones from Songle; the left hand is chosen from an accompaniment database of existing piano scores. The 2018 version handles song structure. — [AIST PDF](https://staff.aist.go.jp/m.goto/PAPER/SIGMUS201808takamori.pdf); [IPSJ](https://ipsj.ixsq.nii.ac.jp/record/183046/files/IPSJ-MUS17116013.pdf)
- Nakamura et al. on piano reduction (Kyoto University line). — [PDF](https://eita-nakamura.github.io/articles/Nakamura_etal_PianoReduction_ja.pdf)

### Inferences
- Yamaha's product is the strongest industry signal from East Asia. It is the first major instrument maker to ship audio-to-piano-score with difficulty levels and an accompaniment-only mode, which is a feature research models lack. It is closed, consumer-only, bans commercial use and is reportedly geo-blocked in the EEA. So it can serve as a qualitative benchmark only, not as a component.
- Etude is the most usable new open model. It uses the AMT-APC extractor this project already uses, and it reports a large subjective gain over AMT-APC (overall 3.50 vs 2.46). The NC-SA weights limit commercial use, but the MIT code could be retrained on our own data. Its 16 GB VRAM requirement exceeds the 3060 Ti (8 GB) but fits the 5070 only if that card has 12 GB+; this needs checking, and the decoder alone is small (25.5M).
- The Chinese platforms have probably not published piano-cover features. Their public AI effort is aimed at song generation and labeling. Chinese consumer tools are transcription plus separation.

### Gaps
- I found nothing public on ByteDance/TikTok, Kuaishou, Alibaba (ModelScope/Qwen), Baidu, Ping An or KKBOX piano-cover work. "Not found" here is not proof of absence; internal features may exist.
- I found nothing for Kawai, Roland, Sony CSL, CyberAgent, Piascore or ぷりんと楽譜 AI arrangements. On ぷりんと楽譜 and Piascore, searches for "AI" returned the artist named AI, not AI-generated arrangements.
- Yamaha Smart Pianist "Audio to Score" was not re-verified in this session.
- I could not get Yamaha's own press release, a technical description or a post-beta status (whether the autumn 2026 full release happened).
- Music2MIDI (MMM 2025, Springer LNCS): I could not determine its authors or affiliation. — [Springer](https://link.springer.com/chapter/10.1007/978-981-96-2064-7_8)
- AMT-APC (arXiv 2409.14086): I did not re-verify its authors. From memory, Komiya and Fukuhara, Japan. UNVERIFIED.

## 2. Follow-ups from the Pop2Piano (KAIST/SNU) and PiCoGen (Yang lab) authors, and new datasets

### Takeaway
I found no 2025–2026 piano-cover follow-up from the Pop2Piano authors (Jongho Choi of Rebellions; Kyogu Lee of SNU MARG). From the PiCoGen side, Chih-Pin Tan and Yang posted **LargeSHS**, a cover-song adaptation dataset, in Nov 2025, and then withdrew it over licensing. On datasets, PIAST (KAIST) and Etude's roughly 500 h J-pop/K-pop pair set are the new data points, and Etude's data is not released as far as I found.

### Cited Findings
- Pop2Piano is by Jongho Choi (Rebellions Inc.) and Kyogu Lee (SNU). Code is at github.com/sweetcocoa/pop2piano and it is integrated into HF Transformers. My searches found no newer piano-cover paper by either author. — [arXiv 2211.00895](https://arxiv.org/pdf/2211.00895); [HF docs](https://huggingface.co/docs/transformers/model_doc/pop2piano); [SNU MARG page](https://snu-marg.notion.site/Pop2Piano-2c4426c5113f4360bbbb7c0be7003d98)
- LargeSHS ("A large-scale dataset of music adaptation") by Chih-Pin Tan, Hsuan-Kai Kao, Li Su and Yi-Hsuan Yang. arXiv 2511.15270, 2025-11-19. — [arXiv](https://arxiv.org/abs/2511.15270)
  - Built from SecondHandSongs: 1.7M+ metadata entries and about 900k audio links, organized as adaptation trees.
  - Withdrawn because "the submitter did not have the right to agree to the license". No piano-specific subset is mentioned.
- Other Yang-lab symbolic piano work surfaced in search, but I did not check whether it is cover-related. UNVERIFIED relevance. — [arXiv 2509.23749](https://arxiv.org/pdf/2509.23749); [arXiv 2510.05881](https://arxiv.org/pdf/2510.05881)
  - "Time-Shifted Token Scheduling for Symbolic Music Generation" (2509.23749).
  - "Segment-Factorized Full-Song Generation on Symbolic Piano Music" (2510.05881).
- KAIST (Nam lab): PianoBind (ISMIR 2025) and the PIAST dataset (see section 1). — [arXiv 2509.04215](https://arxiv.org/abs/2509.04215)
- Etude's dataset: about 7.7k pairs reduced to 4,752 aligned pairs (about 500 h), J-pop/K-pop heavy. I saw no release statement. — [arXiv HTML v2](https://arxiv.org/html/2509.16522v2)

### Inferences
- The PiCoGen line seems to have moved toward large-scale cover and adaptation data (LargeSHS). A re-release is plausible but unconfirmed.
- The Pop2Piano authors appear to have left the topic. Rebellions is an AI-chip company.

### Gaps
- I did not search Google Scholar profiles for Tan, Choi or Lee, so a 2026 workshop or Korean-venue (KCI) paper could have been missed.
- I did not find a re-posted LargeSHS or its license.
- PIAST's license and authors were not verified.

## 3. Releases in Sep–Nov 2026

### Takeaway
In this window, the main release is Yamaha Piano Sheet Converter β (2026-09-11). Just before it, the Zhao/Xia/Wang/Wang cross-modal style paper appeared (arXiv 2026-08-04, ISMIR 2026). I found no other East Asian piano-cover model or product launched in Sep–Nov 2026.

### Cited Findings
- 2026-09-11: Yamaha Piano Sheet Converter β (Win/Mac). The free full-feature beta ran to the end of October 2026; the full release was planned for autumn 2026 and mobile apps for spring 2027. — [ITmedia](https://www.itmedia.co.jp/news/article/2609/14/2000001452/); [data-litmus](https://data-litmus.com/articles/ai/2026-09-14-yamaha-piano-sheet-converter-beta-price/)
- 2026-08-04: arXiv 2608.03050, cross-modal style piano arrangement, ISMIR 2026. Only a demo page; no code was found. — [arXiv](https://arxiv.org/abs/2608.03050)
- 2026-01-23: Etude v2 posted, with public code and weights. This is outside the window but the newest open model. — [arXiv](https://arxiv.org/abs/2509.16522); [GitHub](https://github.com/Xiugapurin/Etude)

### Inferences
- ISMIR 2026 proceedings (presumably Sept–Nov 2026) may contain more East Asian piano-cover papers that I did not enumerate.

### Gaps
- I did not browse the ISMIR 2026 accepted-paper list or the ICASSP 2027 submissions.
- I could not confirm whether Yamaha's full release or its post-beta pricing has happened as of 2026-11-18.
- I found no Sep–Nov 2026 announcements from Chinese or Korean companies.
