# What practitioners actually use for AI piano covers (2025-2026): community evidence

Scope note: Reddit could not be reached. The search tool rejects reddit.com ("domains are not accessible to our user agent"), so this file has **no Reddit evidence**. Most web results on this topic are vendor content: Songscription alone runs dozens of "X vs Y" blog posts. I separate independent sources from vendor ones below. Overall, independent third-party evidence is thin, and much of it is anecdotal (from one tester).

## 1. Which tools people recommend, and what they praise or criticise

### Takeaway
Among end users, the independent tests that exist are about **audio-to-sheet transcription tools**, not generative cover models. The ranking repeats across these tests: **Klangio Piano2Notes ≥ Songscription > AnthemScore >> Melody Scanner / ScoreCloud / PianoConvert / Logic**. All of them do reasonably on solo or isolated piano and badly on full pop mixes, where rhythm and meter are the main failures. In the end-user sources I found, nobody discusses the research cover models (Pop2Piano, AMT-APC, Etude, PiCoGen2) as practical tools. They show up only in papers and in auto-generated "literature review" summaries.

### Cited Findings
- **Chinese tester (sspai.com, 2025-11-14), 9 tools tested on piano audio.** Translated tier list:
  - Top tier: Songscription AI ("more than enough, even on the free version alone", about 80%+ accuracy) and AnthemScore ("sufficient, but troublesome", dated UI, one-time purchase).
  - **Piano2Notes**: "This is the one I recommend most". Accurate, clean staff notation, and the best chord detection in the group.
  - Ivory: falls short of Piano2Notes in everything except the playback sound.
  - Capella audio2score Pro: chord labels mostly wrong, "not worth" 1,337 RMB.
  - "Unusable" tier: **ScoreCloud, Logic Pro audio-to-MIDI, PianoConvert, Melody Scanner** ("error rate too high, completely unusable").
  - Source: [sspai.com/post/103825](https://sspai.com/post/103825)
- **The same test, apparently re-published in English** (Substack, 2026-04-22). It says the input was "low-quality piano stems extracted from pop songs" and gives Piano2Notes as the "top recommendation". **It is probably the same author or test as sspai, so count the two as one independent source, not two.** [propsyst.substack.com](https://propsyst.substack.com/p/i-tested-9-piano-audio-to-sheet-music)
- **MusicRadar review of Songscription** (2025-12-08) by Ethan Hein, an NYU transcription teacher. This is the most rigorous independent test I found.
  - Full mixes:
    - "Good Vibrations": time was "complete chaos", though melody and harmony were substantially in place.
    - "Everybody Laughs": everything two beats off.
    - "Jumpin' Jack Flash": no melody found.
    - "Chain of Fools": C under all the chords.
    - "Crazy": meter wandered between 3/4, 4/4, 11/8 and 6/4.
  - **Piano Arrangement feature** (full mix → piano cover), tested on Björk's "Isobel": it got the rough shape of the vocal melody and chords but couldn't fit them to the beat and added 7/8 bars. "I can see why the Piano Arrangement is an alpha feature."
  - Verdict: it "works best on beginner-level classical repertoire and very simple pop songs recorded on solo piano". He calls it "a remarkable tech demo", and says cleaning up its output would be harder than writing the charts by hand. Headline: "Humans will be doing all the serious music transcription for the foreseeable future."
  - Source: [MusicRadar](https://www.musicradar.com/music-tech/humans-will-be-doing-all-the-serious-music-transcription-for-the-foreseeable-future-songscription-review)
- **Agent-finder review of Songscription** (2026): estimated 85-90% note accuracy on clean solo piano, dropping noticeably on dense mixes. Rough estimate, method unstated. [agent-finder.co](https://agent-finder.co/reviews/songscription)
- **Japanese note.com user (kojikoji)** on Songscription: chord marking has "some accuracy" but isn't fully correct, and it can't identify seventh or slash chords. Anecdotal, one user. [note.com](https://note.com/kojikoji3744/n/n93579a4a7f90?hl=en)
- **Piano2Notes (Klangio)**
  - Vendor claims it isolates the piano part from full ensembles and turns any song or YouTube link into "playable piano note sheets". [klang.io](https://klang.io/?p=37700)
  - It has an arrangement mode with "Simple or Complex" difficulty. [Klangio Updates](https://klang.io/?p=9373)
  - App-store and Trustpilot users are mixed. Complaints are about cost per song and transcription stopping early. Praise includes keeping "the particularities of my rhythms and phrasings". These are anecdotes. [justuseapp](https://justuseapp.com/en/app/1589420929/piano2notes/reviews), [Trustpilot](https://uk.trustpilot.com/review/klang.io)
- **Vendor benchmark (treat with caution):** a search-result snippet of Songscription's blog claims first place in an "October 2026 benchmark", scoring 58.5 vs Klangio's 49.3 on piano sheet music. I could not verify the methodology, and it is self-published. [songscription.ai](https://www.songscription.ai/blog/most-accurate-ai-music-transcription)
- **Comparison sites:** iTechGuides scores the AnthemScore editor 9.1 vs 8.7 for Piano2Notes, and lists Piano2Notes as cheaper ($6.24/mo vs AnthemScore $8.33/mo equivalent). This is an aggregator page, not a listening test. [itechguides](https://www.itechguides.com/compare/anthemscore-vs-piano2notes/)
- **Pop2Piano / AMT-APC / PiCoGen**
  - I found no independent user reviews. What exists is "literature review" summaries (themoonlight.io, AI-written), which repeat the papers' claims, e.g. that Pop2Piano "struggles with complex rhythms". [themoonlight AMT-APC](https://www.themoonlight.io/review/amt-apc-automatic-piano-cover-by-fine-tuning-an-automatic-music-transcription-model)
  - The same goes for PiCoGen beating Pop2Piano subjectively despite lower objective accuracy (MCA). [themoonlight PiCoGen](https://www.themoonlight.io/review/picogen-generate-piano-covers-with-a-two-stage-approach)
  - Pop2Piano is mostly listed on AI-tool directories as a Colab/HF demo that outputs MIDI. [promptlayer](https://www.promptlayer.com/models/pop2piano), [artvy](https://www.artvy.ai/resource/pop2piano-demo)
- **Suno** comes up in HN discussion as a cover tool rather than a sheet tool.
  - One user's workflow: "convert the arrangement using the cover feature in Suno."
  - The OP rejected it as "AI slop."
  - Source: [Ask HN: AI music covers in 2026?](https://news.ycombinator.com/item?id=46641627)

### Inferences
- The only independent ranking (sspai/Substack, one tester) and the vendor-run comparison both put Piano2Notes and Songscription at the top. They are probably the "default" consumer choices in 2026, but independent confirmation is thin: about 2-3 independent sources.
- Melody Scanner, which is often listed on SEO roundups, was rated "unusable" by the one independent tester. Treat its inclusion in listicles sceptically.
- Hein's critique matches the known weaknesses of research cover models (rhythm/beat alignment, meter), even though the commercial "song → piano arrangement" features are reviewed rarely and badly. This suggests **beat/meter alignment is the main unsolved practical problem**, more than pitch.

### Gaps
- No Reddit evidence (domain blocked). r/piano and r/WeAreTheMusicMakers opinions couldn't be checked.
- I found no independent hands-on reviews of AMT-APC, Etude, PiCoGen2 or Pop2Piano as tools, and no Moises piano-arrangement reviews.
- I couldn't find Bilibili or Naver user posts with real tests. Korean and Japanese searches returned only paper summaries and product pages.

## 2. Head-to-head comparisons and blind listening tests by third parties

### Takeaway
I found **no third-party blind listening test** of AI piano-cover tools. The only independent head-to-head is the one-person, 9-tool transcription test (sspai / Substack). Everything else is either vendor comparison blogs (mainly Songscription's own) or authors' own subjective evaluations inside papers.

### Cited Findings
- 9-tool hands-on comparison, informal and with no blinding or numeric scores beyond rough accuracy percentages. Winner: Piano2Notes. [sspai](https://sspai.com/post/103825); [Substack](https://propsyst.substack.com/p/i-tested-9-piano-audio-to-sheet-music)
- Songscription publishes many comparison posts against Klangio, AnthemScore and Melody Scanner. Its own search snippets admit they are "the company's own publication" with no independent check, and its blog advises that "five minutes of testing on your actual material will tell you more than any comparison post." [Songscription vs Klangio vs AnthemScore](https://www.songscription.ai/blog/songscription-vs-klangio-vs-anthemscore), [Klangio alternatives](https://www.songscription.ai/blog/klangio-alternatives)
- The Etude paper (2025) claims quality "comparable to that of human composers" in its own listening test. This is the authors' claim, not a third-party one. [arXiv 2509.16522](https://arxiv.org/html/2509.16522v2)
- Arranger for Hire (Jon Burr, ~Dec 2025):
  - MT3 and YourMT3+ demos: "I tried these and couldn't get them to run."
  - "a significant gap remains between raw data capture and musical intelligence"
  - "In many cases, aural transcription remains the faster, more efficient path."
  - Source: [arrangerforhire.com](https://arrangerforhire.com/audio-to-midi-transcription-in-2025-how-good-is-it-really/)

### Inferences
- A blind listening comparison of Pop2Piano, AMT-APC, Etude, PiCoGen2 and commercial tools would be new. There seems to be no public community benchmark.

### Gaps
- I couldn't locate YouTube comparison videos ("I tried every AI piano cover tool") or their transcripts. They may exist but weren't surfaced.

## 3. Workflows creators use

### Takeaway
The workflow that keeps coming up has three steps:
1. **Stem separation** (Moises, LALAL.AI, UVR, Demucs-based tools).
2. **AI transcription** (Piano2Notes, Songscription, Basic Pitch, NeuralNote).
3. **Manual cleanup and arrangement** in MuseScore, Sibelius or a DAW.

Professionals say that for dense mixes, arranging by ear is often still faster. A minority uses Suno's cover feature, or an arranger keyboard and then Suno.

### Cited Findings
- **The sspai tester's own pipeline:** extract the piano stem from a pop song, then transcribe it. Input was "low-quality piano stems extracted from pop songs". [Substack](https://propsyst.substack.com/p/i-tested-9-piano-audio-to-sheet-music)
- **Arranger for Hire's guidance:**
  - Stem-separate dense mixes, then budget time for removing ghost notes and fixing velocities and instrument assignment.
  - Basic Pitch needs "velocity cleanup". NeuralNote is "Best for solo takes".
  - "AI ... has yet to learn how to identify musical intent."
  - Source: [arrangerforhire.com](https://arrangerforhire.com/audio-to-midi-transcription-in-2025-how-good-is-it-really/)
- **Vendor guides:**
  - Recommend stem separation before transcription and export to MuseScore, Finale or Sibelius to fix voicing and rhythm.
  - Note that separation leaves "spectral ghosting" such as snare bleed.
  - Source: [Songscription: what is stem separation](https://www.songscription.ai/blog/what-is-stem-separation), [recordings to piano sheet music](https://www.songscription.ai/blog/recordings-to-piano-sheet-music)
- **Free browser stack:** demucs separation plus transcription plus printable sheet music. [free-music-demixer](https://free-music-demixer.pages.dev/announcements/2025/01/13/New-feature-printable-sheet-music)
- **HN user vunderba's workflow:** sing a motif, notate it, flesh it out on an arranger keyboard, then "convert the arrangement using the cover feature in Suno". [HN](https://news.ycombinator.com/item?id=46641627)
- **Chinese forum context:** users say transcription (扒谱) is the most time-consuming step and want one-shot audio → editable score, which forum authors consider still hard. Older tools such as AudioScore Ultimate and ScoreCloud are recommended there. [V2EX](https://www.v2ex.com/t/971299)
- **Japanese practice:** Chord ai (chord detection plus 4-stem separation plus MIDI export) is used to work out songs by ear (耳コピ). [aipicks.jp](https://aipicks.jp/mag/chord-ai-guide-2026)

### Inferences
- Practitioners treat AI output as a **draft**, and the human arranger owns rhythm, meter and voicing. No source describes the reverse workflow (generate audio with Suno, then transcribe it to a score) as common.

### Gaps
- I found no documented workflow from top YouTube piano-cover channels (e.g. Synthesia-style creators) on how much AI they use.

## 4. New tools in the last few months (Sep-Nov 2026) said to be clearly better

### Takeaway
The notable new entrant is **Yamaha "Piano Sheet Converter"**. Yamaha announced its free beta on 2026-09-11. It does audio → piano score with 4 difficulty levels, key options and a solo or accompaniment style, and claims its best accuracy is on pop songs. I found **no community verdicts yet** saying it, or any other new tool, is clearly better.

### Cited Findings
- **Yamaha Piano Sheet Converter** (beta v1.0.0, Windows and macOS, announced 2026-09-11; article 2026-09-17):
  - Converts a typical song in about 3 minutes.
  - Options:
    - Difficulty: 入門 / 初級 / 中級 / 上級 (beginner, elementary, intermediate, advanced).
    - Key: automatic, easy, or original.
    - Length: full song or one chorus.
    - Style: solo or accompaniment.
  - Pricing:
    - Free plan: 5 conversions per month.
    - ADVANCED plan: ¥1,980/mo, with editing and PDF/MIDI/MusicXML export.
    - ADVANCED features are unlocked free in the beta (reported until end of Oct 2026).
  - Pop songs have "the highest accuracy".
  - Unavailable in the EEA/GDPR regions.
  - Commercial use of output is prohibited.
  - Full release expected around fall 2026, mobile around spring 2027.
  - Source: [Forest Watch / Impress](https://forest.watch.impress.co.jp/docs/news/2141191.html)
- **PopPianoAI** (YouTube link → piano sheet with chords; listed launch 2026-01-17): Product Hunt shows "No reviews yet", 2 upvotes. No traction evidence. [Product Hunt](https://www.producthunt.com/products/poppianoai/reviews?page=2); [productdirs](https://productdirs.com/projects/poppianoai)
- **Songscription**:
  - Launched its "AI Shazam for sheet music" in June 2025. [TechCrunch](https://techcrunch.com/2025/06/30/songscription-launches-an-ai-powered-shazam-for-sheet-music)
  - It now offers a "piano cover mode" that is "playable … even if the original has no piano". As of Dec 2025, Hein still judged the arrangement feature alpha quality. [MusicRadar](https://www.musicradar.com/music-tech/humans-will-be-doing-all-the-serious-music-transcription-for-the-foreseeable-future-songscription-review)
- **RollTab** (GIGAZINE, 2026-08-23) is a 125M on-device piano *continuation* model, **not** a cover tool. Listed only to rule it out. [GIGAZINE](https://gigazine.net/gsc_news/en/20260823-transformer-piano/); [HN](https://news.ycombinator.com/item?id=49373456)
- **Other HN Show posts** are solo-piano transcription, not covers, e.g. Pianolyze (ByteDance piano transcription in the browser, which "accompaniment can confuse") and PianoConvert ("up to 98%" vendor claim, rated unusable by sspai). [Pianolyze HN](https://news.ycombinator.com/item?id=45799312); [PianoConvert HN](https://news.ycombinator.com/item?id=44367640)

### Inferences
- Yamaha's tool matters most strategically. It is the first major-brand product that does difficulty-levelled song → piano *arrangement* (not just transcription), and it is pop-focused. It is the most likely new benchmark rival. Its quality is unverified by third parties.

### Gaps
- I found no user reviews of Yamaha's beta (Japanese blogs or X posts weren't surfaced).
- I couldn't confirm whether any Sep-Nov 2026 open-source cover model (successor to Etude or PiCoGen2) has gained community traction.
