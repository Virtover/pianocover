# Commercial ready-made piano cover / arrangement products and APIs (as of 2026-11-18)

Context: "Piano Weave" is a paid Android app that turns songs into piano covers (MIDI), with generation running server-side. Research done 2026-11-18 with about 25 search/fetch calls. Pages were checked on that date unless noted. Claims marked **[UNVERIFIED]** were not confirmed against a primary page in this session.

### Ranked shortlist (synthesis; reasoning in the sections below)
| Rank | Option | Type | Piano *arrangement* (not just transcription)? | B2B API? | May outputs go into a third-party paid app? |
|---|---|---|---|---|---|
| 1 | **Klangio API** (Piano2Notes / Melody Scanner tech) | B2B REST API | Transcription. "Piano arrangement" is offered in its consumer apps, but the API page lists only transcription, separation, beat and chord endpoints | **Yes**, public pricing | Requires a separate business contract. Consumer ToS limits commercial use to the user's own compositions |
| 2 | **Music.AI** (Moises B2B) | B2B module API | No piano transcription or arrangement module on the pricing page. It has piano stem separation, chords, beats and sections | Yes, public pricing | Terms not read. Useful as pre-processing (stems/chords) for an in-house arranger |
| 3 | **Songscription / Play Anything** (Piano Cover Creator, Anything Piano) | Consumer web plus iOS | **Yes**, solo piano arrangement with difficulty levels | **No** public API | **No.** ToS says outputs are for "private, non-commercial use" only. A partnership would have to be negotiated |
| 4 | **La Touche Musicale AI Piano Cover Maker / PianoConvert** | Consumer web/app | **Yes**, arrangement in 2 difficulty levels | A "PianoConvert API" help page exists in search results but returned 404 | Unknown |
| 5 | **Yamaha Piano Sheet Converter** (beta) | Desktop app | Yes, effectively (4 difficulties, solo/accompaniment, key and length options) | No | **No.** Commercial use of the scores is prohibited. Not offered in the EEA (this matters if Piano Weave serves the EU/Poland) |
| 6 | **MuseScore Audio-to-Score** (beta) | Free web feature | Transcription to MSCZ (solo piano/guitar sources) | No | No API |
| — | Yamaha Smart Pianist "Audio to Score" | Hardware-tied app feature | Chord-based accompaniment score | No | No |
| — | Suno / Udio "cover" | Generative audio | Audio cover, not MIDI | Only through unofficial wrapper APIs | Plan-dependent. Composition rights are not cleared |

## 1. What each product produces (arrangement vs transcription vs audio cover), formats, difficulty levels

### Takeaway
The market has moved from plain transcription to "solo piano arrangement with difficulty levels". In 2026, Songscription's Piano Cover Creator, La Touche Musicale's AI Piano Cover Maker and Yamaha's new Piano Sheet Converter all output melody in the right hand and harmony/bass in the left, and export MIDI/MusicXML/PDF. Older tools (AnthemScore, Piano2Notes, MuseScore Audio-to-Score) are transcribers and work best on solo-piano input.

### Cited Findings
**Songscription / Play Anything (US, Songscription, Inc.)**
- The Piano Cover Creator produces a "solo piano arrangement", not a note-for-note transcription. It rewrites vocal, guitar or full-band tracks as a playable part that "captures the melody and chords", with melody in the right hand and harmony and bass in the left. It exports PDF, MIDI and MusicXML and has a piano roll and falling-notes play-along. Difficulty ranges from beginner–intermediate to advanced, and a re-level takes about 30 s, "thins the texture but keeps the melody and chords". No prices are listed on that page — [Songscription Piano Cover Creator](https://www.songscription.ai/piano-cover-creator) (accessed 2026-11-18)
- The difficulty picker runs Early beginner → Beginner → Early intermediate → Intermediate → Late intermediate → Original transcription. The leveler "thins chords, steadies rhythms, shrinks stretches" — [Songscription sheet music leveler](https://www.songscription.ai/sheet-music-leveler); [difficulty levels blog](https://www.songscription.ai/blog/sheet-music-difficulty-levels-explained)
- Pricing from the MusicRadar review (Dec 2025): Free gives 10 three-minute transcriptions a month. Plus is $9.99/mo for 5 six-minute transcriptions. Pro is $29.99/mo for 100 fifteen-minute transcriptions. The current pricing page lists the free tier as "unlimited 30-second transcriptions and 10 three-minute transcriptions per month", capped at 5 per day — [MusicRadar review, 2025-12-08](https://www.musicradar.com/music-tech/humans-will-be-doing-all-the-serious-music-transcription-for-the-foreseeable-future-songscription-review); [songscription.ai/pricing](https://songscription.ai/pricing)
- At the Dec 2025 review the piano arrangement feature was still labelled "alpha". By Nov 2026 it is marketed as a main feature — [MusicRadar](https://www.musicradar.com/music-tech/humans-will-be-doing-all-the-serious-music-transcription-for-the-foreseeable-future-songscription-review)
- The company rebranded to **Play Anything** (reported 2026-09-04). Songscription is the creator product. **Anything Piano** is an iOS learning app that "turns any recording into a simplified piano tutorial with guided fingerings", outputs a piano roll and scores the user's playing. Android and web are waitlist-only. The company raised $5M in Nov 2025 and had more than 150k users. Recent additions are skill-level simplification, an AI-chat notation editor, and lead-sheet/duet export in beta — [AI Musicpreneur, 2026-09-04](https://www.aimusicpreneur.com/ai-tools-news/songscription-publisher-licensing-deals-play-anything-rebrand/); [Product Hunt: Anything Piano](https://www.producthunt.com/products/anything-piano)
- Per the vendor listing, Anything Piano produces a piano-roll tutorial "in about a minute" from a pasted link or upload. It is freemium, with a subscription for unlimited songs — [Product Hunt](https://www.producthunt.com/products/anything-piano)

**La Touche Musicale (France): AI Piano Cover Maker / PianoConvert**
- The page says "the AI does not copy the recording: it writes a piano version". Its FAQ separates transcription (the piano part as played) from arrangement (melody plus adapted accompaniment). There are two levels: simplified (fewer complex chords) and full (richer harmonies). Input can be a YouTube, SoundCloud, TikTok or Bandcamp link, an upload (up to 15 min / 100 MB) or a browser recording. Output is a piano roll, on-screen score and PDF/MIDI/MusicXML export. The first 30 s of each song are free and the full cover needs a subscription (no prices shown). The tech is described only as "our arranging AI". No API is mentioned on that page — [AI Piano Cover Maker](https://latouchemusicale.com/en/tools/ai-piano-cover-maker/) (accessed 2026-11-18)
- Search results show a help-center article titled "Piano Convert API" (latouchemusicale.crunch.help/…/piano-convert-api), but it returned **HTTP 404** when fetched. **[UNVERIFIED]** whether a B2B API exists — [search result URL, 404](https://latouchemusicale.crunch.help/it/piano-convert/piano-convert-api)

**Yamaha Piano Sheet Converter (beta v1.0.0, Japan, released 2026-09-11)**
- It "automatically creates piano sheet music just by loading an audio file". Options are 4 difficulties (入門/初級/中級/上級), key (auto / easy / original), length (full song / one chorus) and style (solo score / accompaniment score). A standard song takes about 3 min. The Free plan needs no registration and allows 5 conversions a month. ADVANCED costs ¥1,980/mo with unlimited conversions, editing and PDF/MIDI/MusicXML export, and is free during the beta. It runs on Windows/macOS. A full release is planned for autumn 2026 and iOS/Android for spring 2027. **Not available in EEA/GDPR regions. Commercial use of generated scores is prohibited.** Accuracy is best on pop songs — [Impress Forest Watch, 2026-09-17](https://forest.watch.impress.co.jp/docs/news/2141191.html). No second source was found to confirm this.
- Yamaha's research site lists "transcribing audio into sheet music" under its Musical Score AI technology — [Yamaha Musical Score AI](https://www.yamaha.com/en/tech-design/research/technologies/musical-score-ai)

**Yamaha Smart Pianist "Audio to Score"** (hardware-tied)
- It analyzes chords in a song stored on the device and makes an accompaniment score. Yamaha says chords are "analyzed with high accuracy", but "may differ from the original song". It usually creates 4/4 scores, so 3/4 songs may come out wrong. It does not work with subscription-streaming music — [Smart Pianist User Guide](https://manual.yamaha.com/mi/app/smartpianist/en/smartpianist_en_ug_j1_003.html); [Yamaha Hub: getting the most out of Audio to Score](https://hub.yamaha.com/pianos/p-digital/getting-the-most-out-of-audio-to-score/)
- Songs "that feature a full band and a regular beat translate the best". Harmonically complex or less defined songs are less suited — [Yamaha Hub](https://hub.yamaha.com/pianos/p-digital/getting-the-most-out-of-audio-to-score/); [Yamaha CSP-150 Smart Pianist page](https://usa.yamaha.com/products/musical_instruments/pianos/clavinova/csp-150/smart_pianist.html)

**Klangio (Germany, Klangio GmbH): Piano2Notes, Melody Scanner, Transcription Studio**
- Piano2Notes / Transcription Studio transcribe audio or YouTube to PDF/MIDI/MusicXML (plus GP5 for guitar) and are distributed through MuseHub too — [MuseHub Piano2Notes](https://www.musehub.com/app/piano2notes); [MuseHub Transcription Studio](https://www.musehub.com/app/music-transcription-studio)
- App Store in-app prices: Piano2Notes Pro Monthly $9.99, Pro Yearly $99.99, Klangio Universe Monthly $24.99, plus token packs — [App Store Piano2Notes](https://apps.apple.com/us/app/id1589420929)
- Melody Scanner is described as turning songs into melody plus chords notation. A third-party comparison (a competitor's blog) calls its standalone mobile app neglected, with no iOS update since early 2024 — [Songscription blog: Melody Scanner alternatives](https://www.songscription.ai/blog/melody-scanner-alternatives) (competitor source, treat as biased)
- **[UNVERIFIED]**: whether Klangio sells a dedicated "piano arrangement from full mix" mode, as opposed to transcription of piano recordings, in Nov 2026. The Japanese article index describes Melody Scanner/Piano2Notes as outputting "楽譜やピアノ編曲" (scores or piano arrangements) — [search snippet only](https://klang.io/?p=37700)

**MuseScore / Muse Group: Audio-to-Score (beta, announced 2026-08-17)**
- It "turns audio files into ready-to-play sheet music". Input is an MP3 (≤30 MB, about 3 min) or a YouTube/Audio.com link, and output is an MSCZ file. It currently recognizes **solo piano** sources ("simple pop and neoclassical … piano ballads") and acoustic guitar. It is free and runs in the web desktop version. It uses an in-house deep-learning model "fully developed and trained in-house by MuseScore". No API is mentioned — [mu.se, 2026-08-17](https://www.mu.se/posts/musescore-audio-score-features)
- The MuseScore mobile apps also added ShazamKit-based song recognition that surfaces existing community or licensed scores, which is not generation — [same source](https://www.mu.se/posts/musescore-audio-score-features)

**StaffPad Piano Capture (Muse Group)**
- It transcribes a *live piano performance* into notation, on-device and offline. StaffPad costs $90 on iPad and Windows tablets. It does not make covers of band songs — [Sound On Sound](https://www.soundonsound.com/news/staffpads-new-ai-powered-piano-capture); [MusicRadar](https://www.musicradar.com/news/staffpad-piano-to-score-feature)

**Piano-learning apps: Simply Piano, flowkey, Skoove**
- No source found says they use AI to arrange their song catalogue. A 2026 comparison says they teach from fixed catalogues "because a human arranging each song is how you get careful, graded, well-fingered versions". Their AI is used for listening and feedback — [HackerNoon 2026 piano app comparison](https://hackernoon.com/best-piano-learning-apps-in-2026-an-in-depth-comparison-of-music-education-technology) (secondary source)

**Suno / Udio (generative audio "cover")**
- These produce *audio* covers, not MIDI. Third-party "upload & cover" API wrappers exist (for example aimusicapi.org and udioapi.pro). They need a publicly hosted audio URL and cap input at 8 min — [aimusicapi.org docs](https://aimusicapi.org/ko/docs/v2-upload-cover); [udioapi.pro docs](https://udioapi.pro/docs/v2-upload-cover). These are not official vendor APIs.

**Korean/Asian findings**
- **Wondera** (wondera.ai) markets a "piano version remake tool". It accepts YouTube/Spotify links, the free plan allows 10 files a month, and it has a "piano vocal cover" accompaniment tool. The page mentions a commercial-use license, but it returned 403 on fetch, so this is from the search snippet only — [Wondera (KO)](https://wondera.ai/tools/ko/piano-version-remake-tool). **[UNVERIFIED]** whether output is audio or MIDI.
- **DeepPiano** (iOS) converts piano audio to sheet music and adds practice feedback — [mwm.ai listing](https://mwm.ai/ko/apps/deeppiano-sheet-music-ai/6642699519)
- **虫虫钢琴 (Chongchong Piano)**: the official App Store description lists AI score playback, smart fingering and 五线谱↔简谱 (staff ↔ numbered notation) conversion, plus an AI practice partner. **No** feature for arranging an arbitrary song into piano was found — [App Store 虫虫钢琴](https://apps.apple.com/cn/app/%E8%99%AB%E8%99%AB%E9%92%A2%E7%90%B4-%E4%B8%93%E4%B8%9A%E5%AD%A6%E9%92%A2%E7%90%B4%E7%BB%83%E7%90%B4%E7%A5%9E%E5%99%A8/id1185380895)
- **小星星AI陪练** (Xiaoxingxing AI Practice) does photo/PDF to smart score (OMR), not audio to arrangement — [App Store](https://apps.apple.com/np/app/%E5%B0%8F%E6%98%9F%E6%98%9Fai%E9%99%AA%E7%BB%83-%E6%99%BA%E8%83%BD%E9%92%A2%E7%90%B4%E9%99%AA%E7%BB%83/id1566969508)

**Other**
- PopPianoAI (Product Hunt launch) arranges solo piano from the original pop song in "about 3 minutes". Pricing is shown as "Free" on Product Hunt, and it is an early-stage indie product — [Product Hunt](https://www.producthunt.com/posts/1063620); [PopPianoAI blog](https://poppiano-ai.netlify.app/en/blog/1323)
- AnthemScore (Lunaverus) is an offline desktop transcriber sold as a one-time purchase. Several comparison sites say it usually needs manual cleanup — [Lunaverus compare](https://www.lunaverus.com/compare); [MuseScore forum thread](https://musescore.org/en/node/272456)

### Inferences
- The closest direct competitors to Piano Weave in Nov 2026 are Songscription's Piano Cover Creator, La Touche Musicale's AI Piano Cover Maker and (from spring 2027, mobile) Yamaha Piano Sheet Converter. All three do "arrangement plus difficulty levels plus MIDI/MusicXML/PDF export". Piano Weave's differentiator therefore cannot be just "song → piano MIDI".
- Yamaha's Piano Sheet Converter excludes the EEA and Anything Piano has no Android release yet, so there is a gap on Android in the EU (as of Nov 2026).
- Songscription's arrangement path is probably transcription → leveling/arrangement ("thins chords, steadies rhythms"), not an end-to-end Pop2Piano-style generator. This is inference from their marketing language, not disclosed.

### Gaps
- **Roland Piano Every Day, Casio and Kawai PiaBookPlayer**: no search result showed an audio-to-score or AI-cover feature for any of them. One targeted search returned nothing relevant. Treat it as "no evidence of such a feature" and not as confirmed absence. Casio's 2026 learning-app move is a partnership with ROLI ("ROLI Learn for CASIO"), which is a learning app and not an arranger — [ROLI blog](https://roli.com/blog/roli-and-casio-announce-partnership).
- These were not researched in this session (budget): ScoreCloud, Flat.io (AI features), Samplab, Musicful, Udio official terms, 弹琴吧, Piano Partner AI, and Japanese apps other than Yamaha.
- No exact prices were found for Songscription's current Piano Cover tiers or La Touche Musicale's subscription.

## 2. Quality reputation (independent reviews, complaints)

### Takeaway
Independent testing is thin. The only substantial independent review found (MusicRadar on Songscription, Dec 2025) praised pitch detection but found rhythm and time-signature errors, failure to find melodies in band mixes, and heavy editing needs. Vendors (Yamaha, Klangio, Songscription itself) all say results are best on regular-beat pop and that output is a "first draft".

### Cited Findings
- MusicRadar (Ethan Hein, NYU, 2025-12-08) on Songscription found pitches "generally strong", but rhythm was "consistently late" and often shifted by an eighth note or two beats. Time signatures were wrong. It "could not find the melody" in Jumpin' Jack Flash. It captured the rough shape of Björk's vocal but placed it poorly on the beat. Verdict: "a remarkable tech demo" and "It's helpful in situations where you don't need help, and unhelpful in situations where you do need help." — [MusicRadar](https://www.musicradar.com/music-tech/humans-will-be-doing-all-the-serious-music-transcription-for-the-foreseeable-future-songscription-review)
- Songscription's own blog says realistic output is "a strong first draft you review, not a finished score", and that its leveler "reduces difficulty mechanically" — [Songscription blog](https://www.songscription.ai/blog/sheet-music-leveling-for-teachers)
- On Yamaha Smart Pianist Audio to Score, Pianoo found accuracy "depends a lot on the music material" and that it works "amazingly well with normal pop and light music" — [Pianoo test](https://pianoo.com/test/yamaha-smart-pianist-piano-app-fuer-ios-android/)
- For the Yamaha Piano Sheet Converter beta, Yamaha says accuracy is highest for pop and warns of possible bugs — [Impress](https://forest.watch.impress.co.jp/docs/news/2141191.html)
- Klangio's ToS says "we cannot guarantee that transcriptions will be complete, error-free, meet musical standards" (§14.2) — [Klangio Terms](https://klang.io/terms/)
- Most "X vs Y" comparisons online (Songscription vs Klangio vs AnthemScore) are written by Songscription, a competitor. One even concedes such posts "overstate small accuracy differences" — [Songscription vs Klangio vs AnthemScore](https://www.songscription.ai/blog/songscription-vs-klangio-vs-anthemscore)
- Korean summary of Piano2Notes App Store reviews: it "mostly works but is sometimes inaccurate; needs a quiet recording environment" — [App Store Piano2Notes (KO)](https://apps.apple.com/us/app/piano2notes/id1589420929?l=ko) (search summary, not read in full)

### Inferences
- The common failure modes are melody extraction from dense mixes and rhythm/beat alignment. These are the dimensions Piano Weave's own evaluation should report against competitors (melody accuracy, beat alignment, playability).
- No product publishes objective metrics on arrangement quality, so a small blind listening test or playability benchmark against Songscription and La Touche Musicale would be novel and marketable.

### Gaps
- No independent review was found of Songscription's *arrangement* mode (as opposed to transcription), La Touche Musicale's AI Piano Cover Maker, PopPianoAI or Yamaha Piano Sheet Converter.
- Reddit threads comparing these tools were not surfaced by search. The MuseScore forum threads were found but not read in full.

## 3. APIs/SDKs: availability, pricing, rate limits, terms (can outputs be shown or sold inside another company's paid app?)

### Takeaway
Of the products checked, only **Klangio** publishes a self-serve transcription API with prices (from $99/mo for 500 requests). Its consumer ToS limits commercial use, and API use is "subject to separate terms", so B2B rights must be negotiated. **Music.AI** sells modules for stems, chords and beats but offers no piano transcription or arrangement module. **Songscription/Play Anything, Yamaha, MuseScore and La Touche Musicale** have no public API, and their consumer terms forbid or don't permit commercial reuse. No API was found that lets a third-party paid app resell piano arrangements of copyrighted songs out of the box. Composition (copyright) licensing is a separate issue from vendor terms in every case.

### Cited Findings
**Klangio API** — [klang.io/api](https://klang.io/api/) (accessed 2026-11-18)
- Endpoints: Music Transcription (piano, guitar, bass, vocal), Source Separation, Beat Tracking and Chord Recognition. Output is MIDI, MusicXML, PDF or GP5.
- Plans (monthly):

  | Plan | Price | Requests incl. | Overage | Max audio | Rate limit |
  |---|---|---|---|---|---|
  | Basic (test) | $0 | 50 | none | 15 s | 1/min |
  | Startup | $99 | 500 | $0.20/req | 300 s | 2/s |
  | Business | $499 | 3,000 | $0.17/req | 300 s | 5/s |
  | Enterprise | contact | 10,000 | $0.10/req | 600 s | 10/s |
- FAQ: "Depending on your plan's specification, you will either incur overage charges or be suspended."
- No arrangement or difficulty endpoint is listed on the API page.
- Terms (no date on the page) — [klang.io/terms](https://klang.io/terms/):
  - §5.2.3: "API access and Bulk Transcription are not included in the subscription plans" and are "offered exclusively as separate business services subject to separate terms."
  - §12.4: "Commercial use is permitted exclusively for original musical compositions created by the user", and only when "based on full transcriptions rather than a demo transcription."
  - §12.5: "Transcriptions may only be used by you in accordance with applicable laws, in particular copyright laws", and transcriptions "must not be used to train music transcription AI systems or similar technologies."
  - §7.5: users may not "sublet it or otherwise transfer rights of use to third parties" without prior written consent.
  - §8.3: the uploader warrants they hold "all necessary rights" to uploaded content.

**Music.AI (Moises B2B)** — [music.ai/pricing](https://music.ai/pricing/) (accessed 2026-11-18)
- Tiers: Pay-as-you-go $0 (2 concurrent jobs, 48 h storage). Professional $25/mo (10 concurrent jobs, $25 credit, 5% off modules). Business/Enterprise is custom (SLAs, integration engineer, high-throughput).
- Per-minute rates (PAYG): piano stem separation $0.07, chords $0.04, beats $0.03, sections $0.04, "General Purpose Transcription" $0.08. That last one is listed among lyrics/speech items and is probably speech-oriented **[UNVERIFIED]**.
- **No piano transcription, MIDI or arrangement module** is listed. The API is a REST "Modules → Workflows → Jobs" model — [music.ai/platform](https://music.ai/platform/)
- Terms were not read in this session.

**Songscription / Play Anything** — [ToS, last updated 2026-06-18](https://playanything.com/terms) (songscription.ai/terms redirects here)
- §8.5(d): "Your use of the Service, including any Output, is strictly for your own private, non-commercial use." Output may not be used to train AI.
- §4.6 bars using outputs "for any commercial purpose, including sale, sublicense, or redistribution."
- §8.4: the user must have "all necessary rights to create Transcriptions of any musical works you transcribe", and "the Service does not grant you any such license or permission."
- §10.4 prohibits automated access ("spiders, robots, crawlers, and data mining tools"). No public API is mentioned.
- No API or partner program was found. However, the company is **signing publisher licences**: Spirit Music Group is public, and "several independent publishers" are claimed by the CEO but unverified. The article notes that notation of a copyrighted composition is a derivative use needing permission — [AI Musicpreneur, 2026-09-04](https://www.aimusicpreneur.com/ai-tools-news/songscription-publisher-licensing-deals-play-anything-rebrand/)

**Yamaha Piano Sheet Converter**: no API. Commercial use of generated scores is prohibited and it is unavailable in the EEA — [Impress](https://forest.watch.impress.co.jp/docs/news/2141191.html). Yamaha does run a separate "Music Connect API" B2B program, but no evidence links it to this converter — [Yamaha Music Connect APIs](https://www.yamaha.com/en/about/business/music-connect/apis) (not read; **[UNVERIFIED]** scope)

**MuseScore Audio-to-Score**: free, with no API or third-party licensing mentioned — [mu.se](https://www.mu.se/posts/musescore-audio-score-features). Muse Group runs a MuseHub developer *distribution* marketplace with 110+ partners including Klangio. That is a storefront for selling desktop plugins and apps, not an API — [mu.se payouts post](https://www.mu.se/posts/musehub-exceeds-1-million-in-payouts-to-over-110-developer-partners); [MuseHub Klangio partner page](https://www.musehub.com/partner/klangio)

**Suno (audio covers)**: third-party guides say commercial use requires a paid plan, Suno doesn't warrant copyright ownership, and the Cover feature "does not grant the composition licence required for a commercial cover". One guide reports terms revised 2026-03-26 restricting remix features to non-commercial use. Dates conflict across guides and the official ToS was not read — [JackRighteous guide](https://jackrighteous.com/blogs/guides-using-suno-ai-music-creation/can-you-monetize-suno-remix-cover-extend-rights); [sunowatermark.com](https://sunowatermark.com/suno-commercial-use-rules/) (secondary sources)

### Inferences
- **Integration feasibility ranking for Piano Weave** (server-side, paid Android app):
  1. **Klangio API**: the only off-the-shelf API with piano transcription and public pricing. At $0.10–0.20 per request, it costs about $0.17–0.20 per song at Startup/Business volume. Its 300 s audio cap on lower tiers truncates long songs. It returns *transcription*, not a playable arrangement, so Piano Weave would still need its own arrangement/leveling layer. Its ToS §12.4 suggests outputs for third-party copyrighted songs are not commercially usable without a negotiated B2B contract, and there is a no-training clause. This makes it a fallback or benchmark rather than a core dependency.
  2. **Music.AI**: useful only as a pre-processing vendor (piano/vocal stems, chords, beats at about $0.15/min combined) feeding an in-house arranger. It has no MIDI output for piano.
  3. **Songscription/Play Anything**: the best arrangement quality on paper, but no API and its ToS bans commercial reuse. The only route is a bespoke partnership, which is plausible because they are actively doing B2B deals with publishers, but there is no evidence of a developer program. They are also a direct competitor (Anything Piano Android is on their waitlist).
  4. **La Touche Musicale**: a possible API. A help-center "Piano Convert API" article existed but 404s now. Worth a direct email.
  5. **Yamaha, MuseScore**: not licensable (no API; non-commercial or region-blocked).
- Whatever vendor is used, the vendor's terms are not a licence to the underlying composition. Songscription's move to sign publishers shows the market leader sees composition licensing as necessary for a commercial "song → piano notation" product. Piano Weave faces the same issue whether it builds in-house or licenses an API.
- Since no vendor offers a licensable, commercial-use-cleared "pop audio → piano arrangement MIDI" API, Piano Weave's in-house server-side pipeline (AMT-APC / Pop2Piano-like) is justified on licensing grounds, not just cost.

### Gaps
- Klangio's separate B2B API terms (output rights, sublicensing, white-labeling) are not public. Request them via the API form.
- Music.AI terms and any piano-MIDI module behind Enterprise were not checked.
- La Touche Musicale API status (404) is unconfirmed.
- No partner or affiliate program was found for any vendor.

## 4. Underlying technology (if disclosed)

### Takeaway
Almost no commercial vendor discloses its models. MuseScore says its model is in-house deep learning. Others describe it as "our AI" or describe a transcription → simplification pipeline. No vendor states it uses Pop2Piano or AMT-APC.

### Cited Findings
- MuseScore Audio-to-Score uses an in-house deep-learning model "fully developed and trained in-house by MuseScore". Its recognition feature uses Apple ShazamKit — [mu.se](https://www.mu.se/posts/musescore-audio-score-features)
- La Touche Musicale: "our arranging AI", with no model named — [AI Piano Cover Maker](https://latouchemusicale.com/en/tools/ai-piano-cover-maker/)
- Songscription: the leveler "thins chords, steadies rhythms, shrinks stretches", which implies transcription followed by rule- or ML-based simplification. Its blog has a "how AI music transcription works" explainer — [Songscription leveler](https://www.songscription.ai/sheet-music-leveler); [how AI transcription works](https://www.songscription.ai/blog/how-ai-music-transcription-works)
- Klangio: transcription plus separation plus beat and chord modules, exposed as separate API endpoints (suggesting a modular pipeline) — [klang.io/api](https://klang.io/api/)
- Yamaha Smart Pianist Audio to Score is chord analysis plus accompaniment-pattern scoring, documented as chord-based with a 4/4 assumption — [Yamaha manual](https://manual.yamaha.com/mi/app/smartpianist/en/smartpianist_en_ug_j1_003.html). Piano Sheet Converter tech is not disclosed — [Impress](https://forest.watch.impress.co.jp/docs/news/2141191.html)
- Open research models in this space: AMT-APC, trained on YouTube piano covers, is listed as a free tool — [AIbase AMT-APC](https://www.aibase.com/tool/33840). A Gigazine piece (2026-08-23) covers a transformer piano model, which was not read — [Gigazine](https://gigazine.net/gsc_news/en/20260823-transformer-piano/)

### Inferences
- Commercial products that offer difficulty levels (Songscription, Yamaha, La Touche Musicale) likely use a two-stage design: transcription/analysis first, then arrangement/leveling. This contrasts with end-to-end audio→piano-cover models such as Pop2Piano and AMT-APC, which have no built-in difficulty control. A difficulty-controllable end-to-end model would differentiate Piano Weave.

### Gaps
- No vendor disclosure of training data or of whether it uses Pop2Piano-style models. Training-data provenance matters for licensing risk and was not found for any product.
- The Gigazine 2026-08-23 transformer-piano article was not read and may describe a relevant new model or product.
