# Model and weight licenses (teacher pipeline + product)

Every third-party model used anywhere in pianocover is listed here with where its license
was checked and when. Re-check before a release; READMEs change.

Product position (Krzysztof's): trained weights are not copies or adaptations of their
training data. That position covers *training-data* caveats below. It does **not** cover a
license stated on the weights themselves.

| Model | Used for | Code | Weights | Training data (caveat) | Checked | Source |
|---|---|---|---|---|---|---|
| Transkun V2 (`2.0.pt`, pip `transkun`) | backbone, piano/other transcription | MIT | Not stated separately; ships inside the MIT pip package | MAESTRO (CC BY-NC-SA), not stated in README | 2026-11-09 | github.com/Yujia-Yan/Transkun (README, LICENSE) |
| Demucs `htdemucs` (4-stem) | stems: vocals/drums/bass/other | MIT | "The license of the weights is not clear" (maintainer) | MusDB HQ (partly research-only) + 800 unnamed songs | 2026-11-09 | github.com/adefossez/demucs README |
| Demucs `htdemucs_6s` | **NOT USED** | MIT | Maintainer: "likely CC-NC-BY license is applicable" | Meta internal dataset | 2026-11-09 | github.com/adefossez/demucs README |
| basic-pitch (Spotify, ICASSP 2022 model) | bass, vocal melody, other stems | Apache-2.0 | No separate statement; weights ship inside the Apache-2.0 package | not stated in README | 2026-11-09 | github.com/spotify/basic-pitch |
| beat_this (`final0`) | beats / downbeats | MIT | "The code and the published model weights are released under the MIT license." | 16 datasets, some "fully copyrighted or under limited Creative Commons licenses" | 2026-11-09 | github.com/CPJKU/beat_this README |
| BTC (ISMIR 2019) | chords (correction only) | MIT | Not stated | Isophonics, Robbie Williams, UsPop2002 labels | 2026-11-09 | github.com/jayg996/BTC-ISMIR19 |
| torchcrepe 0.0.24 (CREPE `full`) | vocal melody + bass line (f0 -> notes) | MIT (Copyright 2020 Max Morrison) | Bundled in the package (`torchcrepe/assets/full.pth`); README: "converted [from marl/crepe] using MMdnn". No separate weights license stated. Upstream marl/crepe is MIT and bundles its pretrained model in the MIT package. | MIR-1K, Bach10, RWC-Synth, MedleyDB, MDB-STEM-Synth, NSynth | 2026-11-10 | github.com/maxrmorrison/torchcrepe (LICENSE, README, setup.py), github.com/marl/crepe |

| onnxruntime-gpu 1.22.0 | runtime for basic-pitch | MIT | n/a | n/a | 2026-11-09 | PyPI metadata |

## Audio sources for teacher data

| Source | Usable? | Why | Checked |
|---|---|---|---|
| FMA (`fma_full`, 2017 dump) | yes, per track | `raw_tracks.csv` has `license_url` per track. About 8,350 of 106,574 tracks are CC0 / CC BY / public domain; the rest is NC and/or ND. `curate_fma.py` keeps only tracks that pass `license_ok`. | 2026-11-09 |
| Jamendo API | no, not without a deal | API terms: free for non-commercial use only; commercial use needs a quote from licensing@jamendo.com (devportal.jamendo.com/api_terms_of_use) | 2026-11-09 |
| MTG-Jamendo dataset | no | no per-track license field; secondary sources report it as non-commercial | 2026-11-09 |

## Ready-made training datasets (search 2026-11-10)

No released dataset pairs song audio with a piano cover under a commercial-use license.

| Dataset | License (primary source) | Verdict |
|---|---|---|
| Slakh2100 (band audio + aligned MIDI of every instrument, 145 h) | CC BY 4.0 (zenodo.org/records/4599666) | **Needs legal review.** Its MIDI comes from Lakh MIDI: scraped transcriptions of copyrighted songs. |
| OpenScore Lieder (1,200+ scores, no audio) | CC0 (github.com/OpenScore/Lieder) | Usable; public-domain compositions, art-song style |
| PDMX (254k MuseScore scores, no audio) | CC BY 4.0 dataset, scores marked PD/CC0 | Caveat: scraped, and some pop arrangements are falsely marked public domain. Use only genuinely public-domain compositions (`no_license_conflict` subset). |
| CocoChorales (synthetic chorales, stems + MIDI) | CC BY 4.0 | Usable, but contains no piano |
| MusicNet (classical audio + aligned notes) | CC BY 4.0; audio "CC and public domain", no per-file licenses | Caveat |
| POP909 | MIT (code-style license) over arrangements of copyrighted songs | Not usable |
| Pop2Piano, PiCoGen2, Etude, Music2MIDI, PIAST, MulTTiPop | YouTube-sourced, no license, or not released | Not usable |
| Aria-MIDI, MAESTRO, MoisesDB, MedleyDB, MUSDB18, MTG-Jamendo, GigaMIDI, Classical Trio | NC / NC-SA / academic only | Not usable |
| MetaMIDI | no license; must agree not to share or distribute | Not usable |
| HookTheory / SheetSage | data CC BY-NC-SA 3.0; ToS (Oct 2026) forbids scraping and model training | Not usable |

Not verified: MuseScore's terms on scraping and ML (page returned 403), MusicNet's per-recording licenses, and the license terms of the virtual instruments Slakh used to render its audio.

Open items:
- Transkun: an email to the author confirming `2.0.pt` is under the MIT license would close the question.
- Demucs 4-stem: the weights' license is unstated, and the only stated caveat is about training data (covered by the position above).
