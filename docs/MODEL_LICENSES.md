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

## NonCommercial (NC) data: owner's risk decision, 2026-11-11

The license gate accepts NC-licensed data only with an explicit `--allow-noncommercial` flag
(`import_aligned`, `merge_datasets`, `teacher.pipeline`, `teacher.curate_fma`). NC-SA data
needs `--allow-sharealike` as well. ND and unknown licenses stay refused. This is the owner's
decision, based on background research and not on legal advice:

- Polish Copyright Act art. 26(3) (Dz.U. 2024 poz. 1254): reproducing published works for text
  and data mining is allowed unless the rights holder expressly reserved it (machine-readable
  for openly accessible works). No commercial-use limit, no mention of AI.
- Creative Commons FAQ: a CC license applies "only when a particular use would otherwise not
  be permitted by copyright".
- Unsettled: whether an NC license is itself such a reservation (no ruling found). The
  exception is EU / Polish law; US law depends on fair use.

Conditions to follow for every NC source:
1. Download only from the official source; check its terms and robots.txt for a rights
   reservation and note the result in the log below.
2. Check the dataset's own terms for restrictions beyond the CC license.
3. Do not redistribute the data; keep trained weights on the server.
4. Delete raw copies when training is finished (art. 26(3) allows keeping them only as long
   as needed).

Still excluded: unlicensed YouTube-sourced pair lists (rights holders and collecting
societies have reserved their rights; lawful access is doubtful).

### NC source log
| Date | Dataset | Official source | License | Reservation check | Used for |
|---|---|---|---|---|---|
| (none yet) | | | | | |

## Policy update, 2026-11-12 (owner's decision)

YouTube-sourced audio and covers are allowed when the underlying rights holders have not
reserved their rights. Sony, Universal and Warner catalogues stay excluded. This replaces the
"Still excluded: unlicensed YouTube-sourced pair lists" line above for non-reserved content.
A third-party model's training data is not a blocker; only the license on its code and weights is.

## Ready-made audio -> piano-cover models (checked 2026-11-12)

| Model | Code | Weights | Result |
|---|---|---|---|
| AMT-APC (github.com/310hz/amt-apc) | MIT (LICENSE) | `apc.pth` in the repo's GitHub release, no separate statement | Runs: 5-9 s per song, 2.3 GB peak VRAM (measured, 3060 Ti, 10 FMA CC tracks; one track gave 9 notes). Owner listened: quality not good enough. Rejected on quality. |
| Etude (github.com/Xiugapurin/Etude) | MIT | CC BY-NC-SA 4.0, LICENSE says no commercial use, contact owner | Best published listening score (3.50 vs 3.92 human, Etude paper). Owner judged demos poor. Rejected (license and quality). README asks for 16 GB VRAM (not tested). |
| PiCoGen2 (tanchihpin0517/PiCoGen, v2 branch) | MIT | CC BY-NC-SA 3.0 (HookTheory-derived), per subagent reading of LICENSE | Rejected (license; scored below Etude). |
| Pop2Piano (sweetcocoa/pop2piano) | no LICENSE file (GitHub API: none) | HF card has no license field | No license granted; older and weaker than the above. Rejected. |

## Piano corpora checked 2026-11-12

| Dataset | Terms (primary source) | Verdict |
|---|---|---|
| Gangqinpu v1.0.0 (github.com/wei-zeng98/gangqinpu, HF Wayne42/gangqinpu): 18,786 piano score-performance pairs, 1,007 h | CC BY-NC-SA 4.0; the HF page requires accepting the license and confirming non-commercial use before download | Blocked by the gate: downloading needs a non-commercial-use confirmation. Piano only, no band audio. |
| PIAST (github.com/Hayeonbang/PIAST): 9,673 piano tracks, YouTube IDs + transcribed MIDI + text | README: CC BY-NC 4.0 (badge says MIT) | Usable under the NC decision as a piano-only prior. Not paired with band audio. |

Added 2026-11-13: hFT-Transformer checkpoints are CC BY-NC-SA 4.0 (github.com/sony/hFT-Transformer README, "License"). AMT-APC's `apc.pth` is fine-tuned from an hFT-Transformer checkpoint, so the MIT label on the AMT-APC repo does not settle the weights' license. Retraining the AMT-APC recipe would need a base whose weights are not NC.

## Pop2Piano pair list: rights filter (2026-11-12/13)

Rule (owner's decision): exclude a song if its recording is released or distributed by Sony,
Universal or Warner, any writer is published by Sony Music Publishing, UMPG or Warner Chappell,
or any writer belongs to SACEM, GEMA, STIM or ZAiKS (published opt-outs, seen in press coverage).
Data: `data/meta/` (not in git). Titles via YouTube oEmbed (`scripts/probe_pair_metadata.py`).

- Korean songs (2,138): writers and labels from genie.co.kr (`scripts/genie_rights_lookup.py`).
  Latin-script writer names were classed Korean / not / unsure by an LLM pass (rough).
  Result: 996 keep (1,534 pairs), 447 exclude, 284 writer not recognised, 386 unresolved,
  25 distributed by a major. File: `p2p_rights_kr_final.jsonl`.
- Japanese/Chinese songs (125): MusicBrainz, mostly unresolved (8 keep).
- Western songs (1,385): MusicBrainz (`scripts/mb_rights_lookup.py`) plus an LLM check of the
  196 survivors, since MusicBrainz often lists only compilation labels. 4 keep.
- 3 of 30 sampled covers were private, blocked or removed; expect further loss when downloading.

## Content filter (owner's decision, 2026-11-14)

Not a license rule. Dropped: K-pop idol acts (per artist), and songs whose lyrics or title are
demonic or occult (devils, Satan, witchcraft, shamanic rites, spirit invocation, curses), or that
glorify or immerse the listener in ghosts, goblins, gumiho or zombies. Kept: supernatural themes
used only as a backdrop, e.g. love songs from the Goblin or Hotel Del Luna soundtracks, and
"feels like hell" as a metaphor. Lyrics come from melon.com (`scripts/lyrics_scan.py`, keyword
hits reviewed by hand). Dropped so far: 안예은 - 창귀, 창모 - METEOR, and one cover from a
channel named "PIANO마녀".
