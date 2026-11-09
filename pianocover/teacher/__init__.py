"""Teacher pipeline: offline song -> piano arrangement by *reduction* (not regeneration).

Stages (all models load once and stay resident, see models.TeacherModels):
  1. analysis  (GPU): Demucs 4-stem separation, basic-pitch on vocals / bass / other,
                      Transkun on 'other', beat_this beats + downbeats, BTC chords,
                      drum onsets and energy curves.            -> analysis.Analysis
  2. arrange   (CPU, numpy only): reduce the transcribed parts to a playable piano part at
                      one of five difficulty levels.               -> arrange.arrange()
  3. output: MIDI on the song's timeline + a manifest row for pianocover.import_aligned.

Model licenses and where they were checked: docs/MODEL_LICENSES.md.
"""
