"""Export what the teacher pipeline HEARD, as multi-track MIDI, to judge the inputs by ear.

  python scripts/export_raw_parts.py /data/teacher_proto /out/dir

One <id>_raw_parts.mid per song, on the song's timeline, with separate tracks you can mute:
  1 Melody   CREPE notes from the vocal stem, on a sustaining voice patch (GM Voice Oohs)
  2 Bass     basic-pitch on the bass stem
  3 Other    basic-pitch on the 'other' stem (guitars, keys, synths)
  4 Other(TK) Transkun on the 'other' stem
  5 Chords   detected chords (after the chord check) as held block chords around C4
  6 Drums    detected kick (36) / snare (38) / hi-hat (42) onsets

Dynamics: note velocity = half the stem's own loudness at the note, half the song's
intensity curve (Song.loud_at: mix level, drum activity, instrument density, vocal height
and loudness), so climaxes are loud on every track. The melody also gets continuous
expression (CC11) from the vocal stem's loudness, so held notes swell and fade as sung.
"""
import glob
import os
import sys

import numpy as np
import pretty_midi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pianocover.teacher.analysis import FRAME_HOP, Analysis  # noqa: E402
from pianocover.teacher.arrange import Song, _notes  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "/data/teacher_proto"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "raw_parts")
os.makedirs(out, exist_ok=True)


def stem_level(rms, window_db=35.0, k=7):
    """Stem loudness 0..1 over the `window_db` below its own peak."""
    db = 20 * np.log10(np.asarray(rms, np.float64) + 1e-5)
    db = np.convolve(np.pad(db, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return np.clip((db - (db.max() - window_db)) / window_db, 0, 1)  # headroom up to the true peak


def peak_at(song, t):
    return float(song._peak[int(np.clip(t / FRAME_HOP, 0, len(song._peak) - 1))])


def track(name, ev, program, song, level=None, floor=22, expression=False):
    inst = pretty_midi.Instrument(program=program, name=name)
    notes = _notes(ev)
    # voice: singers push PITCH more than volume at a climax (the stem is compressed), so
    # the height of each note in the singer's range adds to its strength
    if expression and len(notes) > 4:
        p10, pmax = np.percentile(notes[:, 2], 10), notes[:, 2].max()
        height = np.zeros(len(song._peak))
        for s, e, p, v in notes:
            height[int(s / FRAME_HOP):int(e / FRAME_HOP) + 2] = np.clip((p - p10) / max(pmax - p10, 1), 0, 1)
    else:
        height = None

    def strength(t):
        k = int(np.clip(t / FRAME_HOP, 0, len(song._peak) - 1))
        i = float(song._peak[k])
        if level is None:
            return i
        lv = level[min(k, len(level) - 1)]
        return 0.5 * lv + 0.5 * i if height is None else 0.35 * lv + 0.35 * i + 0.30 * height[k]

    for s, e, p, v in notes:
        if e <= s:
            continue
        x = strength(s)
        inst.notes.append(pretty_midi.Note(int(np.clip(floor + (127 - floor) * x ** 1.2, 1, 127)), int(p),
                                           float(s), float(e)))
    if expression and level is not None:
        last = -1
        for k, lv in enumerate(level):
            val = int(np.clip(30 + 97 * strength(k * FRAME_HOP) ** 1.1, 1, 127))
            if abs(val - last) >= 2:
                inst.control_changes.append(pretty_midi.ControlChange(11, val, k * FRAME_HOP))
                last = val
    return inst


def add_voice_detail(inst, curve, song, hop=0.02, bend_range=2):
    """Play the sung pitch itself: pitch bends (slides, vibrato, scoops) around each note and
    20 ms expression from the vocal loudness. Uses the default +-2 semitone bend range, so it
    plays right in players that ignore the bend-range message (RPN 0 is still sent)."""
    if len(curve) == 0:
        return
    f0, db = curve[:, 0], curve[:, 1]
    for num, val in ((101, 0), (100, 0), (6, bend_range), (38, 0)):
        inst.control_changes.append(pretty_midi.ControlChange(num, val, 0.0))
    inst.control_changes = [c for c in inst.control_changes if c.number != 11]
    last_b = None
    for n in sorted(inst.notes, key=lambda x: x.start):
        a, b = int(n.start / hop), min(int(n.end / hop) + 1, len(f0))
        inst.pitch_bends.append(pretty_midi.PitchBend(0, max(n.start - 0.001, 0.0)))
        last_b = 0
        for k in range(a, b):
            if not np.isfinite(f0[k]):
                continue
            val = int(np.clip((f0[k] - n.pitch) / bend_range, -1, 1) * 8191)
            if abs(val - last_b) >= 24:
                inst.pitch_bends.append(pretty_midi.PitchBend(val, k * hop))
                last_b = val
    last = -1
    for k in range(len(db)):
        lv = np.clip((db[k] + 35.0) / 35.0, 0, 1)  # the stem loudness, unsmoothed
        i = float(song._peak[int(np.clip(k * hop / FRAME_HOP, 0, len(song._peak) - 1))])
        val = int(np.clip(25 + 102 * (0.7 * lv + 0.3 * i) ** 1.1, 1, 127))
        if abs(val - last) >= 2:
            inst.control_changes.append(pretty_midi.ControlChange(11, val, k * hop))
            last = val
    inst.control_changes.sort(key=lambda c: c.time)


def effort_curve(a, hop=0.02):
    """Vocal effort 0..1 per 20 ms frame (p10..p98 of the sung frames), lightly smoothed."""
    eff, curve = np.asarray(getattr(a, "vocal_effort", np.zeros(0))), a.vocal_curve
    if len(eff) == 0 or len(curve) == 0:
        return None
    sung = np.isfinite(curve[:len(eff), 0])
    if sung.sum() < 20:
        return None
    lo, hi = np.percentile(eff[:len(sung)][sung], [10, 98])
    e = np.clip((eff - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.convolve(np.pad(e, 3, mode="edge"), np.ones(7) / 7, mode="valid")


def power_layer(inst, effort, hop=0.02):
    """Second voice layer (open 'Choir Aahs') that fades in with vocal effort, while the soft
    'Oohs' layer backs off: the belted, pushed quality that pitch and loudness cannot carry."""
    import copy

    layer = copy.deepcopy(inst)
    layer.program, layer.name = 52, "1b Melody power layer (vocal effort)"

    def e_at(t):
        return float(effort[int(np.clip(t / hop, 0, len(effort) - 1))])

    for c in inst.control_changes:
        if c.number == 11:
            c.value = int(np.clip(c.value * (1 - 0.45 * e_at(c.time)), 1, 127))
    for c in layer.control_changes:
        if c.number == 11:
            c.value = int(np.clip(c.value * min(1.0, 0.1 + 1.15 * e_at(c.time) ** 1.5), 1, 127))
    for n in layer.notes:
        n.velocity = int(np.clip(n.velocity * (0.6 + 0.5 * e_at(n.start)), 1, 127))
    return layer


for path in sorted(glob.glob(os.path.join(root, "analysis", "*.npz"))):
    sid = os.path.basename(path)[:-4]
    a = Analysis.load(path)
    song = Song(a)
    chords = []
    for s, e, root_pc, pcs, bass_pc in song.chords:
        if root_pc is None:
            continue
        for pc in pcs:
            chords.append([s, e, 60 + (pc - 60) % 12, 50])
    lv_v, lv_o, lv_m = stem_level(a.vocal_rms), stem_level(a.other_rms), stem_level(a.mix_rms)
    pm = pretty_midi.PrettyMIDI()
    pm.instruments += [
        track("1 Melody (CREPE)", a.vocal_f0 if len(a.vocal_f0) else a.vocal, 53, song, lv_v, floor=45,
              expression=True),
        track("2 Bass (basic-pitch)", a.bass, 33, song, lv_m),
        track("3 Other (basic-pitch)", a.other, 25, song, lv_o),
        track("4 Other (Transkun)", a.other_piano, 0, song, lv_o),
        track("5 Chords (detected)", np.array(chords).reshape(-1, 4), 48, song, None, floor=15),
    ]
    add_voice_detail(pm.instruments[0], getattr(a, "vocal_curve", np.zeros((0, 2))), song)
    eff = effort_curve(a)
    if eff is not None:
        pm.instruments.insert(1, power_layer(pm.instruments[0], eff))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="6 Drums (kick/snare/hat)")
    for t, kind, strength in np.asarray(a.drum_onsets).reshape(-1, 3):
        note = {0: 36, 1: 38, 2: 42}.get(int(round(kind)), 38)
        x = 0.5 * min(strength, 1.0) + 0.5 * peak_at(song, t)
        drums.notes.append(pretty_midi.Note(int(np.clip(25 + 102 * x, 1, 127)), note, float(t), float(t) + 0.1))
    pm.instruments.append(drums)
    pm.write(os.path.join(out, f"{sid}_raw_parts.mid"))
    print("wrote", sid)
