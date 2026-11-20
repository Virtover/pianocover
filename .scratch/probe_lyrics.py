import io, re, sys, tarfile, pandas as pd, pretty_midi, mido
sys.path.insert(0, "scripts")
from organ_synth import OLD, MODERN, progs
df = pd.read_csv("/data/pdmx/PDMX.csv", low_memory=False)
df = df[df["subset:no_license_conflict"] & df["subset:deduplicated"] & df["mid"].notna()
        & df["license"].isin(["publicdomain", "cc-zero"]) & df.has_lyrics & (df.n_lyrics >= 40)
        & df["song_length.seconds"].between(40, 400)]
txt = df.title.fillna("") + " | " + df.composer_name.fillna("") + " | " + df.artist_name.fillna("")
HYMN = re.compile(r"(?i)hymn|chorale|choral|traditional|trad\.|folk|carol|psalm|schubert|schumann|brahms|mozart|haydn|handel|händel|purcell|dowland|foster|sullivan|wesley|watts|lowell mason|bradbury|doane|sankey|bliss|crosby|monk|dykes|barnby|stainer|arne|beethoven|mendelssohn|bach|grieg|faur|gounod|wolf|silcher|reichardt")
df = df[(txt.str.contains(OLD) | txt.str.contains(HYMN)) & ~txt.str.contains(MODERN)]
print("candidates", len(df))
want = {m.lstrip("./"): i for i, m in zip(df.index, df["mid"])}
n = 0; stats = []
with tarfile.open("/data/pdmx/mid.tar.gz") as tar:
    for m in tar:
        name = m.name.lstrip("./")
        if not (m.isfile() and name in want): continue
        data = tar.extractfile(m).read()
        try:
            mf = mido.MidiFile(file=io.BytesIO(data)); pm = pretty_midi.PrettyMIDI(io.BytesIO(data))
        except Exception: continue
        lyr_tracks = [(k, sum(1 for e in t if e.type == "lyrics")) for k, t in enumerate(mf.tracks)]
        lyr_tracks = [x for x in lyr_tracks if x[1] > 0]
        r = df.loc[want[name]]
        stats.append((want[name], r.title, r.composer_name, r.tracks, [(i.name, i.program, len(i.notes)) for i in pm.instruments], lyr_tracks, len(pm.lyrics), r["song_length.seconds"], r.n_views))
pd.to_pickle(stats, "/data/meta/lyrics_probe.pkl")
has = [s for s in stats if s[6] > 20]
print("parsed", len(stats), "with lyric events", len(has))
for s in sorted(has, key=lambda s: -s[8])[:40]: print(s[:7])
