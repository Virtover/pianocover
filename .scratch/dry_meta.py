import io, json, random, sys, tarfile, pandas as pd, pretty_midi
sys.path.insert(0, "scripts")
import vocal_synth as vs
ps = json.load(open("/data/ext/soulx/soulxsinger/utils/phoneme/phone_set.json"))
df = pd.read_csv("/data/pdmx/PDMX.csv", low_memory=False)
ids = [40679, 190160, 204049, 234565, 227929]
want = {df.loc[i, "mid"].lstrip("./"): i for i in ids}
with tarfile.open("/data/pdmx/mid.tar.gz") as t:
    mids = {want[m.name.lstrip("./")]: t.extractfile(m).read() for m in t if m.name.lstrip("./") in want}
lyr = vs.Lyrics(random.Random(0))
for i in ids:
    pm = pretty_midi.PrettyMIDI(io.BytesIO(mids[i]))
    vox = [x for x in pm.instruments if vs.is_vocal(x) and x.notes]
    print(i, df.loc[i, "title"], [(x.name.strip("\x00"), len(x.notes)) for x in vox], "acc", [(x.name.strip("\x00"), x.program) for x in pm.instruments if x not in vox])
    for v in vox:
        ln = vs.mono_line(v.notes); m = vs.line_meta(ln, lyr)
        bad = {p for s in m for w in s["phoneme"].split() if w != "<SP>" for p in ["en_" + q for q in w[3:].split("-")] if p not in ps}
        print("   ", len(ln), "notes", len(m), "segs", "max seg s", max((s["time"][1]-s["time"][0])/1000 for s in m), "bad", bad)
print(m[0]["text"][:200])
