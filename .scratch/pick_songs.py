import re, pandas as pd, collections
st = pd.read_pickle("/data/meta/lyrics_probe.pkl")
VOX = re.compile(r"(?i)voice|vocal|voix|stimme|singstimme|soprano|sopran|alto|tenor|bass\b|baritone|bariton|mezzo|melody|melodie|canto|singer|choir|chor|lead|s\.|a\.|t\.|b\.")
rows = []
for idx, title, comp, tracks, ins, _, _, secs, views in st:
    vox = [i for i in ins if VOX.search(i[0] or "") and i[1] not in range(0, 8) or i[1] in (52, 53, 54, 85)]
    acc = [i for i in ins if i not in vox and i[2] > 0]
    if not vox or not acc: continue
    kind = "solo" if len(vox) == 1 else "satb" if len(vox) >= 3 else "duet"
    rows.append((views, idx, title, comp, kind, [(v[0], v[1]) for v in vox], [(a[0], a[1]) for a in acc], int(secs)))
print(len(rows), collections.Counter(r[4] for r in rows))
for r in sorted(rows, reverse=True)[:60]: print(r)
pd.to_pickle(rows, "/data/meta/vocal_candidates.pkl")
