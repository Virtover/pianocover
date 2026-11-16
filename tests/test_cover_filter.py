import csv
import pickle

import numpy as np

from pianocover import build_dataset
from pianocover.build_dataset import scan_status
from pianocover.scan_covers import timbre_similarity


def test_scan_status():
    assert scan_status(None, 0.03, 0.9) is None  # not scanned: keep
    assert scan_status({"cover_vocal": "0.000", "timbre": "0.76"}, 0.03, 0.9) is None
    assert scan_status({"cover_vocal": "0.065", "timbre": "0.2"}, 0.03, 0.9) == "cover_vocals"
    assert scan_status({"cover_vocal": "0.0", "timbre": "0.95"}, 0.03, 0.9) == "cover_same_as_original"


def test_timbre_similarity_identical_vs_different():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=38).astype(np.float32), rng.normal(size=38).astype(np.float32)
    stats = {"pop/x": a, "cover/same": a.copy(), "cover/other": b}
    mu = np.zeros(19, np.float32)
    assert timbre_similarity(stats, "pop/x", "cover/same", mu) > 0.999
    assert timbre_similarity(stats, "pop/x", "cover/other", mu) < 0.9


def _item(pop, cov):
    return {"pop_id": pop, "piano_id": cov, "status": "ok", "quality": 0.7, "transpose": 0,
            "key_margin": 0.3, "bad_fraction": 0.0, "n_notes": 300, "duration": 100.0,
            "raw_style": np.arange(8, dtype=np.float64) + len(cov), "events": np.zeros((1, 4))}


def test_refilter_drops_scanned_pairs_and_keeps_report(tmp_path):
    data, out = tmp_path / "raw", tmp_path / "ds"
    data.mkdir()
    out.mkdir()
    items = [_item("p1", "good1"), _item("p2", "good22"), _item("p3", "vocal"), _item("p4", "reup")]
    with (out / "train.pkl").open("wb") as f:
        pickle.dump({"data_root": str(data), "items": items}, f)
    with (out / "val.pkl").open("wb") as f:
        pickle.dump({"data_root": str(data), "items": []}, f)
    with (out / "report.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pop_id", "piano_id", "status", "quality"])
        w.writerows([[r["pop_id"], r["piano_id"], "ok", "0.7"] for r in items] + [["p5", "x", "low_quality", "0.4"]])
    with (data / "cover_scan.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pop_id", "piano_id", "cover_vocal", "timbre"])
        w.writerows([["p1", "good1", "0", "0.1"], ["p3", "vocal", "0.2", "0.1"], ["p4", "reup", "0", "0.99"]])

    build_dataset.main(["--data", str(data), "--out", str(out), "--refilter", "--val-fraction", "0"])

    kept = pickle.load((out / "train.pkl").open("rb"))["items"]
    assert sorted(r["piano_id"] for r in kept) == ["good1", "good22"]  # p2 unscanned: kept
    status = {r["piano_id"]: r["status"] for r in csv.DictReader((out / "report.csv").open())}
    assert status == {"good1": "ok", "good22": "ok", "vocal": "cover_vocals",
                      "reup": "cover_same_as_original", "x": "low_quality"}
    assert (out / "style_stats.json").exists()
