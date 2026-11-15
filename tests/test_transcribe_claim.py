import os
import time

from pianocover.transcribe_covers import STALE_LOCK_SEC, claim


def test_claim_is_exclusive(tmp_path):
    lock = claim(tmp_path, "a")
    assert lock == tmp_path / "a.lock" and lock.exists()
    assert claim(tmp_path, "a") is None  # held by "another worker"


def test_claim_skips_finished_cover(tmp_path):
    (tmp_path / "a.npy").write_bytes(b"")
    assert claim(tmp_path, "a") is None
    assert not (tmp_path / "a.lock").exists()


def test_stale_lock_is_taken_over(tmp_path):
    lock = tmp_path / "a.lock"
    lock.touch()
    old = time.time() - STALE_LOCK_SEC - 60
    os.utime(lock, (old, old))
    assert claim(tmp_path, "a") == lock
    assert time.time() - lock.stat().st_mtime < 60  # fresh claim, not the stale one
    assert [p.name for p in tmp_path.iterdir()] == ["a.lock"]  # no leftovers


def test_fresh_lock_is_respected(tmp_path):
    (tmp_path / "a.lock").touch()
    assert claim(tmp_path, "a") is None
