# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import stat
from pathlib import Path

import pytest

from .inflight import InflightRegistry, inflight_session


def test_claim_release_reentrant(tmp_path: Path) -> None:
    reg = InflightRegistry(tmp_path)
    pid = os.getpid()
    uids = ["a", "b", "c"]

    claimed = reg.claim(uids, pid=pid)
    assert set(claimed) == set(uids)
    assert set(reg.get_inflight(uids)) == set(uids)

    # Re-entrant: same PID claims the same items -> returns as ours
    claimed2 = reg.claim(uids, pid=pid)
    assert set(claimed2) == set(uids)

    # Update worker info after claim (simulates post-submission update)
    reg.update_worker_info(["a", "b"], job_id="12345", job_folder="/logs")
    info = reg.get_inflight(["a", "b"])
    assert info["a"].job_id == "12345"
    assert info["a"].job_folder == "/logs"
    assert info["b"].job_id == "12345"

    reg.release(["a", "b"])
    remaining = reg.get_inflight(uids)
    assert list(remaining) == ["c"]

    reg.release(["c"])
    assert reg.get_inflight(uids) == {}
    reg.close()


def test_claim_conflict_and_dead_worker(tmp_path: Path) -> None:
    reg = InflightRegistry(tmp_path)
    my_pid = os.getpid()
    dead_pid = 2**20 + 7  # very high PID, unlikely to exist

    # Dead worker: claim from dead PID, then reclaim from live caller
    reg.claim(["x"], pid=dead_pid)
    assert "x" in reg.get_inflight(["x"])
    claimed = reg.claim(["x"], pid=my_pid)
    assert claimed == ["x"]
    assert reg.get_inflight(["x"])["x"].pid == my_pid

    # Live conflict: claim from live PID, another PID cannot steal it
    reg.claim(["y"], pid=my_pid)
    other_reg = InflightRegistry(tmp_path)
    claimed = other_reg.claim(["y"], pid=dead_pid)
    assert claimed == []
    other_reg.close()

    reg.release(["x", "y"])
    reg.close()


@pytest.mark.parametrize("break_mode", ["corrupt", "permissions"])
def test_graceful_degradation(
    tmp_path: Path, break_mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt or inaccessible DB -> logged warning, fallback values, no crash.
    After degradation, the DB auto-recovers on the next access.
    """
    reg = InflightRegistry(tmp_path)
    reg.claim(["warmup"], pid=os.getpid())
    reg.release(["warmup"])
    reg.close()

    db_path = tmp_path / "inflight.db"
    assert db_path.exists()

    if break_mode == "corrupt":
        db_path.write_bytes(b"NOT A SQLITE DB")
    elif break_mode == "permissions":
        db_path.chmod(0o000)

    reg2 = InflightRegistry(tmp_path)
    with caplog.at_level(logging.WARNING):
        claimed = reg2.claim(["a", "b"], pid=os.getpid())
        assert set(claimed) == {"a", "b"}  # fallback: claim all
        result = reg2.get_inflight(["a"])
        assert result == {}  # fallback: empty
        reg2.release(["a"])  # should not raise
    reg2.close()

    if break_mode == "permissions":
        db_path.chmod(stat.S_IRWXU)

    assert "Inflight registry" in caplog.text or "unavailable" in caplog.text

    # Auto-recovery: _try_reset deleted the corrupt DB, next access recreates it
    reg3 = InflightRegistry(tmp_path)
    claimed = reg3.claim(["recovered"], pid=os.getpid())
    assert claimed == ["recovered"]
    assert "recovered" in reg3.get_inflight(["recovered"])
    reg3.release(["recovered"])
    reg3.close()


def test_db_deleted_mid_session(tmp_path: Path) -> None:
    """Deleting the DB mid-session doesn't crash — SQLite recreates it."""
    reg = InflightRegistry(tmp_path)
    reg.claim(["a"], pid=os.getpid())

    db_path = tmp_path / "inflight.db"
    db_path.unlink()

    # Release on a deleted DB should not crash
    reg.release(["a"])
    reg.close()


def test_inflight_session(tmp_path: Path) -> None:
    # None registry -> yields all UIDs unchanged
    with inflight_session(None, ["a", "b"]) as claimed:
        assert claimed == ["a", "b"]

    # Normal flow: claims visible during session, released after
    reg = InflightRegistry(tmp_path)
    with inflight_session(reg, ["x", "y"]) as claimed:
        assert set(claimed) == {"x", "y"}
        reg2 = InflightRegistry(tmp_path)
        assert set(reg2.get_inflight(["x", "y"])) == {"x", "y"}
        reg2.close()
    reg3 = InflightRegistry(tmp_path)
    assert reg3.get_inflight(["x", "y"]) == {}
    reg3.close()

    # Exception path: items still released in finally
    reg4 = InflightRegistry(tmp_path)
    with pytest.raises(ValueError, match="boom"):
        with inflight_session(reg4, ["a"]) as claimed:
            assert claimed == ["a"]
            raise ValueError("boom")
    reg5 = InflightRegistry(tmp_path)
    assert reg5.get_inflight(["a"]) == {}
    reg5.close()


def test_wait_for_inflight(tmp_path: Path) -> None:
    dead_pid = 2**20 + 7  # very high PID, unlikely to exist

    # Dead worker: wait detects dead PID and reclaims the item
    reg = InflightRegistry(tmp_path)
    reg.claim(["stale"], pid=dead_pid)
    reg2 = InflightRegistry(tmp_path)
    reclaimed = reg2.wait_for_inflight(["stale"])
    assert reclaimed == ["stale"]
    assert reg2.get_inflight(["stale"]) == {}
    reg2.close()
    reg.close()

    # Own PID: skipped to prevent self-deadlock (returns immediately)
    reg3 = InflightRegistry(tmp_path)
    pid = os.getpid()
    reg3.claim(["mine"], pid=pid)
    reclaimed = reg3.wait_for_inflight(["mine"])
    assert reclaimed == []
    assert "mine" in reg3.get_inflight(["mine"])
    reg3.release(["mine"])
    reg3.close()
