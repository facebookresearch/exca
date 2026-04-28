# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from pathlib import Path

import pytest

from . import inflight, registry


def test_registry_operations(tmp_path: Path) -> None:
    reg = inflight.InflightRegistry(tmp_path)
    pid = os.getpid()
    dead_pid = 2**20 + 7

    # Claim, query, re-entrant claim
    claimed = reg.claim(["a", "b", "c"], pid=pid)
    assert set(claimed) == {"a", "b", "c"}
    assert set(reg.get(["a", "b", "c"])) == {"a", "b", "c"}
    assert set(reg.claim(["a", "b", "c"], pid=pid)) == {"a", "b", "c"}

    # Update worker info (post-submission update)
    reg.update_worker_info(["a", "b"], job_id="12345", job_folder="/logs")
    info = reg.get(["a", "b"])
    assert info["a"].job_id == "12345" and info["b"].job_folder == "/logs"

    # Release subset, verify remainder
    reg.release(["a", "b"])
    assert list(reg.get(["a", "b", "c"])) == ["c"]
    reg.release(["c"])
    assert reg.get(["a", "b", "c"]) == {}

    # Dead worker reclaim via claim()
    reg.claim(["x"], pid=dead_pid)
    claimed = reg.claim(["x"], pid=pid)
    assert claimed == ["x"] and reg.get(["x"])["x"].pid == pid

    # Live conflict: cannot steal from a live worker
    reg.claim(["y"], pid=pid)
    other = inflight.InflightRegistry(tmp_path)
    assert other.claim(["y"], pid=dead_pid) == []
    other.close()

    reg.release(["x", "y"])
    reg.close()


def test_inflight_session(tmp_path: Path) -> None:
    # None registry -> yields all UIDs unchanged
    with inflight.inflight_session(None, ["a", "b"]) as claimed:
        assert claimed == ["a", "b"]

    # Normal flow: claims visible during session, released after
    reg = inflight.InflightRegistry(tmp_path)
    with inflight.inflight_session(reg, ["x", "y"]) as claimed:
        assert set(claimed) == {"x", "y"}
        check = inflight.InflightRegistry(tmp_path)
        assert set(check.get(["x", "y"])) == {"x", "y"}
        check.close()
    check2 = inflight.InflightRegistry(tmp_path)
    assert check2.get(["x", "y"]) == {}
    check2.close()

    # Exception path: items still released in finally
    reg2 = inflight.InflightRegistry(tmp_path)
    with pytest.raises(ValueError, match="boom"):
        with inflight.inflight_session(reg2, ["a"]) as claimed:
            assert claimed == ["a"]
            raise ValueError("boom")
    check3 = inflight.InflightRegistry(tmp_path)
    assert check3.get(["a"]) == {}
    check3.close()

    # Local session: marks claims with job_id="local"
    reg_local = inflight.InflightRegistry(tmp_path)
    with inflight.inflight_session(reg_local, ["loc"], local=True) as claimed:
        assert claimed == ["loc"]
        check_loc = inflight.InflightRegistry(tmp_path)
        info = check_loc.get(["loc"])["loc"]
        assert info.job_id == inflight._LOCAL_JOB_ID
        check_loc.close()

    # Nested / re-entrant: inner session must NOT release outer's claim
    outer_reg = inflight.InflightRegistry(tmp_path)
    with inflight.inflight_session(outer_reg, ["z"]) as outer_claimed:
        assert outer_claimed == ["z"]
        inner_reg = inflight.InflightRegistry(tmp_path)
        with inflight.inflight_session(inner_reg, ["z"]) as inner_claimed:
            assert inner_claimed == ["z"]
        check4 = inflight.InflightRegistry(tmp_path)
        assert "z" in check4.get(["z"]), "inner released outer's claim"
        check4.close()
    check5 = inflight.InflightRegistry(tmp_path)
    assert check5.get(["z"]) == {}
    check5.close()


def test_wait_for_inflight(tmp_path: Path) -> None:
    dead_pid = 2**20 + 7

    # Dead worker: wait detects dead PID and reclaims
    reg = inflight.InflightRegistry(tmp_path)
    reg.claim(["stale"], pid=dead_pid)
    reg2 = inflight.InflightRegistry(tmp_path)
    assert reg2.wait_for_inflight(["stale"]) == ["stale"]
    assert reg2.get(["stale"]) == {}
    reg2.close()
    reg.close()

    # Non-Slurm job with fake job_id: must not hang, reclaimed as dead.
    # Regression test for the case where a non-Slurm submitit job
    # (DebugExecutor/LocalExecutor) accidentally gets job_id recorded.
    reg_ns = inflight.InflightRegistry(tmp_path)
    reg_ns.claim(["non_slurm"], pid=dead_pid)
    reg_ns.update_worker_info(["non_slurm"], job_id="99999", job_folder="/nonexistent")
    info = reg_ns.get(["non_slurm"])["non_slurm"]
    assert not info.is_alive(), "fake Slurm job should not appear alive"
    reg_ns2 = inflight.InflightRegistry(tmp_path)
    assert reg_ns2.wait_for_inflight(["non_slurm"]) == ["non_slurm"]
    reg_ns2.close()
    reg_ns.close()

    # Own PID: skipped to prevent self-deadlock
    reg3 = inflight.InflightRegistry(tmp_path)
    reg3.claim(["mine"])
    assert reg3.wait_for_inflight(["mine"]) == []
    assert "mine" in reg3.get(["mine"])
    reg3.release(["mine"])
    reg3.close()


def test_no_job_timeout() -> None:
    """Claim without job_id that exceeds no_job_timeout is treated as dead."""
    recent = inflight.WorkerInfo(pid=os.getpid(), claimed_at=time.time())
    assert recent.is_alive()

    stale = inflight.WorkerInfo(pid=os.getpid(), claimed_at=time.time() - 700)
    assert not stale.is_alive(no_job_timeout=600), "stale PID-only claim should be dead"

    # With Slurm job_id set, the timeout does not apply
    with_slurm = inflight.WorkerInfo(
        pid=os.getpid(),
        job_id="12345",
        job_folder="/nonexistent",
        claimed_at=time.time() - 700,
    )
    assert with_slurm.is_alive(no_job_timeout=600)

    # With local job_id, the timeout does not apply either
    local = inflight.WorkerInfo(
        pid=os.getpid(), job_id="local", claimed_at=time.time() - 700
    )
    assert local.is_alive(no_job_timeout=600)


def test_db_deletion_unblocks_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting inflight.db while a process is waiting should unblock it."""
    alive_pid = os.getpid() + 1  # different PID, will be checked for liveness

    reg = inflight.InflightRegistry(tmp_path)
    reg.claim(["a", "b"], pid=alive_pid)

    # Make the blocker appear alive so wait_for_inflight enters the polling loop
    monkeypatch.setattr(
        inflight.WorkerInfo, "is_alive", lambda self: self.pid == alive_pid
    )

    waiter = inflight.InflightRegistry(tmp_path)
    # Seed the connection so it's cached before deletion
    waiter.get(["a", "b"])

    # Delete the DB — simulates user intervention
    db_path = tmp_path / "inflight.db"
    assert db_path.exists()
    db_path.unlink()

    # Next poll should detect deletion, reconnect to empty DB, and return
    result = waiter.wait_for_inflight(["a", "b"])
    assert result == [], "items should be forgotten (DB gone), not reclaimed"
    waiter.close()
    reg.close()


def test_large_batch_operations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises bulk performance: dedup Slurm waits, transactional release,
    chunked get."""
    dead_pid = 2**20 + 7

    # Slurm wait deduplication: 5 items across 2 jobs → 2 wait() calls
    wait_calls: list[str] = []
    original_wait = inflight.WorkerInfo.wait

    def tracking_wait(self: inflight.WorkerInfo) -> None:
        if self.job_id is not None:
            wait_calls.append(self.job_id)
        original_wait(self)

    monkeypatch.setattr(inflight.WorkerInfo, "wait", tracking_wait)
    reg = inflight.InflightRegistry(tmp_path)
    for uid in ["a", "b", "c", "d", "e"]:
        reg.claim([uid], pid=dead_pid)
    reg.update_worker_info(["a", "b", "c"], job_id="111", job_folder="/nonexistent")
    reg.update_worker_info(["d", "e"], job_id="222", job_folder="/nonexistent")
    waiter = inflight.InflightRegistry(tmp_path)
    waiter.wait_for_inflight(["a", "b", "c", "d", "e"])
    assert sorted(wait_calls) == ["111", "222"]
    waiter.close()
    reg.close()

    # Chunked get + transactional release with > QUERY_BATCH_SIZE items
    reg2 = inflight.InflightRegistry(tmp_path)
    n = registry.QUERY_BATCH_SIZE * 3 + 17
    uids = [f"item_{i}" for i in range(n)]
    reg2.claim(uids)
    assert len(reg2.get(uids)) == n
    assert len(reg2.get(uids + ["missing"])) == n
    reg2.release(uids)
    assert reg2.get(uids) == {}
    reg2.close()


def test_inflight_session_retries_lost_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When another worker grabs an item between wait and claim,
    inflight.inflight_session must re-wait instead of silently skipping."""
    competitor_pid = 2**20 + 13
    wait_calls = 0
    alive_calls = 0
    original_wait = inflight.InflightRegistry.wait_for_inflight
    original_is_alive = inflight.WorkerInfo.is_alive

    def wait_then_inject(
        self: inflight.InflightRegistry, item_uids: list[str]
    ) -> list[str]:
        nonlocal wait_calls
        result = original_wait(self, item_uids)
        wait_calls += 1
        if wait_calls == 1:
            rival = inflight.InflightRegistry(tmp_path)
            rival.claim(["x"], pid=competitor_pid)
            rival.close()
        return result

    def patched_is_alive(self: inflight.WorkerInfo) -> bool:
        nonlocal alive_calls
        if self.pid == competitor_pid:
            alive_calls += 1
            return alive_calls == 1  # alive first check, dead on retry
        return original_is_alive(self)

    monkeypatch.setattr(inflight.InflightRegistry, "wait_for_inflight", wait_then_inject)
    monkeypatch.setattr(inflight.WorkerInfo, "is_alive", patched_is_alive)

    reg = inflight.InflightRegistry(tmp_path)
    with inflight.inflight_session(reg, ["x"]) as claimed:
        assert claimed == ["x"]
    assert wait_calls >= 2, f"expected retry, got {wait_calls} wait calls"
