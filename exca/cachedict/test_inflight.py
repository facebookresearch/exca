# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
import time
from pathlib import Path

import pytest

from . import inflight, registry

_DEAD_PID = 2**20 + 7
_DEAD_LIVENESS = inflight.Pid(host=socket.gethostname(), pid=_DEAD_PID)
_DEAD = inflight.WorkerInfo(token="dead", liveness=_DEAD_LIVENESS)


def test_inflight_lifecycle(tmp_path: Path) -> None:
    reg = inflight.InflightRegistry(tmp_path)
    dead = inflight.InflightRegistry(tmp_path, worker=_DEAD)

    # Claim, query
    claimed = reg.claim(["a", "b", "c"])
    assert set(claimed) == {"a", "b", "c"}
    assert set(reg.get(["a", "b", "c"])) == {"a", "b", "c"}

    # Update worker info (post-submission update)
    job = inflight.Slurm(job_id="12345", folder="/logs")
    reg.update_liveness(["a", "b"], job)
    info = reg.get(["a", "b"])
    assert info["a"].liveness == job

    # Release subset, verify remainder
    reg.release(["a", "b"])
    assert list(reg.get(["a", "b", "c"])) == ["c"]
    reg.release(["c"])
    assert reg.get(["a", "b", "c"]) == {}

    # Dead worker reclaim via claim()
    dead.claim(["x"])
    claimed = reg.claim(["x"])
    assert claimed == ["x"] and reg.get(["x"])["x"].token == reg.worker.token

    # Live conflict: cannot steal from a live worker
    reg.claim(["y"])
    assert dead.claim(["y"]) == []
    dead.close()

    reg.release(["x", "y"])
    reg.close()


def test_inflight_session(tmp_path: Path) -> None:
    def seen(uids: list[str]) -> dict[str, inflight.WorkerInfo]:
        with inflight.InflightRegistry(tmp_path) as r:
            return r.get(uids)

    def fresh() -> inflight.InflightRegistry:
        return inflight.InflightRegistry(tmp_path)

    # None: yields uids unchanged.
    with inflight.inflight_session(None, ["a", "b"]) as claimed:
        assert claimed.uids == ("a", "b")
        assert not claimed.waited

    # Normal: claims visible during session, released after.
    with inflight.inflight_session(fresh(), ["x", "y"]) as claimed:
        assert set(claimed.uids) == {"x", "y"}
        assert not claimed.waited
        info = seen(["x", "y"])
        assert set(info) == {"x", "y"}
        assert info["x"].liveness == inflight.Pid.here()
    assert seen(["x", "y"]) == {}

    # Exception: items still released in finally.
    with pytest.raises(ValueError, match="boom"):
        with inflight.inflight_session(fresh(), ["a"]) as claimed:
            assert claimed.uids == ("a",)
            raise ValueError("boom")
    assert seen(["a"]) == {}

    # Handed off: released by the work, not the session.
    with inflight.inflight_session(fresh(), ["h", "k"]) as claimed:
        claimed.hand_off(["h"])
    assert list(seen(["h", "k"])) == ["h"]


def test_wait_for_inflight(tmp_path: Path) -> None:
    # Dead worker: wait detects dead PID and reclaims
    reg = inflight.InflightRegistry(tmp_path, worker=_DEAD)
    reg.claim(["stale"])
    reg2 = inflight.InflightRegistry(tmp_path)
    reg2.wait_for_inflight(["stale"])
    assert reg2.get(["stale"]) == {}, "dead worker's item should be reclaimed"
    reg2.close()
    reg.close()


_ME = inflight.Pid.here()
_ELSEWHERE = inflight.Pid(host="elsewhere", pid=os.getpid())
_FAKE_JOB = inflight.Slurm(job_id="99999", folder="/nonexistent")


@pytest.mark.parametrize(
    "liveness,age,alive",
    [
        (_ELSEWHERE, 0, True),  # other host: alive until timeout
        (_ELSEWHERE, 700, False),
        (_ME, 700, True),  # same host: pid, no timeout
        (_DEAD_LIVENESS, 0, False),
        (_FAKE_JOB, 700, False),  # unreconstructable job
    ],
)
def test_is_alive(liveness: inflight.Liveness, age: float, alive: bool) -> None:
    worker = inflight.WorkerInfo(
        token="t", liveness=liveness, claimed_at=time.time() - age
    )
    assert worker.is_alive(no_job_timeout=600) is alive


def test_db_deletion_unblocks_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting inflight.db while a process is waiting should unblock it."""
    reg = inflight.InflightRegistry(tmp_path, inflight.WorkerInfo("blocker", _ME))
    reg.claim(["a", "b"])

    # Make the blocker appear alive so wait_for_inflight enters the polling loop
    monkeypatch.setattr(
        inflight.WorkerInfo, "is_alive", lambda self: self.token == "blocker"
    )

    waiter = inflight.InflightRegistry(tmp_path)
    # Seed the connection so it's cached before deletion
    waiter.get(["a", "b"])

    # Delete the DB — simulates user intervention
    db_path = tmp_path / "inflight.db"
    assert db_path.exists()
    db_path.unlink()

    # Next poll should detect deletion, reconnect to empty DB, and return
    waiter.wait_for_inflight(["a", "b"])  # must return, not hang
    assert waiter.get(["a", "b"]) == {}, "items forgotten after DB deletion"
    waiter.close()
    reg.close()


def test_large_batch_operations(tmp_path: Path) -> None:
    reg = inflight.InflightRegistry(tmp_path)
    n = registry.QUERY_BATCH_SIZE * 3 + 17
    uids = [f"item_{i}" for i in range(n)]
    reg.claim(uids)
    assert len(reg.get(uids)) == n
    assert len(reg.get(uids + ["missing"])) == n
    reg.release(uids)
    assert reg.get(uids) == {}
    reg.close()


def test_inflight_session_retries_lost_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When another worker grabs an item between wait and claim,
    inflight.inflight_session must re-wait instead of silently skipping."""
    wait_calls = 0
    alive_calls = 0
    original_wait = inflight.InflightRegistry.wait_for_inflight
    original_is_alive = inflight.WorkerInfo.is_alive

    def wait_then_inject(self: inflight.InflightRegistry, item_uids: list[str]) -> None:
        nonlocal wait_calls
        original_wait(self, item_uids)
        wait_calls += 1
        if wait_calls == 1:
            worker = inflight.WorkerInfo("rival", _ME)
            rival = inflight.InflightRegistry(tmp_path, worker)
            rival.claim(["x"])
            rival.close()

    def patched_is_alive(self: inflight.WorkerInfo) -> bool:
        nonlocal alive_calls
        if self.token == "rival":
            alive_calls += 1
            return alive_calls == 1  # alive first check, dead on retry
        return original_is_alive(self)

    monkeypatch.setattr(inflight.InflightRegistry, "wait_for_inflight", wait_then_inject)
    monkeypatch.setattr(inflight.WorkerInfo, "is_alive", patched_is_alive)

    reg = inflight.InflightRegistry(tmp_path)
    with inflight.inflight_session(reg, ["x"]) as claimed:
        assert claimed.uids == ("x",)
        assert claimed.waited
    assert wait_calls >= 2, f"expected retry, got {wait_calls} wait calls"
