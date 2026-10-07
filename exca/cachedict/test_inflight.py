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
import submitit

from . import inflight, registry

_DEAD_PID = 2**20 + 7
_DEAD = inflight.WorkerInfo(pid=_DEAD_PID, token="dead", host=socket.gethostname())


def test_inflight_lifecycle(tmp_path: Path) -> None:
    reg = inflight.InflightRegistry(tmp_path)
    dead = inflight.InflightRegistry(tmp_path, worker=_DEAD)

    # Claim, query
    claimed = reg.claim(["a", "b", "c"])
    assert set(claimed) == {"a", "b", "c"}
    assert set(reg.get(["a", "b", "c"])) == {"a", "b", "c"}

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
        assert (info["x"].host, info["x"].pid) == (socket.gethostname(), os.getpid())
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


@pytest.mark.parametrize(
    "host,pid,job_id,age,alive",
    [
        ("elsewhere", os.getpid(), None, 0, True),  # other host: alive until timeout
        ("elsewhere", os.getpid(), None, 700, False),
        ("here", os.getpid(), None, 700, True),  # same host: pid, no timeout
        ("here", _DEAD_PID, None, 0, False),
        ("here", _DEAD_PID, "99999", 0, False),  # fake job id must not hang
        ("elsewhere", os.getpid(), "12345", 700, False),  # unreconstructable job
    ],
)
def test_is_alive(
    host: str, pid: int, job_id: str | None, age: float, alive: bool
) -> None:
    worker = inflight.WorkerInfo(
        pid=pid,
        host=socket.gethostname() if host == "here" else host,
        job_id=job_id,
        job_folder=None if job_id is None else "/nonexistent",
        claimed_at=time.time() - age,
    )
    assert worker.is_alive(no_job_timeout=600) is alive


def test_db_deletion_unblocks_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting inflight.db while a process is waiting should unblock it."""
    reg = inflight.InflightRegistry(tmp_path, inflight.WorkerInfo(pid=0, token="blocker"))
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
            worker = inflight.WorkerInfo(pid=0, token="rival")
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


def test_record_worker_info_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slurm = submitit.SlurmJob[None](folder=tmp_path, job_id="42")
    local = submitit.LocalJob[None](folder=tmp_path, job_id="1234")

    reg = inflight.InflightRegistry(tmp_path)
    with inflight.inflight_session(reg, ["s", "l"]) as claim:
        claim.record_worker_info(slurm, uids=["s"])
        claim.record_worker_info(local, uids=["l"])
        info = reg.get(["s", "l"])
        # claim handed to the Slurm job, whose local subprocess runs on its node
        monkeypatch.setattr(inflight.socket, "gethostname", lambda: "node-2")
        claim.record_worker_info(local, uids=["s"])
        s = reg.get(["s"])["s"]
    assert (info["s"].job_id, info["s"].job_folder) == ("42", str(slurm.paths.folder))
    assert (info["l"].pid, info["l"].job_id) == (1234, None)
    assert (s.host, s.pid, s.job_id) == ("node-2", 1234, "42"), "job must be kept"
