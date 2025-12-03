import threading
import time
import typing as tp
from pathlib import Path

import filelock
import pydantic

from .map import MapInfra


class Worker(pydantic.BaseModel):
    infra: MapInfra = MapInfra()
    computed: list[int] = []

    @infra.apply(item_uid=str)
    def process(self, items: tp.Sequence[int]) -> tp.Iterator[int]:
        for item in items:
            self.computed.append(item)
            yield item * 2


def test_locking_wait(tmp_path: Path) -> None:
    """Test that a worker waits if item is locked"""
    worker = Worker(infra={"folder": tmp_path})

    # Get the actual cache folder (triggers creation)
    cache_folder = worker.infra.cache_dict.folder
    assert cache_folder is not None
    (cache_folder / "locks").mkdir(parents=True, exist_ok=True)

    # Lock item "2" manually
    lock_path = cache_folder / "locks" / "2.lock"
    lock = filelock.FileLock(lock_path)
    lock.acquire()

    # Function to run worker in a thread
    def run_worker():
        # This should block on item 2 until we release it
        list(worker.process([2]))

    t = threading.Thread(target=run_worker)
    t.start()

    # Give it a moment to start and block
    time.sleep(0.5)
    assert t.is_alive()  # Should be still waiting

    # Release lock
    lock.release()

    # Should finish now
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert 2 in worker.computed


def test_locking_completed(tmp_path: Path) -> None:
    """Test that if locked item is completed, second worker skips it"""
    worker1 = Worker(infra={"folder": tmp_path})
    # Use same config to get same UID folder
    worker2 = Worker(infra={"folder": tmp_path})

    cache_folder = worker1.infra.cache_dict.folder
    assert cache_folder is not None
    (cache_folder / "locks").mkdir(parents=True, exist_ok=True)

    # Lock item "2"
    lock_path = cache_folder / "locks" / "2.lock"
    lock = filelock.FileLock(lock_path)
    lock.acquire()

    def run_worker2():
        # Should wait then skip
        list(worker2.process([2]))

    t = threading.Thread(target=run_worker2)
    t.start()

    time.sleep(0.5)
    assert t.is_alive()

    # Pretend we computed it and wrote to cache
    with worker1.infra.cache_dict.writer() as w:
        w["2"] = 4

    # Release lock
    lock.release()

    t.join(timeout=2.0)
    assert not t.is_alive()
    # Worker 2 should NOT have computed it
    assert 2 not in worker2.computed
