import os
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
            time.sleep(0.1)  # Simulate work
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


def test_incremental_processing(tmp_path: Path) -> None:
    """Test that workers can process non-overlapping items in parallel"""
    worker1 = Worker(infra={"folder": tmp_path})
    worker2 = Worker(infra={"folder": tmp_path})

    start_time = time.time()

    def run_worker1():
        # Process items [1, 2, 3, 4]
        list(worker1.process([1, 2, 3, 4]))

    def run_worker2():
        # Process items [3, 4, 5, 6] - overlaps with worker1 on [3, 4]
        time.sleep(0.05)  # Start slightly after worker1
        list(worker2.process([3, 4, 5, 6]))

    t1 = threading.Thread(target=run_worker1)
    t2 = threading.Thread(target=run_worker2)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    elapsed = time.time() - start_time

    # Worker1 processes [1, 2, 3, 4]
    # Worker2 should process [5, 6] while worker1 works on [1, 2]
    # Then worker2 gets [3, 4] from cache (worker1 already computed them)
    # So worker2 only computes [5, 6]

    # Check that worker2 only computed non-overlapping items
    assert 5 in worker2.computed
    assert 6 in worker2.computed
    # Worker2 should NOT have computed 3 or 4 (worker1 did)
    assert 3 not in worker2.computed
    assert 4 not in worker2.computed

    # Verify all items were processed by someone
    all_results = set(worker1.computed) | set(worker2.computed)
    assert all_results >= {1, 2, 3, 4, 5, 6}

    # Parallel execution should be faster than sequential
    # Each worker processes 0.1s per item
    # Worker1: 4 items = 0.4s
    # Worker2: 2 items (5, 6) = 0.2s, runs in parallel with worker1
    # Total should be ~0.5s if incremental, ~0.6s if sequential
    assert elapsed < 1.0, f"Took {elapsed}s, should be < 1.0s if processing incrementally"


def test_stale_lock_detection(tmp_path: Path) -> None:
    """Test that stale locks from crashed workers are detected and removed"""
    worker = Worker(
        infra={"folder": tmp_path, "lock_timeout": 2}
    )  # Short timeout for test

    cache_folder = worker.infra.cache_dict.folder
    assert cache_folder is not None
    lock_dir = cache_folder / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    # Create a "stale" lock by creating the lock file and making it old
    stale_lock_path = lock_dir / "5.lock"
    stale_lock_path.touch()

    # Make the lock file old (older than lock_timeout)
    old_time = time.time() - 5  # 5 seconds ago
    os.utime(stale_lock_path, (old_time, old_time))

    # Worker should detect stale lock, remove it, and process the item
    start = time.time()
    result = list(worker.process([5]))
    elapsed = time.time() - start

    # Should complete quickly (not wait for full timeout)
    assert elapsed < 1.0, f"Took {elapsed}s, should not wait for stale lock"

    # Item should have been processed
    assert 5 in worker.computed
    assert result == [10]  # 5 * 2
