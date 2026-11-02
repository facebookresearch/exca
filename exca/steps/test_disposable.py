import typing as tp
from pathlib import Path

import pytest

import exca

from . import backends, base, conftest


class _StoredAdd(conftest.Add):
    CACHE_TYPE: tp.ClassVar[str | None] = "Pickle"


@pytest.mark.parametrize("keep_in_ram", [False, True])
def test_disposable_retry_recomputes_deleted_staging(
    tmp_path: Path, keep_in_ram: bool
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    staged_infra: tp.Any = {
        **infra,
        "lifecycle": "disposable",
        "keep_in_ram": keep_in_ram,
    }
    producer = _StoredAdd(value=1, infra=staged_infra)
    consumer = conftest.Add(value=10, fail_on="all", infra=infra)
    chain = base.Chain(steps=[producer, consumer])
    with pytest.raises(ValueError, match="Triggered"):
        chain.run(1.0)
    owner = chain.lookup(1.0)._cache
    assert owner is not None
    assert owner.cache_dict is owner.cache_dict
    staged = next(iter(owner.staged.values()))
    uid = producer.lookup(1.0).uid
    peer = exca.cachedict.CacheDict[tp.Any](staged.paths.cache_folder)
    with peer.write():
        del peer[uid]
    chain.lookup(1.0).clear_cache()
    with pytest.raises(ValueError, match="Triggered"):
        chain.run(1.0)
    assert producer.calls == [1.0, 1.0]


@pytest.mark.parametrize("fail", [False, True])
def test_disposable_cleanup_preserves_result_and_releases_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    infra: tp.Any = {"backend": "Cached", "folder": tmp_path}
    staged_infra: tp.Any = {**infra, "lifecycle": "disposable"}
    producer = conftest.Add(value=1, infra=staged_infra)
    consumer = conftest.Add(value=10, fail_on={3.0} if fail else None, infra=infra)
    chain = base.Chain(steps=[producer, consumer])
    clear = backends._StepCache.clear

    def fail_clear(cache: backends._StepCache, uids: tp.Iterable[str]) -> None:
        if "staging" in cache.paths.step_folder.parts and list(uids):
            raise OSError("reclaim failed")
        clear(cache, uids)

    monkeypatch.setattr(backends._StepCache, "clear", fail_clear)
    if fail:
        with pytest.raises(ValueError, match="Triggered"):
            list(chain.run_many([1.0, 2.0]))
    else:
        assert list(chain.run_many([1.0, 2.0])) == [12.0, 13.0]
    handle = chain.lookup(1.0)
    assert handle.result() == 12.0
    with backends.inflight.InflightRegistry(handle.paths.step_folder) as registry:
        assert not registry.get()
