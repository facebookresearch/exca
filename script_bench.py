import cProfile
import logging
import time
import typing as tp

import numpy as np
import pydantic

import exca as xk

logging.getLogger("exca").setLevel(logging.DEBUG)


class DumpMap(pydantic.BaseModel):
    param: int = 12
    infra: xk.MapInfra = xk.MapInfra(version="1")

    @infra.apply(
        item_uid=str,
        cache_type="MemmapArrayFile",
    )
    def process(self, items: tp.Iterable[str]) -> tp.Iterator[np.ndarray]:
        for item in items:
            yield np.random.rand(200, self.param)


profiler = cProfile.Profile()
profiler.enable()
cfg = DumpMap(infra={"folder": "./cache-test-memfile-context", "keep_in_ram": False})
t0 = time.time()
cfg.process([str(k) for k in range(4000)])
t1 = time.time()
print(t1 - t0)  # 0.3s creation, 0.15ss read
profiler.disable()
profiler.dump_stats("dump-file.prof")

profiler = cProfile.Profile()
profiler.enable()
cfg = DumpMap(infra={"folder": "./cache-test-memfile", "keep_in_ram": False})
t0 = time.time()
cfg.process([str(k) for k in range(4000)])
t1 = time.time()
print(t1 - t0)  # 23s creation, 0.2s read
profiler.disable()
profiler.dump_stats("dump-file.prof")

profiler = cProfile.Profile()
profiler.enable()
cfg = DumpMap(infra={"folder": "./cache-test-jsonl", "keep_in_ram": False})
t0 = time.time()
cfg.process([str(k) for k in range(4000)])
t1 = time.time()
print(t1 - t0)  # 18s creation, 0.2s read
profiler.disable()
profiler.dump_stats("dump-file.prof")


cfg = DumpMap(infra={"folder": "./cache-test-main", "keep_in_ram": False})
t0 = time.time()
cfg.process([str(k) for k in range(4000)])
t1 = time.time()
print(t1 - t0)  # 5.6s creation, 0.03s check


t0 = time.time()
for x, y in cfg.infra.cache_dict.items():
    print(x)
t1 = time.time()
print(t1 - t0)  # 0.5s (memfile)  11s

keys = [str(k) for k in range(4000)]
t0 = time.time()
for x in keys:
    y = cfg.infra.cache_dict[x]
    _ = np.array(y)
    # print(x)
t1 = time.time()  # 0.5 (memfile)  13s (standard)
print(t1 - t0)  # 0.5s (memfile)  11s
