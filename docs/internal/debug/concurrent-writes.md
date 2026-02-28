# Concurrent-write duplication in shared CacheDict folders

Tracking doc for investigating and fixing data duplication when multiple
processes write to the same CacheDict cache folder.

## Observed problem

An EMG extraction cache (Sivakumar2024 study, 1109 unique recordings) was
expected to be ~281 GB but occupied **4.93 TB** on disk — a **17.5× inflation**.

### Root cause

16 worker processes (8 threads on `learnfair6033` + 8 on `learnfair6034`)
all computed the same 1097 recordings and wrote to the same cache folder.
Each worker creates its own `{hostname}-{threadid}.data` shared file via
`DumpContext.shared_file()`, so 16 separate `.data` files were produced,
each containing a full copy of the binary data (~308 GB each).

An additional `devfair0679` worker wrote 12 non-overlapping recordings.

```
17 info files, 17564 total JSONL entries
1109 unique keys
1097 keys × 16 copies = 17552 duplicate entries
  12 keys × 1 copy  = 12 unique entries
```

On read, `CacheDict._read_info_files()` loads all info files and
`_key_info.update()` keeps whichever reader returns last (non-deterministic).
The other 15 copies of each key's binary data become unreferenced but remain
on disk — `_cleanup_orphaned_jsonl_files` only removes files whose **first
line** is blanked (i.e., explicitly deleted), not files that merely lost the
"last-writer-wins" race.

## Manual dedup command

All 16 learnfair workers have identical key sets. Keep one learnfair
worker + devfair, delete the other 15 data file pairs:

```bash
# In the cache folder on the remote machine:
# Dry run — shows what would be deleted:
for f in *-info.jsonl; do
  case "$f" in
    devfair0679-2912891-*|learnfair6033-1713079-*) ;;
    *) prefix="${f%-info.jsonl}"
       echo "DELETE: $f data/${prefix}.data data/${prefix}-data.jsonl"
       ;;
  esac
done

# Actual deletion (removes ~4.6 TB):
for f in *-info.jsonl; do
  case "$f" in
    devfair0679-2912891-*|learnfair6033-1713079-*) ;;
    *) prefix="${f%-info.jsonl}"
       echo "DELETE: $f data/${prefix}.data data/${prefix}-data.jsonl"
       ;;
  esac
done
```

After cleanup: 281 GB (1109 unique recordings across 2 info files).

TODO in
/checkpoint/jarod/brainai/cache/emg2qwerty-new-mne-v2/neuralset.extractors.neuro.EmgExtractor._get_data,1/frequency=2000,name=EmgExtractor-5ffb61ff/data

---

## TODO: investigate why dedup didn't fire

### Question 1: Why didn't MapInfra prevent duplicate submissions?

`MapInfra._find_missing()` filters out already-cached items before
submitting jobs (map.py:309-358). It also calls `JobChecker.wait()` to
wait for pending jobs. `JobChecker` uses
`submitit.core.utils.JobPaths.get_first_id_independent_folder` to strip
slurm placeholders (`%j`, etc.) from the executor path, so all
experiments sharing the same cache uid land in the same
`running-jobs/` folder. Cross-experiment visibility is not the problem.

The real issue is a **TOCTOU race** between checking and registering:

```
# _find_missing (map.py:344-350):
jcheck = JobChecker(folder=executor.folder)
jcheck.wait()          # ← CHECK: reads running-jobs/, finds nothing
keys = set(self.cache_dict)
missing = ...          # returns 1097 missing

# ... later in _method_override (map.py:388-410):
executor.submit(...)   # submits slurm jobs
jcheck.add(jobs)       # ← REGISTER: writes .pkl files (too late)
```

If N experiments call `_find_missing` before any reaches `jcheck.add`,
they all see an empty `running-jobs/` folder and all submit jobs.

Additionally, `_call_and_store` (map.py:506-508) rechecks the cache at
job start, but if all jobs start before any has finished writing, every
job sees an empty cache and proceeds.

- [ ] **Fix**: move `jcheck.add()` before `executor.submit()`, or use a
  file lock around the check-submit-register sequence.
- [ ] **Verify**: confirm this is the actual code path used for the EMG
  data (MapInfra vs Step infrastructure — the `#key` format
  `{"cls":"Sivakumar2024","method":"_load_raw",...}` could be either).

### Question 2: Can we deduplicate on the fly and use orphan cleanup?

Current orphan cleanup (`_cleanup_orphaned_jsonl_files`) only deletes
files whose first line is blanked (confirming explicit deletion via
`__delitem__`). It deliberately skips files with valid first lines to
avoid deleting concurrent writes in progress.

Proposed approach:

- [ ] **Detect duplicate keys at read time** — after `_read_info_files`
  loads all info files, identify keys that appear in multiple JSONL
  files.
- [ ] **Blank duplicate entries** — for each duplicate, blank the entry
  in the non-winning JSONL file (same as `__delitem__` does). This
  makes them eligible for orphan cleanup.
- [ ] **Let existing orphan cleanup handle the rest** — once all entries
  in a JSONL file are blanked, `_cleanup_orphaned_jsonl_files` will
  delete the JSONL and its associated `data/` files.
- [ ] **Safety**: only do this when not inside a `write()` context, to
  avoid interfering with concurrent writers.

### Question 3: Can we prevent this at submission time?

- [ ] **Cross-experiment lock file** — before submitting jobs, acquire a
  file lock in the cache folder. Other experiments would wait or skip
  items already being computed.
- [ ] **Cache-folder-level JobChecker** — store pending job IDs in the
  cache folder (not the executor folder) so different experiments can
  see each other's in-flight work.
