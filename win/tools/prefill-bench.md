# prefill-bench, ttft-probe and rad-warm-bench

Three small measurement tools from alpha.4's phase 3 (`counters-and-prefill`). Each is a
measurement: run it alone on the machine (run class M), on a server or drive nothing else is using.
Python 3, standard library only; `-I` keeps a stray file in the working directory from being
imported.

## prefill_bench.py: prefill speed on fixed, fresh text

BetterBench's prefill at 32K moved 6-11% between two warm runs on one server
(`artifacts/pm/results/2026-11-20-0918-tp2-baseline-a2.md`). This bench sends the same text every
time, so two repeats differ only in the run:

- **Fixed:** one body of English-like prose made from a seed (`--seed`) by the script itself, with
  no corpus and no download. It is cut to each size at a token boundary through the server's
  `/tokenize`, to within 0.5%.
- **Fresh:** every request starts with a line naming the run, the size and the repeat, so the
  prefix cache never holds its first block. The bench warns if `timings.cache_n` is not 0.
  - The body's n-gram rows are the same every time.
  - After the warm-up pass they are in the page cache, as on a server started with `ngram-warm`.
- **Speed:** the server's `timings.prompt_n / prompt_ms`, from admission to first token, with
  `max_tokens: 1`. The client's wall time is kept beside it.
- **Order:** sizes interleave (8K, 32K, 64K, 128K, 8K, ...), one request at a time, after
  `--warmup` discarded passes. A drift then spreads over every size.

```
python -I win\tools\prefill_bench.py --port 8000 --sizes 32K,64K,128K --repeats 8 --stats --out run.json
```

- **Sizes** are in tokens, with K = 1024. 128K needs a `max-model-len` above 131,072.
- **Output:** a progress line per request. The summary per size gives the median tok/s, the spread
  ((max - min) / median), the CV, the B.4 floor (the larger of 3% and twice the spread) and the
  client's median.
- **`--stats`:** each request's `/stats` deltas: `link.ar_bytes`, `ar_calls`, `h2d_bytes` and the
  expert stager's `staged_bytes`, all rank 0. The summary adds the median `ar_bytes` per prompt
  token. On Flash-Next TP2 with `ar-bytes-count` that is about 501,760 B: 96 fused writes x 2,560 x
  2 bytes, plus the plain all-reduces' 10,240.
- **`--deadline UNIX_TIME`:** stop starting passes after that time, for a script with a window to
  keep.
- **JSON** (`--out`): `meta`, `summary` per size, and every row.

Exit status: 0; 1 if a request failed; 2 on usage.

## ttft_probe.py: the TTFT tail under steady short requests

The probe runs `--concurrency` workers for `--seconds`, after `--warmup` seconds it discards.
- **Requests:** short streamed chat requests, 48 fixed tasks with a unique first line. Temperature
  0.7, a fixed seed per request number, `ignore_eos`, `--max-tokens` 96 by default.
- **TTFT:** the client's time from send to the first chunk carrying text. The server's own
  `timings.ttft_ms` is kept beside it.
- **Summary:** p50, p90, p99 and max, with a bootstrap 95% interval on p50 and p99, so two arms'
  tails can be compared against noise.

```
python -I win\tools\ttft_probe.py --port 8000 --concurrency 8 --seconds 300 --out c8.json
```

Exit status: 0; 1 if more than 1% of requests failed.

## rad-warm-bench: the n-gram table's read into the file cache, alone

Built with the tools (`build-win\<flavor>\bin\rad-warm-bench.exe`). It needs no engine and no
card, and times `--ngram-warm`'s read of a container's n-gram table.

```
rad-warm-bench --evict D:\models\...\other.rad --evict-gib 120 --io alpha3 --verify --label A model.rad
```

- **`--io`:** the warm read's mode, `io=pread|handle|prefetch,readers=N,piece=SIZE`.
  - `alpha3` is the first form, 8 readers of 16 MiB `pread`s.
  - The default is `handle`, 16 readers of 64 MiB, each through its own handle.
  - The engine takes the same spec from `BRILLIANCE_NGRAM_WARM_IO`.
- **`--evict FILE`:** reads `--evict-gib` of another container through the cache first. The table's
  pages are then the oldest and go back first.
  - Without elevation there is no purge of the standby list.
  - Read more than the standby list holds (`\Memory\Standby Cache * Bytes`).
- **`--verify`:** reads the table a second time. A cached pass runs at memory-copy speed, about
  11 GB/s here, against the drive's 6.8.
  - The gap is narrow, so the solo-window script also samples `\PhysicalDisk(0 D:)\Disk Read
    Bytes/sec`, which reads 0 for a cached pass.

Output: one JSON line per read, `{label, what: evict|warm|verify, io, bytes, secs, gbps}`.

## Checking the scripts without a server

`python -I win\tools\bench_selftest.py` runs both clients against a fake server, with no model and
no card. It checks the sizes, that no request hits the cache, the spread, the per-token `ar_bytes`
and the probe's percentiles.
