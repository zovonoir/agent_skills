---
name: rocm-optiq-trace
description: Collect ROCm Optiq-compatible ROCpd traces for multi-GPU SGLang or ATOM inference, preserve CUDA Graph execution, merge per-rank databases into one multi-GPU file, and validate GPU/API/communication events. Use when the user requests ROCm Optiq, rocprofv3, ROCpd, GPU timeline, communication-kernel, or optional CPU call-stack profiling.
---

# ROCm Optiq Trace Collection

Produce one ROCpd database that contains all GPU ranks and opens directly in
ROCm Optiq. The default workflow captures HIP APIs, CUDA/HIP Graph launches,
individual GPU kernels, memory operations, and communication kernels.

CPU call stacks are optional. They have been verified with Systems Profiler
1.7.0 plus ROCprofiler SDK 1.3.2, installed under an isolated prefix on top of
the ROCm 7.2.4 ATOM image.

## Default workload

Unless the user specifies otherwise:

- TP2 / EP2 on two GPUs.
- Exact input length: 2048 tokens.
- Exact output length: 5 tokens.
- One request, `temperature=0`, `random-range-ratio=1.0`.
- Warm the same request once before collection.
- Keep CUDA Graph enabled.
- Capture a short delayed window after startup and warmup.
- Merge per-rank ROCpd files into one database.
- Add `--enable-layerwise-nvtx-marker` when kernels need to be traced back to
  model source, then relabel the markers before viewing.

Profiling timings are not benchmark results.

## Non-negotiable rules

- Use `rocprofv3 --output-format rocpd` for GPU/API-only collection.
- Use `rocprof-sys-attach -F rocpd` when CPU call-stack samples are required.
- Run profiling containers with a writable cwd such as `-w /app`; never use
  container cwd `/`.
- Write profiler output to container-local `/tmp`, then copy completed files
  to a mounted host directory.
- Include `%pid%` in the output filename.
- Size the collection window generously and keep the server idle on both sides
  of the measured run. Never try to hit a narrow window precisely.
- Do not pass `--disable-cuda-graph`.
- Preserve `--disable-radix-cache` for this Qwen3.8 first-knife path.
- Validate databases before delivery.
- Anchor all ranks to one time origin when merging, and never "correct" rank
  timestamps by shifting them.
- Never claim CPU call stacks were collected unless the database contains
  nonempty samples/call-stack data.

## Known ROCprofiler Docker cwd bug

ROCprofiler SDK 1.1.0 can fail during ROCpd generation when container cwd is
`/`:

```text
ring_buffer: munmap failed: Invalid argument
mmap failed with errno 22
rocprofv3 caught signal 6
```

Always add:

```bash
docker run -w /app ...
```

A partial database produced after this error must not be delivered.

## 1. Preflight

Check tools inside the model runtime image:

```bash
command -v rocprofv3
rocprofv3 --version
cat /opt/rocm/.info/version
```

Verify ROCpd generation with a finite GPU program before profiling the model:

```bash
docker run --rm -w /app \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD:/trace" IMAGE bash -lc '
    rocprofv3 \
      --output-directory /tmp \
      --output-file rocpd-smoke \
      --output-format rocpd \
      --kernel-trace --hsa-trace -- \
      python -c '\''import torch
x=torch.randn(1024,1024,device="cuda")
y=x@x
torch.cuda.synchronize()
print(y.shape)'\''
    status=$?
    cp /tmp/rocpd-smoke_results.db /trace/
    exit $status
  '
```

Proceed only when rocprofv3 exits 0 and SQLite integrity is `ok`.

## 2. Start the Graph-enabled profiled server

Choose a delayed collection window long enough for model loading, decode Graph
capture, and one warmup. Read the window-timer section below before picking the
numbers; the window does not open when you think it does.

This example nominally collects from second 600 through 1200:

```bash
docker run --rm --name MODEL-graph-trace -w /app \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --network=host \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -v MODEL_PATH:/model:ro \
  -v OUTPUT_DIR:/trace \
  IMAGE bash -lc '
    export SGLANG_PLUGINS=atom_sglang
    export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

    exec rocprofv3 \
      --output-directory /tmp \
      --output-file model-graph-2k5-tp2-%pid% \
      --output-format rocpd \
      --runtime-trace \
      --collection-period 600:600:1 -- \
      python -m sglang.launch_server \
        --model-path /model \
        --host 0.0.0.0 --port 30080 \
        --tp-size 2 --ep-size 2 \
        --kv-cache-dtype bf16 \
        --page-size 64 \
        --context-length 4096 \
        --max-running-requests 4 \
        --mem-fraction-static 0.75 \
        --disable-radix-cache \
        --trust-remote-code
  '
```

Wait for server readiness. Confirm the log shows:

```text
Capture target decode CUDA graph begin
Capture target decode CUDA graph end
The server is fired up and ready to roll!
```

`disable_cuda_graph=False` must appear in resolved server arguments. Prefill
Graph may be automatically disabled by this model path; that does not mean
decode Graph is disabled.

## 2a. The collection-period timer is per process

`--collection-period` delay is counted from the moment rocprofiler initializes
inside each process, not from `docker run`. SGLang scheduler ranks are forked
children, so their timer starts late and the real window slides later by that
amount. Measured on this Qwen3.8 path:

```text
t=0s     docker run
t=44s    sglang::scheduler_TP0 / TP1 appear   <- delay is counted from here
t=664s   window actually opens    (44 + 600)
t=1264s  window actually closes   (44 + 600 + 600)
```

The offset is not a constant. It depends on weight load speed, JIT, and decode
Graph capture time. A run scheduled against container time can silently fall
outside the window and produce a database with none of the measured work.

This already caused one lost collection. With `--collection-period 420:120:1`
and two identical requests at container time 431 and 481, only the second one
landed inside the window. A 120-second window cannot absorb a 44-second drift.

### Use a wide window instead of precise timing

An open window over an idle server costs nothing. rocprofiler only records when
HIP calls and kernel dispatches actually happen, so an idle window produces no
events, no buffer growth, and no database growth. Scheduler RSS measured while
the window was open and the server idle:

```text
t=905s   10.63 GB
t=920s   10.63 GB   window open, no GPU activity
t=951s   10.63 GB   RSS completely flat
```

So prefer a window wide enough that drift cannot matter, and leave the server
idle on both ends. Database size is determined only by the work actually run
inside the window, never by how long the window stays open.

### Calibrating the offset when you need it

Poll for the scheduler process instead of guessing:

```bash
T0=$(date +%s)
while ! docker exec CONTAINER pgrep -f 'sglang::scheduler_TP0' >/dev/null 2>&1; do
  sleep 1
done
echo "timer origin t=$(( $(date +%s) - T0 ))s; window opens at origin + DELAY"
```

### No event-based trigger exists in 1.1.0

rocprofv3 1.1.0 gates collection by time only. `--kernel-iteration-range`
counts kernel dispatches, which is unusable when one pass dispatches tens of
thousands of kernels. `--marker-trace` records ROCTx ranges but does not gate
collection. There is no signal or API to start and stop a window on demand, so
the wide-window approach is the only reliable option on this image.

## 3. Warm up and capture

Run this once before the collection window, then once inside the window:

```bash
docker exec MODEL-graph-trace \
  python -m atom.benchmarks.benchmark_serving \
    --model /model --tokenizer /model \
    --backend sglang \
    --base-url http://127.0.0.1:30080 \
    --dataset-name random \
    --random-input-len 2048 \
    --random-output-len 5 \
    --random-range-ratio 1.0 \
    --num-prompts 1 \
    --max-concurrency 1 \
    --request-rate inf \
    --ignore-eos \
    --num-warmups 0 \
    --disable-tqdm
```

The benchmark client sends `temperature: 0.0`. Leave at least a hundred seconds
of idle margin at both ends of the collection window rather than a few seconds,
because the window boundary is not where the nominal delay puts it.

## 4. Finalize each scheduler rank

Find scheduler PIDs:

```bash
docker exec MODEL-graph-trace ps -eo pid,ppid,comm,args
```

After the window ends, finalize ranks one at a time:

```bash
docker exec MODEL-graph-trace kill -QUIT RANK0_PID
# Wait for "SQLite3 generation :: total" and "output generation".

docker exec MODEL-graph-trace kill -QUIT RANK1_PID
# Wait for the same completion markers.
```

ROCprofiler 1.1.0 can remain in its signal handler after writing the database.
Copy only after `tool finalization` appears:

```bash
docker exec MODEL-graph-trace cp \
  /tmp/model-graph-2k5-tp2-RANK0_PID_results.db \
  /trace/model-graph-2k5-tp2-rank0.db

docker exec MODEL-graph-trace cp \
  /tmp/model-graph-2k5-tp2-RANK1_PID_results.db \
  /trace/model-graph-2k5-tp2-rank1.db
```

## 5. Merge ranks into one ROCpd database

Newer `rocpd` versions provide:

```bash
rocpd merge \
  -i model-graph-2k5-tp2-rank0.db model-graph-2k5-tp2-rank1.db \
  -d . -o model-graph-2k5-tp2-unified
```

The ROCm 7.2.4 ATOM image ships rocprofv3 1.1.0, whose `rocpd` exposes only
`convert`, `query`, and `summary`. Expect to merge manually there.

If the installed `rocpd` has no `merge` subcommand, use
[scripts/merge_rocpd.py](scripts/merge_rocpd.py), which also applies the time
anchor described below:

```bash
python3 scripts/merge_rocpd.py \
  model-graph-2k5-tp2-rank0.db model-graph-2k5-tp2-rank1.db \
  model-graph-2k5-tp2-unified.db
```

What it does:

1. Copy rank0 as the output database.
2. Attach rank1 with SQLite.
3. Copy every UUID-suffixed concrete table from rank1.
4. Recreate each base `rocpd_*` view as `UNION ALL` over matching UUID tables.
5. Keep higher-level views from rank0; they join by both ID and GUID, so they
   work across both ranks.
6. Add a time-origin anchor to every late-starting rank.
7. Run `VACUUM` and `PRAGMA integrity_check`.

Do not concatenate tables while dropping the GUID. IDs overlap across ranks;
GUID is required to prevent invalid cross-rank joins.

## 5a. Per-rank time origins make collectives look staggered

Merged timestamps are already correct. Every rank sits on the same system clock
domain and absolute values are directly comparable. The trap is on the viewing
side: a viewer that zeroes each process against that process's own first event
injects the gap between process start times into every kernel on that rank.

The symptom is a synchronizing collective that appears to run at different
times on each GPU while showing nearly identical durations. Observed on this
Qwen3.8 path, where the two profiler sessions started 626.303 us apart:

```text
allgather_vec, per-rank origins
  GPU0 start 5566.945597 ms   dur 0.170600 ms
  GPU1 start 5566.315777 ms   dur 0.171121 ms   <- finishes before GPU0 starts

allgather_vec, one shared origin
  GPU0 start 5566.945597 ms
  GPU1 start 5566.942080 ms                     <- real gap is 3.5 us
```

That picture is physically impossible for a signal-based collective, which is
the tell that the origin is wrong rather than the data. Durations stay correct
because the normalization only translates the timeline.

Short kernels expose this and long ones hide it. At a 626 us origin gap, a
170 us `allgather_vec` shows zero interval overlap between ranks, while a
1.8 ms `ncclDevKernel` still overlaps 100% and looks fine.

### Shifting timestamps cannot fix it

Per-process normalization is shift-invariant. Adding an offset to one rank
moves that rank's own minimum by the same amount, so the subtraction cancels
and the display does not change:

```text
before  display = t - origin
after   display = (t + s) - (origin + s) = t - origin
```

The only way to move the display is to change which event is the minimum.

### Fix: anchor every rank to one origin

`merge_rocpd.py` inserts one zero-work region named `ROCPD_MERGE_TIME_ANCHOR`
into each late-starting rank, placed at the earliest event across all ranks.
Every process then reports the same first-event timestamp, so per-process
normalization becomes global normalization. Verified result:

| | merge without anchor | merge with anchor |
| --- | --- | --- |
| origin gap between ranks | 626.303 us | 0.000 us |
| `ar_ll128` median delta / overlap | 50.83 us / 6.7% | 2.56 us / 100% |
| `allgather_vec` median delta / overlap | 628.22 us / 0% | 2.91 us / 100% |

Nothing else changes. Kernel dispatch count, total GPU time, and Graph launch
count stay identical; only one region row and one string row are added.

For inspection outside a viewer, use
[scripts/export_global_timeline.py](scripts/export_global_timeline.py). It
writes a CSV of every kernel on one shared origin, keeping the per-rank value
in a second column so the two conventions can be compared directly, plus a
paired table of communication kernels with an interval-overlap flag.

### Do not confuse this with real rank skew

Genuine entry skew survives the fix and must not be anchored away. On this path
`cross_device_reduce` still showed a 836 us median entry gap after anchoring,
with rank0 arriving first in 79.6% of launches, while `ar_ll128` in the same
trace aligned to 2.56 us. A residual gap on one collective family only, with
others tightly aligned, is a load-imbalance signal and not a clock artifact.

## 6. Validate the unified trace

At minimum verify:

```python
import os
import sqlite3

p = "model-graph-2k5-tp2-unified.db"
con = sqlite3.connect(p)

assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
assert con.execute("SELECT COUNT(*) FROM rocpd_info_process").fetchone()[0] >= 2
assert con.execute("SELECT COUNT(*) FROM rocpd_kernel_dispatch").fetchone()[0] > 0
assert con.execute(
    "SELECT COUNT(*) FROM regions WHERE name LIKE 'hipGraphLaunch%'"
).fetchone()[0] > 0

# All ranks must report the same first-event timestamp, or collectives will
# look staggered in any viewer that zeroes each process separately.
origins = con.execute(
    """
    SELECT pid, MIN(start) FROM (
        SELECT pid, start FROM regions
        UNION ALL
        SELECT pid, start FROM kernels
    ) GROUP BY pid
    """
).fetchall()
assert len({value for _, value in origins}) == 1, origins

print("size", os.path.getsize(p))
print("processes", con.execute(
    "SELECT pid, command FROM processes ORDER BY pid"
).fetchall())
print("kernels by GPU", con.execute(
    """
    SELECT pid, agent_log_index, COUNT(*), SUM(duration)
    FROM kernels
    GROUP BY pid, agent_log_index
    ORDER BY pid, agent_log_index
    """
).fetchall())
con.close()
```

For communication analysis, search kernel names for:

- `cross_device_reduce`
- `allgather_vec`
- `ar_ll`
- `allreduce`, `all_reduce`, `rccl`, or `nccl`

Stop the container and verify GPU memory is released.

## 7. Attributing kernels to model source

A default `--runtime-trace` database answers "which HIP call launched this
kernel" but not "where in the model that call came from". Both halves need
care, and the second one needs an extra server flag.

### Kernel to HIP API: use the stack, not the correlation id

`rocpd_event.correlation_id` is not populated by rocprofv3 1.1.0. Every row
holds `0`, so any join on `corr_id` silently collapses the whole trace onto one
arbitrary region:

```text
rocpd_event.correlation_id: 227014 rows, 1 distinct value, all zero
```

Use `stack_id` and `parent_stack_id` instead. A kernel dispatch event's
`parent_stack_id` is the `stack_id` of the region that launched it, and the
chain walks upward from there. That resolved all 22602 kernels with no misses,
and it distinguishes Graph replay from eager launches:

```text
   960x  aiter::dynamic_per_token_scaled_quant   <- hipGraphLaunch
   785x  _gated_pointwise_kernel                 <- hipModuleLaunchKernel
   485x  Cijk_..._MT256x256x32                   <- hipExtModuleLaunchKernel
```

Also avoid joining through the `regions` view for this. It carries a correlated
subquery for `category` and no index on the join column; the same query that
took nine minutes there finished in 0.6 seconds against the concrete
`rocpd_*_<uuid>` tables.

Without markers the chain stops at the HIP API. `call_stack` and `line_info`
are `{}` throughout, and the only semantic name available is `ncclAllReduce`
from `RCCL_API`.

### HIP API to model source: enable layerwise markers

Add to the server launch:

```text
--enable-layerwise-nvtx-marker
```

SGLang registers forward hooks that push one NVTX range per module.
`torch.cuda.nvtx` maps to ROCTx on ROCm, and `--runtime-trace` already collects
the Marker API, so nothing changes on the rocprofv3 side. Module ranges nest,
so the marker chain above a kernel reconstructs the module path:

```text
model.model
  model.model.layers.3
    model.model.layers.3.self_attn
      model.model.layers.3.self_attn.qkv_proj
        hipLaunchKernel
          ck::kernel_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle_2lds<...>
```

Measured coverage on 2048/5 at concurrency 32:

| Phase | Kernels | Attributed to a module | GPU time |
| --- | ---: | ---: | ---: |
| eager / prefill | 12812 | 11560 (90.2%) | 5428 ms |
| Graph replay / decode | 9790 | 0 (0.0%) | 110 ms |

Decode gets nothing, because PyTorch forward hooks do not run during Graph
replay. Accept this rather than disabling Graph: prefill dominates this
workload, and a graph-off run would no longer be the configuration under study.

The flag is effectively free here. A measured pass took 5.66 s with markers
against 5.62 s without, inside run-to-run noise, because the hooks only run on
the Python side during prefill.

### Relabel markers before viewing

rocprofv3 names every ROCTx region after the API that produced it, so a viewer
shows a stack of identical `roctxThreadRangeA` rows. The pushed text lives in
`rocpd_event.extdata` as a JSON `message` field and never reaches
`region.name_id`, which points at the single string `roctxThreadRangeA` for all
marker rows.

The nesting is already correct at that point; only the labels are missing. Fix
it with [scripts/relabel_roctx_markers.py](scripts/relabel_roctx_markers.py):

```bash
python3 scripts/relabel_roctx_markers.py unified.db unified-labeled.db
```

Two details matter when parsing. The `message` is SQL-escaped, so single quotes
in the Python dict repr arrive doubled, and `rocpd_string.string` is UNIQUE, so
new labels need `INSERT OR IGNORE` followed by a lookup.

Verified end state in ROCm Optiq: readable module call stacks, kernels visible
inside Graph regions, and ranks aligned on one time origin.

## Profiling overhead

Measured on Qwen3.8-Flash-Next PTPC-FP8, TP2/EP1, decode Graph enabled, 32
concurrent requests, `--runtime-trace`, on two MI308X. Four conditions, where B
and C share one server instance so model load, JIT, and Graph capture are
identical:

| Condition | 2048/5 duration | 2048/1024 duration |
| --- | --- | --- |
| A no profiler | 5.51 s | 27.36 s |
| B attached, window closed | 5.54 s (+0.4%) | 27.57 s (+0.8%) |
| C attached, window collecting | 5.62 s (+1.9%) | 30.16 s (+10.2%) |
| D attached, window reopened closed | 5.52 s (+0.0%) | 27.36 s (-0.0%) |

Read this as two separate costs:

- Merely having rocprofv3 attached is free. Both B deltas sit inside run-to-run
  noise, which is about 1% on this path. A long-lived service can stay wrapped
  and open a window only when needed.
- Active collection is cheap for short decode and expensive for long decode.
  The 2048/5 case moves 1.9%, so its trace is representative. The 2048/1024
  case moves 10.2%, so absolute times from such a trace must not be quoted as
  serving performance.

Overhead also grows within a window as the event buffer fills. The two
2048/1024 passes inside one window measured 29.26 s then 31.07 s, tracking
scheduler RSS growth from 10.22 GB to 10.63 GB. Prefer short windows containing
few passes over long windows accumulating many.

### Measuring overhead on a new path

Condition D is what makes the result trustworthy. Without it, a slowdown during
collection cannot be separated from drift, thermal effects, or cache state.
Re-run the same case after the window closes and confirm both throughput and
RSS return to baseline.

Use RSS as the independent check that the window is really open. Run load
during the expected closed phase and confirm RSS stays flat, then run the same
load inside the window and confirm RSS climbs. This is more reliable than
trusting the nominal window arithmetic.

Collect throughput numbers only. Kill the container without finalizing so no
time is spent generating a database that will be discarded.

## CPU call-stack collection

### Verified compatibility

The following combination was built and tested successfully in a temporary
copy of `rocm/atom-dev:sglang-latest`:

- Base OS: Ubuntu 24.04.
- Base ROCm: 7.2.4.
- Backported ROCprofiler SDK: 1.3.2.
- Backported ROCm Systems Profiler: 1.7.0.
- Install prefix: `/opt/rocm-tools-7.14.1`.
- Existing `/opt/rocm` was not replaced.

Both TP2 scheduler ranks could be attached and detached without interrupting
the SGLang service. A warmed 2K/5-token request completed while attached.

Observed rank0 output:

```text
integrity=ok
rocpd_sample=69790
rocpd_kernel_dispatch=12460
rocpd_region=47642
hipGraphLaunch=5
```

Observed rank1 output:

```text
integrity=ok
rocpd_sample=72796
rocpd_kernel_dispatch=12460
rocpd_region=48168
hipGraphLaunch=5
```

### Build requirements

Systems 1.7.0 must be paired with ROCprofiler SDK 1.3.2 from the same
`therock-7.14.1` release. Building Systems alone against the image's SDK 1.1.0
does not provide a usable attach tool.

Build SDK first:

```bash
cmake -S rocprofiler-sdk -B sdk-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm-tools-7.14.1 \
  -DCMAKE_PREFIX_PATH="/opt/rocm-tools-7.14.1;/opt/rocm" \
  -DROCPROFILER_BUILD_TESTS=OFF \
  -DROCPROFILER_BUILD_INTEGRATION_TESTS=OFF \
  -DROCPROFILER_BUILD_SAMPLES=OFF \
  -DROCPROFILER_BUILD_BENCHMARK=OFF \
  -DROCPROFILER_BUILD_DOCS=OFF

cmake --build sdk-build --target install --parallel 16
```

Confirm this library exists before configuring Systems:

```text
/opt/rocm-tools-7.14.1/lib/librocprofiler-sdk-rocattach.so
```

Build Systems:

```bash
cmake -S rocprofiler-systems -B systems-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm-tools-7.14.1 \
  -DCMAKE_PREFIX_PATH="/opt/rocm-tools-7.14.1;/opt/rocm" \
  -DROCPROFSYS_BUILD_TESTING=OFF \
  -DROCPROFSYS_INSTALL_TESTING=OFF \
  -DROCPROFSYS_BUILD_EXAMPLES=OFF \
  -DROCPROFSYS_INSTALL_EXAMPLES=OFF \
  -DROCPROFSYS_USE_PYTHON=OFF \
  -DROCPROFSYS_USE_PAPI=OFF \
  -DROCPROFSYS_BUILD_DYNINST=ON \
  -DROCPROFSYS_BUILD_ELFUTILS=OFF \
  -DROCPROFSYS_BUILD_LIBIBERTY=ON \
  -DROCPROFSYS_BUILD_TBB=ON \
  -DROCPROFSYS_BUILD_LIBUNWIND=OFF \
  -DROCPROFSYS_BUILD_SQLITE3=OFF

cmake --build systems-build --target install --parallel 16
```

The release component tarballs omit some git submodule contents. Fetch
dependencies at the exact `therock-7.14.1` gitlink revisions when possible.
Do not silently substitute arbitrary dependency revisions in a reusable image.

### Start an attachable SGLang server

Set attach and sampling variables before Python starts so spawned scheduler
processes inherit them:

```bash
export ROCP_TOOL_ATTACH=1
export ROCPROFSYS_USE_SAMPLING=ON
export ROCPROFSYS_SAMPLING_REALTIME=ON
export ROCPROFSYS_SAMPLING_FREQ=100
export ROCPROFSYS_USE_ROCPD=ON
export ROCPROFSYS_TRACE=OFF

export PATH=/opt/rocm-tools-7.14.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-tools-7.14.1/lib:\
/opt/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
```

Then launch SGLang normally with CUDA Graph enabled. Do not wrap it with
`rocprof-sys-run`.

Warm the request and identify scheduler PIDs:

```bash
ps -eo pid,ppid,comm,args
```

### Attach and detach a scheduler

Start one attach session per `sglang::scheduler_TP*_EP*` process:

```bash
rocprof-sys-attach \
  -p SCHEDULER_PID \
  -o /tmp/sglang-attach-rank0 \
  -F rocpd
```

Wait for:

```text
Attached to process SCHEDULER_PID. Press ENTER to detach.
```

Run the measured 2K/5-token request, then press Enter. The output is under a
timestamped directory:

```text
/tmp/sglang-attach-rank0/<timestamp>/rocpd-<pid>-0.db
```

For noninteractive automation, use a FIFO:

```bash
rm -f /tmp/attach-control
mkfifo /tmp/attach-control
exec 3<>/tmp/attach-control

rocprof-sys-attach \
  -p SCHEDULER_PID \
  -o /tmp/sglang-attach-rank0 \
  -F rocpd <&3
```

After the measured request, detach from another shell:

```bash
printf '\n' > /tmp/attach-control
```

Repeat for every scheduler rank, validate each database, then merge them using
the same GUID-preserving process described earlier.

### Important 1.7.0 ROCpd SQL bug

The attached target's command line is inserted into ROCpd metadata without
properly escaping embedded double quotes. A target launched with:

```bash
python -c 'x=torch.randn(..., device="cuda")'
```

can abort during database generation with:

```text
Database Error
near "cuda": syntax error
```

Use a script or module path instead:

```bash
python /path/to/target.py
python -m sglang.launch_server ...
```

Keep complex quoted code out of the target command line.

### Old Systems Profiler limitation

Systems Profiler 1.3.0 bundled in the ROCm 7.2.4 image has no
`rocprof-sys-attach`. Wrapping multi-process SGLang with:

```bash
rocprof-sys-run --sample realtime --use-rocm true ...
```

causes spawned scheduler processes to fail with:

```text
std::future_error: No associated state
```

Do not use the wrapper workflow for multi-process SGLang and do not deliver its
failed output.

Also note that native sampling commonly shows CPython/native C++ frames rather
than rich Python function names, which makes it a poor way to locate model
code. For source attribution inside a ROCpd trace, prefer the ROCTx marker
route in section 7; it keeps CUDA Graph enabled and needs no rebuilt profiler.
Use Torch Profiler when full Python call stacks are required; those traces are
not the same ROCpd artifact.

## Report back

Include:

- Model, TP/EP, GPU count, input/output lengths.
- Whether decode CUDA Graph capture succeeded.
- Collection window and trace type, plus how the measured run was confirmed to
  fall inside the real window rather than the nominal one.
- Which measured passes landed inside the window, and which were dropped.
- Unified database path and size.
- Process/GPU/kernel/Graph-launch counts.
- Confirmation that all ranks share one time origin after merging.
- Communication kernel names found, and whether any residual entry skew looks
  like real load imbalance rather than a normalization artifact.
- Whether layerwise markers were enabled, and the share of kernels attributed
  to a module split by prefill and Graph-replay decode.
- Explicit CPU call-stack status and any version blocker.
- Confirmation that the container stopped and GPUs were released.
