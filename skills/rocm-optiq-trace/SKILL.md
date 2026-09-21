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

Profiling timings are not benchmark results.

## Non-negotiable rules

- Use `rocprofv3 --output-format rocpd` for GPU/API-only collection.
- Use `rocprof-sys-attach -F rocpd` when CPU call-stack samples are required.
- Run profiling containers with a writable cwd such as `-w /app`; never use
  container cwd `/`.
- Write profiler output to container-local `/tmp`, then copy completed files
  to a mounted host directory.
- Include `%pid%` in the output filename.
- Do not pass `--disable-cuda-graph`.
- Preserve `--disable-radix-cache` for this Qwen3.8 first-knife path.
- Validate databases before delivery.
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
capture, and one warmup. This example collects from second 300 through 390:

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
      --collection-period 300:90:1 -- \
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

The benchmark client sends `temperature: 0.0`. Leave several seconds of margin
at both ends of the collection window. If child-process timer alignment is
uncertain, issue a second identical request near the middle of the window.

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

If the installed `rocpd` has no `merge` subcommand, merge manually:

1. Copy rank0 as the output database.
2. Attach rank1 with SQLite.
3. Copy every UUID-suffixed concrete table from rank1.
4. Recreate each base `rocpd_*` view as `UNION ALL` over matching UUID tables.
5. Keep higher-level views from rank0; they join by both ID and GUID, so they
   work across both ranks.
6. Run `VACUUM` and `PRAGMA integrity_check`.

Do not concatenate tables while dropping the GUID. IDs overlap across ranks;
GUID is required to prevent invalid cross-rank joins.

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
than rich Python function names. For Python-level call stacks, use Torch
Profiler separately; those traces are not the same ROCpd artifact.

## Report back

Include:

- Model, TP/EP, GPU count, input/output lengths.
- Whether decode CUDA Graph capture succeeded.
- Collection window and trace type.
- Unified database path and size.
- Process/GPU/kernel/Graph-launch counts.
- Communication kernel names found.
- Explicit CPU call-stack status and any version blocker.
- Confirmation that the container stopped and GPUs were released.
