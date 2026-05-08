---
name: sglang-torch-profiler
description: Collect and analyze SGLang Torch Profiler traces for GPU kernel, Python call stack, and Perfetto investigation. Use when profiling SGLang inference, running bench_offline_throughput with --profile, controlling trace size, interpreting .trace.json.gz files, or comparing kernel-level behavior on AMD/ROCm systems.
---

# SGLang Torch Profiler

## Use This Skill When

Use this skill to help the user collect SGLang Torch Profiler traces, keep trace files small enough to inspect, and analyze kernel-level execution with Perfetto or scripts.

Default goals:

- Capture eager-mode traces with Python call stacks and GPU kernels.
- Keep traces focused on prefill plus a few decode steps.
- Identify major kernel categories and compare versions.
- Verify whether an ATOM plugin path is visible in the trace.

## Collection Rules

Always prefer eager-mode profiling for kernel attribution:

- Add `--profile`.
- Set `SGLANG_TORCH_PROFILER_DIR` to the output directory.
- Add `--disable-cuda-graph` unless the user explicitly wants graph-level profiling.
- Keep `--random-output-len` small, usually `5`.
- Keep `--num-prompts` small, usually `2`.

Why `--disable-cuda-graph` matters:

- With CUDA Graph enabled, the trace mostly shows `hipGraphLaunch` or graph runner events.
- With CUDA Graph disabled, the trace includes Python function events and fine-grained kernel launches.

Do not treat profile throughput as benchmark throughput. Profiling changes runtime behavior and adds overhead.

## Basic Collection Command

Use this template for an offline throughput profile:

```bash
SGLANG_TORCH_PROFILER_DIR=/output/dir \
python3 -m sglang.bench_offline_throughput \
  --model-path /path/to/model \
  --dataset-name random \
  --random-input-len 60381 \
  --random-output-len 5 \
  --num-prompts 2 \
  --tp-size 2 \
  --ep-size 2 \
  --profile \
  --attention-backend aiter \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --disable-cuda-graph
```

Adjust `--random-input-len`, `--tp-size`, `--ep-size`, model path, and GPU visibility to match the workload. Avoid increasing output length unless the user needs long-decode behavior.

## InferenceX Profile Config

When the user uses an InferenceX-style `run_profile.sh`, use this config shape:

```bash
#!/bin/bash
IMAGE="atom-sglang:latest"
MODEL_PATH="/.cache/huggingface/Qwen3.5-27B"
MODEL_PREFIX="qwen3.5"
HOST_MODEL_MOUNT_PATH="/raid/models"

CONTAINER_ENV_OVERRIDES=(
    "SGLANG_DISABLE_CUDNN_CHECK=1"
    "SGLANG_USE_AITER=1"
    "SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1"
    "SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1"
    "SGLANG_USE_AITER_NEW_CA=false"
    "SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models"
)
EXTRA_CONTAINER_MOUNTS=()
EXTRA_DOCKER_ARGS=(
    -e "HIP_VISIBLE_DEVICES=4,5"
    -e "CUDA_VISIBLE_DEVICES=4,5"
)
SERVER_ARGS=(
    --attention-backend aiter
    --trust-remote-code
    --mem-fraction-static 0.9
)
POST_START_COMMANDS=()

# Format: "ISL OSL NUM_PROMPTS TP EP [extra args]"
PROFILE_CONFIGS=(
    "60381 5 2 2 2 --disable-cuda-graph"
)
```

Run it with:

```bash
CONFIG_FILE=./config_profile.sh bash run_profile.sh
```

## Expected Outputs

After collection, inspect `SGLANG_TORCH_PROFILER_DIR`. Expected files:

```text
<timestamp>-host.trace.json.gz
<timestamp>-TP-0-EP-0.trace.json.gz
<timestamp>-TP-1-EP-1.trace.json.gz
```

Usually start with `TP-0-EP-0`. Compare multiple GPU traces only when investigating imbalance or per-rank differences.

If the GPU trace is too large for Perfetto, reduce `--random-output-len` and `--num-prompts`.

## Perfetto Analysis

Open <https://ui.perfetto.dev/> and drag in the `.trace.json.gz` file.

For call-stack attribution:

- Confirm `python_function` events exist.
- Search for model or plugin paths, such as `atom/` or `ATOM/`.
- Correlate expensive kernels with the nearest Python function events.

If only graph launch events are visible, the workload likely ran with CUDA Graph enabled. Re-profile with `--disable-cuda-graph`.

## Script Analysis

Use this script to summarize GPU kernels by total duration:

```python
import gzip
import json
from collections import defaultdict

with gzip.open("trace.json.gz", "rt") as f:
    data = json.load(f)

events = data.get("traceEvents", [])
kernel_totals = defaultdict(lambda: {"count": 0, "total_us": 0})

for event in events:
    if event.get("cat") == "kernel" and event.get("ph") == "X":
        name = event.get("name", "")
        duration = event.get("dur", 0)
        kernel_totals[name]["count"] += 1
        kernel_totals[name]["total_us"] += duration

for name, stats in sorted(kernel_totals.items(), key=lambda item: -item[1]["total_us"])[:20]:
    print(f"{stats['count']:>5}x  {stats['total_us'] / 1000:>8.1f}ms  {name[:80]}")
```

Use this snippet to verify whether ATOM paths appear in Python events:

```python
for event in events:
    if event.get("cat") == "python_function":
        name = str(event.get("name", ""))
        if "atom/" in name or "ATOM/" in name:
            print(name)
```

## Kernel Categories

Use these labels when grouping SGLang traces:

| Keyword | Category |
| --- | --- |
| `Cijk_` | GEMM / hipBLASLt |
| `mha_batch`, `FmhaBatch` | Attention |
| `paged_attention` | Paged attention decode |
| `chunk_gated_delta`, `chunk_fwd_kernel` | GDN chunk |
| `fused_sigmoid_gating` | GDN fused recurrent |
| `conv1d` | GDN conv1d |
| `rmsnorm`, `layer_norm` | Normalization |
| `CatArray` | Tensor concat |
| `elementwise` | Element-wise ops |
| `cross_device_reduce` | AllReduce |

For comparisons, define one categorizer and reuse it across baseline and candidate traces:

```python
def categorize(name):
    if "Cijk_" in name:
        return "GEMM"
    if "mha_batch" in name or "FmhaBatch" in name:
        return "Attention"
    if "paged_attention" in name:
        return "PagedAttention"
    if "chunk_gated_delta" in name or "chunk_fwd_kernel" in name:
        return "GDN-Chunk"
    if "rmsnorm" in name or "layer_norm" in name:
        return "Norm"
    if "cross_device_reduce" in name:
        return "AllReduce"
    if "elementwise" in name:
        return "Elementwise"
    return "Other"
```

## Report Back

When summarizing a profiling run, include:

- The exact profiling command or config used.
- Whether CUDA Graph was disabled.
- Trace file paths and sizes.
- Top kernel categories by total time.
- Any missing expected Python call stacks or ATOM paths.
- Caveats about profiling overhead and multi-stream overlap.
