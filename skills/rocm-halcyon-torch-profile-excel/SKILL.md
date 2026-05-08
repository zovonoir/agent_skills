---
name: rocm-halcyon-torch-profile-excel
description: Set up rocm-halcyon, parse PyTorch Torch Profiler Chrome trace JSON files, export GPU kernel data to Excel, and explain the resulting fields. Use when converting .trace.json or .trace.json.gz profiler files to .xlsx, running trace_to_xlsx.py, or analyzing rocm-halcyon Excel exports for AMD GPU traces.
---

# rocm-halcyon Torch Profile Excel

## Use This Skill When

Use this skill when the user wants an agent to turn PyTorch Torch Profiler Chrome trace files into Excel using `rocm-halcyon`, then explain the exported GPU kernel rows.

Default behavior:

- Work in a temporary clone or container rather than modifying the user's source tree.
- Parse plain `.json` trace files with `rocm-halcyon`.
- Decompress `.trace.json.gz` to `.json` before parsing.
- Export one row per GPU kernel to `.xlsx`.
- Explain duration, stream, call stack, CPU op, and input shape fields.

Do not modify `rocm-halcyon` parser code just to support gzipped input. Do not edit `setup.py` to add dependencies unless the user explicitly wants package changes.

## Clean Ubuntu Setup

Use Docker when the user wants a reproducible clean environment:

```bash
docker run -d --name rocm_halcyon_ubuntu22_test \
  -v "/raid/users/jialzhu/code_backup:/workspace" \
  -w /tmp \
  ubuntu:22.04 sleep infinity
```

Install system and Python dependencies:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates git gzip build-essential cmake ninja-build \
  python3 python3-dev python3-pip
python3 -m pip install --no-cache-dir wheel setuptools
python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-cache-dir tqdm openpyxl pandas
'
```

Dependency notes:

- `build-essential`, `cmake`, `ninja-build`, and `python3-dev` build `halcyon_core`.
- `torch` is needed because `rocm_halcyon.__init__` imports `profiler_bridge.py`.
- `tqdm` is used by the parser.
- `openpyxl` is needed for `pandas.to_excel()`.

## Clone And Install rocm-halcyon

Clone with submodules:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
rm -rf /tmp/rocm-halcyon
git clone --recursive https://github.com/zovonoir/rocm-halcyon /tmp/rocm-halcyon
test -f /tmp/rocm-halcyon/external/pybind11/CMakeLists.txt
'
```

If the repository already exists but submodules are missing:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
cd /tmp/rocm-halcyon
git submodule update --init --recursive
test -f external/pybind11/CMakeLists.txt
'
```

Build and install:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
cd /tmp/rocm-halcyon
rm -rf build dist rocm_halcyon.egg-info
rm -f rocm_halcyon/halcyon_core*.so
python3 setup.py bdist_wheel
python3 -m pip install --no-cache-dir dist/*.whl --force-reinstall
python3 - <<'"'"'PY'"'"'
import rocm_halcyon as rh
print("rocm_halcyon import ok")
print(rh.__file__)
PY
'
```

If CMake cache points at stale paths, clean `build`, `dist`, `rocm_halcyon.egg-info`, and any existing `halcyon_core*.so` before rebuilding.

## Prepare Trace Input

`rocm-halcyon` expects plain JSON. For compressed profiler output, decompress first:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
gzip -dc /workspace/path/to/profile.trace.json.gz > /tmp/trace.json
test -s /tmp/trace.json
'
```

For an existing plain trace:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
cp /workspace/path/to/profile.trace.json /tmp/trace.json
test -s /tmp/trace.json
'
```

## Export Script

Use this minimal script as `/tmp/trace_to_xlsx.py`:

```python
import rocm_halcyon as rh

ops = rh.parse_torch_profiler("./trace.json", device_type="amd")

rh.export_to_excel(
    ops,
    {
        "gpu_kernel_name": rh.GPUKernelNameVisitor(),
        "cpu_kernel_name": rh.CPUKernelNameVisitor(),
        "call_stack": rh.CallStackVisitor(),
        "kernel_duration": rh.KernelDurationVisitor(),
        "start_timestamp": rh.KernelStartTimestampVisitor(),
        "end_timestamp": rh.KernelEndTimestampVisitor(),
        "input_shape": rh.KernelInputShapeVisitor(),
        "stream": rh.KernelStreamIdVisitor(),
    },
    file_name="./trace1.xlsx",
    sheet_name="trace-data",
)
```

AMD traces should use `device_type="amd"`. NVIDIA traces may use `device_type="nv"`, but NVIDIA analysis is not the reliable path for this workflow.

If the workspace already has `trace_to_xlsx.py`, copy it into the container:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
cp /workspace/trace_to_xlsx.py /tmp/trace_to_xlsx.py
'
```

## Export Excel

Run from the directory that contains `trace.json`:

```bash
docker exec rocm_halcyon_ubuntu22_test bash -lc '
set -euxo pipefail
cd /tmp
python3 trace_to_xlsx.py
cp /tmp/trace1.xlsx /workspace/trace1.xlsx
'
```

Verify from the host:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("/raid/users/jialzhu/code_backup/trace1.xlsx")
df = pd.read_excel(path, sheet_name="trace-data")
print(path, path.stat().st_size)
print(len(df), list(df.columns))
print(df.head(3).to_string(index=False))
PY
```

## Excel Field Semantics

Each row represents one GPU kernel execution extracted from the PyTorch profiler trace.

| Field | Meaning |
| --- | --- |
| `gpu_kernel_name` | GPU kernel symbol, such as HIP, CK, Triton, or collectives kernel names |
| `cpu_kernel_name` | Associated CPU-side PyTorch or custom op, such as `aten::mm` or `aiter::mha_batch_prefill` |
| `call_stack` | Inferred upper-level call stack from profiler events |
| `kernel_duration` | Kernel runtime in microseconds |
| `start_timestamp` | Trace start timestamp |
| `end_timestamp` | `start_timestamp + kernel_duration` |
| `input_shape` | Recorded op input shapes and scalar arguments when available |
| `stream` | GPU stream ID |

Missing `input_shape` or `cpu_kernel_name` usually means the parser could not match the GPU kernel to a CPU op, or the profiler trace did not record enough metadata.

## Analysis Snippets

Summarize durations and expensive CPU op groups:

```python
import pandas as pd

df = pd.read_excel("trace1.xlsx", sheet_name="trace-data")
print(df["kernel_duration"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
print(df["cpu_kernel_name"].fillna("<NA>").value_counts().head(20))
print(
    df.groupby(df["cpu_kernel_name"].fillna("<NA>"))["kernel_duration"]
    .agg(["size", "sum", "mean", "max"])
    .sort_values("sum", ascending=False)
    .head(20)
)
```

Interpret with these caveats:

- `kernel_duration` is per-kernel time in microseconds.
- Summing by `cpu_kernel_name` estimates aggregate GPU kernel contribution, not end-to-end wall time.
- Kernels on different streams can overlap.
- For one stream, use `start_timestamp`, `end_timestamp`, and `stream` to inspect ordering and gaps.
- Very large single-kernel durations should be cross-checked in Perfetto.

## Cleanup

Restore output ownership and remove the container when done:

```bash
docker exec rocm_halcyon_ubuntu22_test chown "$(id -u):$(id -g)" /workspace/trace1.xlsx
docker rm -f rocm_halcyon_ubuntu22_test
```

## Report Back

When reporting results, include:

- Input trace path and whether it was decompressed.
- Exact export script or visitor fields used.
- Output `.xlsx` path, sheet name, row count, and columns.
- Top expensive kernels or CPU op groups if analyzed.
- Stream-overlap and profiling-data caveats.
