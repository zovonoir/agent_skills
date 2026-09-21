---
name: rocm-halcyon-optiq-excel
description: Set up rocm-halcyon, parse ROCpd databases produced by rocprofv3 (the files ROCm Optiq opens), export GPU kernel data to Excel, and explain the resulting fields. Use when converting a ROCpd .db to .xlsx, running parse_optiq_trace, or analyzing kernel timing from an Optiq trace instead of a Torch Profiler trace.
---

# rocm-halcyon Optiq Excel

## Use This Skill When

Use this skill when the user wants an agent to turn a rocprofv3 ROCpd database into Excel using `rocm-halcyon`, then explain the exported GPU kernel rows.

Default behavior:

- Work in a temporary clone or container rather than modifying the user's source tree.
- Parse a merged multi-rank `.db` with `rh.parse_optiq_trace`.
- Export one row per GPU kernel to `.xlsx`.
- Explain duration, timestamp, gap, device, pid, and stream fields.

Choose this path over the Torch Profiler path when CUDA Graph must stay enabled. Torch Profiler needs `--disable-cuda-graph` to show individual kernels, which changes the configuration under study. rocprofv3 records kernels inside Graph replay, so the trace describes the real serving path.

Choose the Torch Profiler path instead when CPU op names, input shapes, or Python call stacks are required. This parser cannot provide them; see the field table below.

## Producing The Input

The expected input is one merged, origin-anchored database from the `rocm-optiq-trace` skill. Collect and merge there first, then come back here.

Two properties of that database matter:

- All ranks must share one time origin. Otherwise every rank carries its own process start offset and cross-rank comparison is wrong.
- Multiple ranks should already be merged into one file.

A single-rank database parses too. The parser emits a warning so the operator knows the timeline covers only one GPU.

Relabeled ROCTx markers do not affect this export. They matter in the Optiq viewer, not here.

## Clean Ubuntu Setup

Use Docker when the user wants a reproducible clean environment:

```bash
docker run -d --name rocm_halcyon_optiq_test \
  -v "/raid/users/jialzhu/qwen3.8/profiles:/workspace" \
  -w /tmp \
  ubuntu:22.04 sleep infinity
```

Install system and Python dependencies:

```bash
docker exec rocm_halcyon_optiq_test bash -lc '
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates git build-essential cmake ninja-build \
  python3 python3-dev python3-pip
python3 -m pip install --no-cache-dir wheel setuptools
python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-cache-dir tqdm openpyxl pandas parse perfetto
'
```

Dependency notes:

- `build-essential`, `cmake`, `ninja-build`, and `python3-dev` build `halcyon_core`.
- `torch` is needed because `rocm_halcyon.__init__` imports `profiler_bridge.py`.
- `parse` and `perfetto` are imported by `interfaces.py` even on this path.
- `tqdm` is used by the parser, `openpyxl` by `pandas.to_excel()`.
- `sqlite3` is in the standard library, so the ROCpd reader itself adds no dependency.

## Clone And Install rocm-halcyon

Clone with submodules:

```bash
docker exec rocm_halcyon_optiq_test bash -lc '
set -euxo pipefail
rm -rf /tmp/rocm-halcyon
git clone --recursive https://github.com/zovonoir/rocm-halcyon /tmp/rocm-halcyon
test -f /tmp/rocm-halcyon/external/pybind11/CMakeLists.txt
'
```

Build and install:

```bash
docker exec rocm_halcyon_optiq_test bash -lc '
set -euxo pipefail
cd /tmp/rocm-halcyon
rm -rf build dist rocm_halcyon.egg-info
rm -f rocm_halcyon/halcyon_core*.so
python3 setup.py bdist_wheel
python3 -m pip install --no-cache-dir dist/*.whl --force-reinstall
cd /tmp && python3 -c "
import rocm_halcyon as rh
print(\"rocm_halcyon import ok\", rh.__file__)
print(\"parse_optiq_trace:\", hasattr(rh, \"parse_optiq_trace\"))
"
'
```

Run the import check from a directory other than the repository root. Inside the repo the local `rocm_halcyon/` package shadows the installed one and fails on the missing `halcyon_core` extension.

### Building on a host instead of a container

When the host already has `cmake`, `ninja`, `g++`, and `torch`, install into an isolated target directory rather than the user's environment:

```bash
python3 -m pip install --no-cache-dir --target /tmp/halcyon_deps tqdm parse perfetto
cp -r /path/to/rocm-halcyon /tmp/halcyon_build
cd /tmp/halcyon_build && PYTHONPATH=/tmp/halcyon_deps python3 setup.py bdist_wheel
python3 -m pip install --no-cache-dir --target /tmp/halcyon_deps dist/*.whl --upgrade
```

Then run every command with `PYTHONPATH=/tmp/halcyon_deps`. Do not create a venv on top of a venv with `--system-site-packages`; it inherits the base interpreter's packages, not the parent venv's, so an already installed `torch` disappears.

## Export Script

`parse_optiq_trace` returns the same `Kernel` list as `parse_torch_profiler`, so `export_to_excel` and every visitor work unchanged. Use this script as `/tmp/optiq_to_xlsx.py`:

```python
import rocm_halcyon as rh

ops = rh.parse_optiq_trace("./unified.db")

rh.export_to_excel(
    ops,
    {
        "gpu_kernel_name": rh.GPUKernelNameVisitor(),
        "kernel_duration": rh.KernelDurationVisitor(),
        "start_timestamp": rh.KernelStartTimestampVisitor(),
        "end_timestamp": rh.KernelEndTimestampVisitor(),
        "kernel_gap": rh.KernelGapVisitor(),
        "device_id": rh.KernelDeviceVisitor(),
        "pid": rh.KernelProcessIdVisitor(),
        "stream": rh.KernelStreamIdVisitor(),
    },
    file_name="./optiq.xlsx",
    sheet_name="optiq-data",
)
```

Do not add the shape, call-stack, or launch-cost visitors here. They exist for the Torch Profiler path and would only produce empty columns.

## Export Excel

```bash
docker exec rocm_halcyon_optiq_test bash -lc '
set -euxo pipefail
cd /tmp
cp /workspace/optiq/unified.db /tmp/unified.db
python3 optiq_to_xlsx.py
cp /tmp/optiq.xlsx /workspace/optiq.xlsx
'
```

Verify from the host:

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_excel("/raid/users/jialzhu/qwen3.8/profiles/optiq.xlsx", sheet_name="optiq-data")
print(len(df), list(df.columns))
print("globally sorted by start:", df.start_timestamp.is_monotonic_increasing)
print(df.groupby(["device_id", "pid", "stream"]).size())
print(df.head(3).to_string(index=False))
PY
```

Expect roughly 5 seconds for 45000 kernels. Row count should equal `SELECT COUNT(*) FROM rocpd_kernel_dispatch` in the source database.

## Excel Field Semantics

Each row is one GPU kernel dispatch.

| Field | Meaning |
| --- | --- |
| `gpu_kernel_name` | GPU kernel symbol, such as HIP, CK, Triton, or collectives kernel names |
| `kernel_duration` | Kernel runtime in microseconds, converted from the database's nanoseconds |
| `start_timestamp` | Microseconds since the earliest event in the whole trace, across all ranks |
| `end_timestamp` | `start_timestamp + kernel_duration` |
| `kernel_gap` | Microseconds between this kernel's start and the previous kernel's end on the same `(device_id, stream)`; 0 for the first kernel of each |
| `device_id` | GPU index, from `agent_log_index` |
| `pid` | Scheduler process that issued the kernel, which is what distinguishes ranks |
| `stream` | GPU stream ID |

Rows are sorted by `start_timestamp` across all devices and streams, so consecutive rows usually belong to different GPUs. Filter by `device_id` or `stream` to get one lane; filtering preserves time order.

`start_timestamp` normally does not begin at 0. The origin is the earliest event of any kind, and the first HIP API call precedes the first kernel. This keeps Excel timestamps on the same basis as the Optiq viewer, so the two can be read side by side.

### Fields this path cannot provide

The following stay empty for every row, so do not export them and do not report them as missing data:

`cpu_op_name`, `torch_op_name`, `call_stack`, `input_shape`, `output_shape`, `input_dtype`, `output_dtype`, `input_strides`, `output_strides`, `grid`, `block`, `smem`, `host_launching_cost`, `annotation`, `user_annotation`, `module_name`, `module_type`, `module_depth`, `parent_module`, `correlation`.

Two reasons. Shape and CPU op metadata only exist in a Torch Profiler trace. And `correlation` is deliberately left empty because rocprofv3 1.1.0 writes zero into `correlation_id` for every event, so a correlation-based join would collapse the whole trace onto one arbitrary record.

To attribute kernels to model source from a ROCpd database, use the ROCTx marker route in the `rocm-optiq-trace` skill rather than looking for these columns.

## Analysis Snippets

Rank up the expensive kernels and compare the two ranks:

```python
import pandas as pd

df = pd.read_excel("optiq.xlsx", sheet_name="optiq-data")

print(df.groupby("gpu_kernel_name")["kernel_duration"]
        .agg(["size", "sum", "mean", "max"])
        .sort_values("sum", ascending=False)
        .head(20))

print(df.pivot_table(index="gpu_kernel_name", columns="device_id",
                     values="kernel_duration", aggfunc="sum")
        .sort_values(df.device_id.min(), ascending=False)
        .head(20))
```

Find scheduling holes on one lane:

```python
lane = df[(df.device_id == 2) & (df.stream == 12)]
print(lane.nlargest(20, "kernel_gap")[
    ["gpu_kernel_name", "start_timestamp", "kernel_gap"]
])
```

Interpret with these caveats:

- Summing `kernel_duration` estimates aggregate GPU kernel contribution, not end-to-end wall time.
- Kernels on different devices or streams overlap, so never sum across lanes to get elapsed time.
- Profiling adds overhead. An open rocprofv3 collection window cost about 1.9% on a short-decode case and about 10.2% on a long-decode case, so absolute times are not serving benchmarks.
- Large single-kernel durations should be cross-checked in Optiq.

## Cleanup

```bash
docker exec rocm_halcyon_optiq_test chown "$(id -u):$(id -g)" /workspace/optiq.xlsx
docker rm -f rocm_halcyon_optiq_test
```

For a host build, remove `/tmp/halcyon_build` and `/tmp/halcyon_deps`.

## Report Back

When reporting results, include:

- Input database path, whether it was merged and origin-anchored, and how many ranks it holds.
- Exact export script or visitor fields used.
- Output `.xlsx` path, sheet name, row count, and columns.
- Row count agreement with `rocpd_kernel_dispatch` in the source database.
- Top expensive kernels or per-device comparison if analyzed.
- Overlap and profiling-overhead caveats.
