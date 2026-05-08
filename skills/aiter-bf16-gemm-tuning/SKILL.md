---
name: aiter-bf16-gemm-tuning
description: Tune missing AITER BF16 GEMM shapes for SGLang or ATOM inference workloads on AMD ROCm systems. Use when collecting bf16_untuned_gemm.csv, running gemm_tuner.py, applying tuned_gemm CSV files, debugging "not found tuned config" warnings, or validating tuned GEMM coverage.
---

# AITER BF16 GEMM Tuning

## Use This Skill When

Use this skill when the user needs to improve AITER `TunedGemm` coverage for BF16 inference workloads.

The usual symptom is a server log warning like:

```text
[aiter] shape is M:128, N:5120, K:8704 ... not found tuned config ... using torch solution:0
```

Missing tuned configs can fall back to `torch.nn.functional.linear` / `aten::mm`, which is often slower than a tuned hipBLASLt, asm, skinny, or Triton solution.

## Workflow

Follow this order:

1. Collect missing GEMM shapes from the target workload.
2. Run `gemm_tuner.py` to produce a tuned CSV.
3. Apply the tuned CSV to AITER.
4. Restart inference and verify the warnings disappear.

Do not mix tuned CSVs across GPU models. The `cu_num` and GPU architecture are part of the lookup key.

## Collect GEMM Shapes

Prefer automatic collection with `AITER_TUNE_GEMM=1`:

```bash
AITER_TUNE_GEMM=1 python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 8888 \
  --tensor-parallel-size 2 \
  ...
```

After the server starts, send real requests. Startup alone may only collect CUDA Graph capture shapes.

Use at least:

- One long-sequence request to trigger prefill large-`M` shapes.
- A few short requests to trigger decode small-`M` shapes.

The collected shapes are written to:

```text
aiter/configs/bf16_untuned_gemm.csv
```

If the process is short-lived or the file is lost, extract shapes from logs:

```bash
rg "not found tuned config" server.log \
| python3 -c "
import re
import sys

shapes = set()
for line in sys.stdin:
    match = re.search(r'M:(\d+), N:(\d+), K:(\d+)', line)
    if match:
        shapes.add(tuple(map(int, match.groups())))

print('M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle')
for m, n, k in sorted(shapes):
    print(f'{m},{n},{k},False,torch.bfloat16,torch.bfloat16,False,False')
" > bf16_untuned_gemm.csv
```

The boolean columns must be Python-style `True` / `False`, not quoted strings. Prefer the automatically generated CSV when possible.

## Run The Tuner

Run the tuner from AITER's `gradlib` directory:

```bash
cd /path/to/aiter/gradlib/gradlib

python3 gemm_tuner.py \
  -i /path/to/bf16_untuned_gemm.csv \
  -o /path/to/bf16_tuned_gemm.csv \
  --mp 1
```

Use multiple GPUs when available:

```bash
python3 gemm_tuner.py \
  -i /path/to/bf16_untuned_gemm.csv \
  -o /path/to/bf16_tuned_gemm.csv \
  --mp 8
```

Expected runtime:

- Each shape can take roughly 3 to 10 minutes.
- 300 shapes can take 20 to 30 hours on one GPU.
- 8 GPUs can reduce that to roughly 3 to 4 hours.

Common tuner arguments:

| Argument | Meaning |
| --- | --- |
| `-i`, `--input_file` | Untuned shape CSV |
| `-o`, `--tuned_file` | Tuned result CSV |
| `--mp` | Number of GPUs for parallel tuning |
| `--all` | Retune all shapes, including existing tuned shapes |
| `--verbose` | Print detailed tuner logs |
| `--errRatio` | Numerical error tolerance, default often `0.05` |
| `--sort` | Sort output |

## Apply Tuned Config

Prefer placing the result in AITER's model config directory:

```bash
cp bf16_tuned_gemm.csv /path/to/aiter/aiter/configs/model_configs/mymodel_bf16_tuned_gemm.csv
```

AITER scans `configs/model_configs/` for `*bf16_tuned_gemm*.csv` and merges them into the main config.

Alternatively, pass one or more config files with `AITER_CONFIG_GEMM_BF16`:

```bash
export AITER_CONFIG_GEMM_BF16="/path/to/bf16_tuned_gemm.csv"
export AITER_CONFIG_GEMM_BF16="/path/a.csv:/path/b.csv"
```

For Docker injection:

```bash
docker run ... \
  -v /host/path/tuned.csv:/workspace/tuned.csv:ro \
  image bash -c "
    cp /workspace/tuned.csv /opt/aiter/aiter/configs/model_configs/mymodel_bf16_tuned_gemm.csv
    python3 -m sglang.launch_server ...
  "
```

## Validate Coverage

Restart the inference service and inspect logs:

```bash
rg -c "not found tuned config" server.log
```

Success means the relevant workload no longer emits missing-config warnings for the shapes you tuned.

If warnings remain:

- Confirm the tuned CSV is loaded by the same Python environment that runs inference.
- Confirm the GPU model and `cu_num` match the machine where tuning was performed.
- Confirm request shapes match the collected shapes.
- Collect and tune the newly reported shapes incrementally.

## Tuned CSV Format

A tuned CSV row usually looks like:

```csv
cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw
80,1,5120,3072,False,torch.bfloat16,torch.bfloat16,False,False,hipblaslt,204196,0,12.34,Cijk_...,0.0,2.55,2551.93
```

Important fields:

- `cu_num`: compute unit count; GPU-model specific.
- `libtype`: backend such as `hipblaslt`, `asm`, `skinny`, or `triton`.
- `solidx`: hipBLASLt solution index.
- `us`: measured runtime in microseconds.

## Merge Incremental Results

When combining tuned CSVs, keep one header:

```bash
head -1 a.csv > merged.csv
tail -n +2 a.csv >> merged.csv
tail -n +2 b.csv >> merged.csv
```

Only merge files for the same GPU model and compatible AITER version unless the user explicitly wants a combined archive.

## Alternative: HIP Online Tuning

`HIP_ONLINE_TUNING=1` can tune hipBLASLt algorithm selection during first inference:

```bash
export HIP_ONLINE_TUNING=1
```

Use this only as a fallback or quick experiment. It does not cover the full AITER offline search space, such as skinny, asm, and Triton backends.

## Report Back

When reporting results, include:

- The workload used to collect shapes.
- Number of untuned shapes collected.
- Tuner command and GPU count.
- Tuned CSV path.
- Validation result from missing-config log checks.
- Any GPU model or AITER-version caveats.
