---
name: inferencex-local-vllm-bench
description: Run local InferenceX vLLM benchmarks on AMD MI355X/ROCm for large models such as DeepSeek-V4-Pro. Use when asked to validate benchmark numbers with InferenceX, run DP/TP/EP sweeps, compare hardware/backend configurations, or reproduce local vLLM serving performance.
---

# InferenceX Local vLLM Benchmark

Use this skill to run **InferenceX locally** on a single GPU node, without GitHub Actions or Slurm. The known-good local target in this repo is DeepSeek-V4-Pro on 8x MI355X with vLLM.

## Golden Rules

- Use `/home/jialzhu/deepseekv4/InferenceX` and InferenceX's own benchmark client for local validation.
- Do **not** kill unrelated containers. Before launching, identify GPU owners with `docker ps`, `rocm-smi --showpidgpus --showuse --showmemuse`, and `ps`.
- Prefer `tmux` for long benchmark runs so they survive Cursor/SSH interruption.
- Preserve logs and result JSONs. Always write server logs, benchmark logs, and a `summary.json`.
- Keep model, image, server args, ISL/OSL, concurrency, prompt count, warmup count, and chat-template flags aligned when comparing frameworks.

## Important Local Paths

Repository root:

```bash
/home/jialzhu/deepseekv4
```

InferenceX checkout:

```bash
/home/jialzhu/deepseekv4/InferenceX
```

DeepSeek-V4-Pro local model path inside the InferenceX benchmark container:

```bash
/.cache/huggingface/deepseek-v4-pro
```

Host mount used to expose models:

```bash
/mnt -> /.cache/huggingface
```

Known working Docker image for MI355 local tests:

```bash
sabreshao/vllm:dsv4_0605
```

Known working local script:

```bash
/home/jialzhu/deepseekv4/InferenceX/run_dp8_aiter_100k_tmux.sh
```

Example output directory:

```bash
/home/jialzhu/deepseekv4/InferenceX/local_runs/run_dp8_aiter_100k_20260611_083459
```

## Preflight Checks

Run these before launching a large model:

```bash
tmux ls 2>/dev/null || true
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
rocm-smi --showuse --showmemuse --showpidgpus
```

If GPU memory is occupied, map PIDs back to commands:

```bash
ps -eo pid,ppid,user,cmd | rg 'vllm|python|docker run|benchmark|sglang|inferencex|aiter' || true
```

Only stop containers that are clearly created by the current task, such as `inferencex-dsv4-dp8-aiter`. Ask before stopping other users' containers.

## Container Launch Template

Use a unique container name and port per run.

```bash
docker rm -f inferencex-dsv4-dp8-aiter 2>/dev/null || true

docker run -dit \
  --name inferencex-dsv4-dp8-aiter \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v /mnt:/.cache/huggingface \
  -v /home/jialzhu:/dockerx \
  -v /home/jialzhu/deepseekv4/InferenceX:/workspace \
  --privileged \
  --ipc=host \
  --network=host \
  --shm-size=128g \
  --tmpfs /model_ram:size=1024G,mode=1777 \
  --entrypoint /bin/bash \
  -w /workspace \
  sabreshao/vllm:dsv4_0605 \
  -lc 'while true; do sleep 3600; done'
```

## vLLM Server: MI355 DP8 + EP + AITER

Inside the container:

```bash
export PORT=19091
export MODEL='/.cache/huggingface/deepseek-v4-pro'
export HF_HOME='/.cache/huggingface/'
export HF_HUB_CACHE='/.cache/huggingface'
export SNAP='/.cache/huggingface/deepseek-v4-pro/'
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH='/app/vllm:'${PYTHONPATH:-}

cd /workspace
source benchmarks/benchmark_lib.sh

RUN_DIR="/workspace/local_runs/run_dp8_aiter_100k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
SERVER_LOG="$RUN_DIR/server.log"
GPU_METRICS_CSV="$RUN_DIR/gpu_metrics.csv"
export GPU_METRICS_CSV

start_gpu_monitor --output "$GPU_METRICS_CSV"

vllm serve "$MODEL" --port "$PORT" \
  --host localhost \
  --dtype auto \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --block-size 256 \
  --no-enable-prefix-caching \
  --max-model-len 102656 \
  --max-num-batched-tokens 2048 \
  --distributed-executor-backend mp \
  --trust-remote-code \
  --gpu-memory-utilization 0.6 \
  --moe-backend aiter \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --kv-cache-dtype fp8 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","pass_config":{"fuse_allreduce_rms":true}}' \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
```

Important: for DeepSeek-V4 chat-template benchmarking with InferenceX `benchmark_serving.py`, pass both `--use-chat-template` and `--dsv4`. Passing only `--dsv4` raises:

```text
ValueError: --dsv4 requires --use-chat-template to be set.
```

## Benchmark Command Template

Use InferenceX's benchmark client:

```bash
python3 /workspace/utils/bench_serving/benchmark_serving.py \
  --backend vllm \
  --base-url "http://0.0.0.0:$PORT" \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 102400 \
  --random-output-len 1 \
  --random-range-ratio 1.0 \
  --num-prompts "$PROMPTS" \
  --num-warmups 2 \
  --request-rate inf \
  --max-concurrency "$CONC" \
  --burstiness 1.0 \
  --trust-remote-code \
  --use-chat-template \
  --dsv4 \
  --save-result \
  --percentile-metrics ttft,tpot,itl,e2el \
  --result-dir "$CASE_DIR" \
  --result-filename "$RESULT_FILE" \
  --ignore-eos
```

For the 100K/1 DP8 sweep, run:

```text
CONC=1  PROMPTS=2
CONC=2  PROMPTS=4
CONC=4  PROMPTS=8
CONC=8  PROMPTS=16
CONC=16 PROMPTS=32
```

## Run in tmux

Use this if the run may take a while:

```bash
cd /home/jialzhu/deepseekv4/InferenceX
tmux new-session -d -s inferencex_dp8_aiter_100k \
  './run_dp8_aiter_100k_tmux.sh 2>&1 | tee /home/jialzhu/deepseekv4/inferencex_dp8_aiter_100k_tmux.log'
```

Check progress:

```bash
tmux attach -t inferencex_dp8_aiter_100k
```

Check completion without attaching:

```bash
tmux ls 2>/dev/null || true
tail -n 120 /home/jialzhu/deepseekv4/inferencex_dp8_aiter_100k_tmux.log
```

## Summarize Results

After all cases complete, create/read:

```bash
$RUN_DIR/summary.json
```

Useful fields:

- `completed`
- `duration`
- `total_token_throughput`
- `output_throughput`
- `mean_ttft_ms`
- `median_ttft_ms`
- `p99_ttft_ms`
- `mean_tpot_ms`

Known MI355 DP8 + EP + AITER result from this workspace:

```text
case_01 conc=1  prompts=2   TTFT=32499.30 ms  total=3150.78 tok/s
case_02 conc=2  prompts=4   TTFT=33786.64 ms  total=6061.39 tok/s
case_03 conc=4  prompts=8   TTFT=38067.93 ms  total=10653.95 tok/s
case_04 conc=8  prompts=16  TTFT=41733.91 ms  total=19481.79 tok/s
case_05 conc=16 prompts=32  TTFT=70378.62 ms  total=20107.85 tok/s
```

## Compare Against B300

B300 MegaMoE reference from `B300-RECIPE-MEGAMOE.MD`:

```text
TP=1, DP=8, enable_expert_parallel, moe_backend=deep_gemm_mega_moe
block_size=256, max_model_len=102656, max_num_batched_tokens=2048
```

Reference results:

```text
CON1  PROMPT2   TTFT=10958.29 ms  total=9344.30 tok/s
CON2  PROMPT4   TTFT=10232.86 ms  total=19729.07 tok/s
CON4  PROMPT8   TTFT=10612.47 ms  total=38005.05 tok/s
CON8  PROMPT16  TTFT=10955.93 ms  total=70867.44 tok/s
CON16 PROMPT32  TTFT=18998.92 ms  total=74147.16 tok/s
```

When comparing MI355 and B300, state clearly that B300 uses `deep_gemm_mega_moe`, while MI355 local run uses `aiter`. Do not attribute the gap to InferenceX without checking serving backend differences.

## DP Dummy Batch Notes

vLLM has a DP dummy batch mechanism. If one DP rank has no unfinished real request but the global DP group still has unfinished requests, that rank can execute a dummy batch to keep DP ranks synchronized.

Source locations in the local vLLM tree:

- `vllm/v1/engine/llm_engine.py`: `LLMEngine.has_unfinished_requests_dp()` sets `should_execute_dummy_batch=True`; `LLMEngine.step()` calls `engine_core.execute_dummy_batch()`.
- `vllm/v1/engine/core.py`: DP busy loop calls `execute_dummy_batch()` when the engine is in a running state but did not execute a real ready request.
- `vllm/v1/worker/gpu_worker.py`: `GPUWorker.execute_dummy_batch()` calls `model_runner._dummy_run(...)`.
- `vllm/v1/worker/gpu_model_runner.py`: `_dummy_run()` and `maybe_randomize_inputs()` describe DP dummy runs.
- `vllm/v1/worker/dp_utils.py`: `coordinate_batch_across_dp()` synchronizes token counts, cudagraph mode, and DP padding across ranks.

Important: dummy batches are not real benchmark prompts. They do not count toward `completed` requests or real input throughput.

## Cleanup

The provided script has a trap that removes its own container:

```bash
docker rm -f inferencex-dsv4-dp8-aiter
```

If a run fails before cleanup, remove only the known task container. Do not remove unrelated user containers.

Check GPU release:

```bash
rocm-smi --showuse --showmemuse --showpidgpus
```
