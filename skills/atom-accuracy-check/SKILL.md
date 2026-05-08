---
name: atom-accuracy-check
description: Verify that ATOM kernel modifications (Triton/HIP fused kernels, RMSNorm, MRoPE, GDN etc.) do not introduce accuracy regressions. Use when modifying ATOM model_ops or models code and need to confirm output quality before submitting a PR.
---

# ATOM Accuracy Check

## Use This Skill When

Use this skill after modifying ATOM kernel code (e.g., `layernorm.py`, `attention_gdn.py`, `qwen3_next.py`) and before submitting a PR. It validates that the model still produces coherent, correct outputs across multiple categories.

## Prerequisites

- A working Docker image with SGLang + ATOM installed (e.g., `atom-sglang:latest`)
- The modified ATOM code accessible (either built into the image or mounted via PYTHONPATH)
- At least one idle GPU pair (for TP2 models like Qwen3.5-27B)
- The model weights accessible (e.g., mounted from `/raid/models`)

## Workflow

### Step 1: Start a Server with the Modified Code

Launch a container with the modified ATOM code and start a server. Key points:

- Use `--skip-server-warmup` to avoid getting stuck on warmup health checks
- Use a unique port (e.g., 9999) to avoid conflicts
- Use a unique container name to avoid conflicts with other users
- Mount the modified ATOM code via PYTHONPATH, not `pip install`

```bash
docker run -d \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --user root --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --pid=host --network host --privileged \
  -e HIP_VISIBLE_DEVICES=4,5 -e CUDA_VISIBLE_DEVICES=4,5 \
  -e HF_HOME=/.cache/huggingface/ \
  -e SGLANG_DISABLE_CUDNN_CHECK=1 \
  -e SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1 \
  -e SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1 \
  -e SGLANG_USE_AITER_NEW_CA=false \
  -e PYTHONPATH=/code_backup/ATOM:/opt/sglang/python \
  --mount type=bind,src=/raid/models,dst=/.cache/huggingface/ \
  --mount type=bind,src=/path/to/modified/ATOM,dst=/code_backup/ATOM \
  --name atom_accuracy_test \
  atom-sglang:latest bash -c "python3 -m sglang.launch_server \
    --model-path /.cache/huggingface/Qwen3.5-27B \
    --host 0.0.0.0 --port 9999 \
    --tensor-parallel-size 2 --expert-parallel-size 2 \
    --attention-backend aiter --trust-remote-code \
    --mem-fraction-static 0.9 --disable-radix-cache \
    --disable-custom-all-reduce --skip-server-warmup \
    > /tmp/server.log 2>&1"
```

### Step 2: Wait for Server Ready

The server typically takes 3-5 minutes to become healthy (model loading + CUDA graph capture):

```bash
for i in $(seq 1 60); do
  if docker exec atom_accuracy_test curl -s -f "http://0.0.0.0:9999/health" >/dev/null 2>&1; then
    echo "READY after $((i*5))s"
    break
  fi
  sleep 5
done
```

**Important**: Also verify that ATOM is actually loaded, not silently falling back to SGLang native:

```bash
docker exec atom_accuracy_test grep "ATOM model class" /tmp/server.log
# Should see: [atom] ATOM model class for Qwen3_5ForConditionalGeneration is <class 'atom.models.qwen3_5.Qwen3_5ForCausalLM'>
```

If this line is missing, ATOM plugin did not load. Check for import errors:

```bash
docker exec atom_accuracy_test grep -E "Error|ImportError|ModuleNotFoundError" /tmp/server.log | head -5
```

### Step 3: Run Accuracy Tests

Send multiple test prompts covering different capabilities. Use `enable_thinking: false` to get direct answers without the thinking token overhead:

```bash
PORT=9999; CTR=atom_accuracy_test

# Test 1: Simple Math
echo "[Test 1 - Math]"
docker exec $CTR curl -s --max-time 120 "http://0.0.0.0:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"/.cache/huggingface/Qwen3.5-27B","messages":[{"role":"user","content":"1+1等于几？请直接回答，不要思考过程"}],"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}' \
| python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Test 2: Code Generation
echo "[Test 2 - Code]"
docker exec $CTR curl -s --max-time 120 "http://0.0.0.0:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"/.cache/huggingface/Qwen3.5-27B","messages":[{"role":"user","content":"Write a Python function to check if a number is prime. Keep it short."}],"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}' \
| python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Test 3: Chinese Text
echo "[Test 3 - Chinese]"
docker exec $CTR curl -s --max-time 120 "http://0.0.0.0:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"/.cache/huggingface/Qwen3.5-27B","messages":[{"role":"user","content":"请用三句话总结快速排序算法的核心思想"}],"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}' \
| python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"

# Test 4: Logical Reasoning
echo "[Test 4 - Logic]"
docker exec $CTR curl -s --max-time 120 "http://0.0.0.0:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"/.cache/huggingface/Qwen3.5-27B","messages":[{"role":"user","content":"If all roses are flowers, and all flowers need water, do roses need water? Answer yes or no and explain briefly."}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}' \
| python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

### Step 4: Evaluate Results

**PASS criteria** — all of these must hold:

| Test | Expected |
|------|----------|
| Math | Correct answer "2" |
| Code | Syntactically valid Python function with correct logic |
| Chinese | Coherent 3-sentence summary in Chinese |
| Logic | "Yes" with valid syllogistic reasoning |

**FAIL indicators** — any of these means accuracy regression:

- Garbled/nonsensical output (e.g., `codexfunction(n`, `快速，选择选`)
- Truncated responses that stop mid-sentence
- Completely wrong answers to simple questions
- Empty responses or error messages

### Step 5: Cleanup

```bash
docker rm -f atom_accuracy_test 2>/dev/null
```

## Known Issues

### `--reasoning-parser qwen3` can cause hangs

If the server is started with `--reasoning-parser qwen3`, the model may enter thinking mode and generate very long responses, causing curl to timeout. Either:
- Remove `--reasoning-parser qwen3` from SERVER_ARGS, or
- Add `"chat_template_kwargs":{"enable_thinking":false}` to the request (as shown above)

### ATOM plugin silently fails to load

SGLang's model registry catches ImportError when loading external model packages and silently falls back to native models. Always check the server log for `[atom] ATOM model class` to confirm ATOM is active. Common failure causes:
- AITER version mismatch (missing symbols like `gated_rmsnorm_fp8_group_quant`)
- Old ATOM installation residue in `site-packages/atom/`
- `pip install -e` failure in read-only mounted volumes (use PYTHONPATH instead)

### fused GDN kernel accuracy issue

The `fused_sigmoid_gating_delta_rule_update` kernel (replacing `fused_recurrent_gated_delta_rule` in decode path) was found to produce garbled outputs. The two kernels are NOT numerically equivalent. Always run this accuracy check after modifying GDN-related code.

## Example: Real Failure Case

When testing `fused_sigmoid_gating_delta_rule_update` in `attention_gdn.py`:

```
[Test 1 - Math] 2                    ← OK
[Test 2 - Code] codexfunction(n      ← FAIL: garbled
[Test 3 - Chinese] 快速，选择选        ← FAIL: garbled
[Test 4 - Logic] Yes                  ← Partial: no explanation
```

This correctly identified an accuracy regression in the fused GDN kernel, preventing a bad PR from being submitted.
