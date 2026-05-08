---
name: atom-sglang-plugin
description: Configure, launch, verify, and troubleshoot ATOM as an external SGLang model plugin on AMD ROCm systems. Use when enabling SGLANG_EXTERNAL_MODEL_PACKAGE, validating ATOM model replacement, debugging silent fallback to native SGLang models, or developing ATOM plugin code inside Docker.
---

# ATOM SGLang Plugin

## Use This Skill When

Use this skill when the user wants to run SGLang with ATOM model implementations, verify that ATOM replaced the native SGLang model path, or debug why the plugin silently fell back.

The main success signal is a server log line like:

```text
[atom] ATOM model class for Qwen3_5ForConditionalGeneration is <class 'atom.models.qwen3_5.Qwen3_5ForCausalLM'>
```

Do not treat this line as sufficient:

```text
INFO - Platform plugin atom is activated
```

That only means the vLLM platform plugin was loaded. It does not prove model-level SGLang replacement succeeded.

## Operating Modes

ATOM can run in two modes:

- Standalone mode: ATOM serves the model directly.
- SGLang plugin mode: SGLang loads ATOM's external model package and uses ATOM model wrappers and optimized kernels.

Prefer SGLang plugin mode unless the user explicitly asks for standalone ATOM.

Standalone example:

```bash
python3 -m atom.entrypoints.openai_server \
  --model /path/to/model \
  --kv_cache_dtype fp8 \
  -tp 8
```

## Plugin Requirements

Before launch, verify:

- The environment contains both SGLang and ATOM.
- The AITER version matches the ATOM code being used.
- The model architecture is registered by ATOM.
- `SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models` is set.
- `--trust-remote-code` is included when required by the model.

ATOM plugin model registrations usually live under:

```text
ATOM/atom/plugin/sglang/models/
ATOM/atom/plugin/register.py
```

Common supported architectures include:

- `Qwen3_5ForConditionalGeneration`
- `Qwen3_5MoeForConditionalGeneration`
- `DeepseekV3ForCausalLM`
- `Qwen3MoeForCausalLM`
- `Qwen3NextForCausalLM`

Unregistered architectures can fall back to native SGLang implementations.

## Launch Template

Use this minimal plugin launch:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 8888 \
  --tensor-parallel-size 2 \
  --expert-parallel-size 2 \
  --attention-backend aiter \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --disable-radix-cache
```

Use these environment variables when running SGLang + AITER on ROCm:

```bash
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1
export SGLANG_ROCM_USE_AITER_LINEAR_FP8HIPB=1
export SGLANG_USE_AITER_NEW_CA=false
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
```

Optional ATOM-specific flag:

```bash
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
```

Only enable optional fusion flags when they are relevant to the model and ATOM build.

## Docker Launch Template

Use this pattern when the user wants to launch inside Docker:

```bash
docker run -d \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --network host \
  --privileged \
  -e HIP_VISIBLE_DEVICES=4,5 \
  -e CUDA_VISIBLE_DEVICES=4,5 \
  -e SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
  -e SGLANG_USE_AITER=1 \
  -e SGLANG_DISABLE_CUDNN_CHECK=1 \
  --mount type=bind,src=/raid/models,dst=/.cache/huggingface/ \
  image_name \
  python3 -m sglang.launch_server \
    --model-path /.cache/huggingface/Model-Name \
    --host 0.0.0.0 \
    --port 8888 \
    --tensor-parallel-size 2 \
    --attention-backend aiter \
    --trust-remote-code
```

For active ATOM development, mount the source tree and use `PYTHONPATH`:

```bash
docker run ... \
  -e "PYTHONPATH=/code_backup/ATOM:/opt/sglang/python" \
  --mount type=bind,src=/host/path/ATOM,dst=/code_backup/ATOM \
  ...
```

After editing host-side ATOM code, restart the container or server process so Python reloads the changed modules.

## Verify Plugin Activation

Check the server log immediately after model loading.

Success:

```text
[atom] ATOM model class for Qwen3_5ForConditionalGeneration is <class 'atom.models.qwen3_5.Qwen3_5ForCausalLM'>
```

Suspicious or insufficient:

```text
INFO - Platform plugin atom is activated
```

If the success line is missing, test whether the model entry class imports:

```bash
python3 -c "from atom.plugin.sglang.models.qwen3_5 import EntryClass; print(EntryClass)"
```

If this command fails, SGLang may have caught the `ImportError` and silently fallen back to native model code.

## Troubleshooting

When ATOM does not activate, check these in order:

1. Confirm `SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models` exists in the server process environment.
2. Import the target ATOM model entry class directly with `python3 -c`.
3. Check for AITER API mismatch, especially when using ATOM main branch with an older AITER package.
4. Remove stale installed ATOM files if editable install and `site-packages` conflict.
5. Verify the model architecture is registered by ATOM.

Clean stale ATOM installs with:

```bash
pip uninstall -y atom
rm -rf /usr/local/lib/python3.10/dist-packages/atom*
pip install -e ./ATOM
```

If editable install fails inside a mounted Docker directory, prefer `PYTHONPATH` over changing package source code or global Python configuration.

## Code Map

Use this map when locating plugin-related code:

```text
ATOM/atom/
├── plugin/
│   ├── sglang/
│   │   ├── models/
│   │   │   ├── qwen3_5.py
│   │   │   └── base_model_wrapper.py
│   │   ├── attention_backend/
│   │   └── utils/
│   ├── vllm/
│   └── register.py
├── models/
├── model_ops/
└── config.py
```

## Report Back

When finishing an ATOM plugin investigation, report:

- The exact server launch command and relevant environment variables.
- Whether the `[atom] ATOM model class` log line appeared.
- Any direct import errors from `atom.plugin.sglang.models.*`.
- The AITER and ATOM source/version relationship if known.
- Whether the model architecture is registered or fell back to native SGLang.
