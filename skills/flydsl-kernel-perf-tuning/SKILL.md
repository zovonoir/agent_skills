---
name: flydsl-kernel-perf-tuning
description: Debug, benchmark and optimize FlyDSL GPU kernels on AMD Instinct (gfx942 MI300/MI308X, also gfx950) - correctness harness, pure-GPU timing with rocprofv3, final-ISA instruction/register/LDS analysis, occupancy estimates, hardware counters (LDS bank conflicts, VMEM instructions), one-change-at-a-time ablations and gain ceilings. Use when writing or tuning a FlyDSL (flydsl, fx., flyc.kernel) kernel, comparing it with Triton/other kernels, finding its bottleneck, or when FlyDSL layouts, MFMA fragments, tiled copies, LDS DMA or runtime loops behave unexpectedly.
---

# FlyDSL Kernel Performance Tuning

A measurement-driven loop for making a FlyDSL kernel correct first, then fast. Every
claim about a bottleneck must come from a number (time, ISA count, counter), and every
optimization is kept or reverted on a measured result.

## Non-negotiable rules

- **Correctness before speed.** No timing until the kernel passes a reference
  comparison that includes edge cases. Re-run it after every change.
- **Time with `rocprofv3 --kernel-trace`, not CUDA events.** Events include host
  launch overhead (~60 us per FlyDSL launch); a 74 us kernel measured as 186 us.
  Use events only for kernels well above 0.5 ms and cross-check once.
- **One change per measurement.** Keep the previous version (copy the file) so a
  change that does not pay off is reverted, not stacked.
- **Give every kernel a distinct name** before profiling; identical names merge in the
  trace.
- **Do not reinstall flydsl / aiter / torch** in the container. Report a broken
  environment instead.
- Run inside the FlyDSL container (repo mounted, e.g. `docker exec -w /work fly-dev ...`).

## Workflow

```
Progress:
- [ ] 1. Correctness harness (reference, edge cases, sentinel-filled outputs)
- [ ] 2. Baseline timing on realistic + stress shapes, next to reference kernels
- [ ] 3. Final-ISA stats: instruction mix, VGPR/AGPR/LDS, occupancy
- [ ] 4. Hardware counters, as ratios against the reference kernel
- [ ] 5. Rank hypotheses, then one ablation per hypothesis
- [ ] 6. Isolate fixed costs with constructed inputs
- [ ] 7. Estimate the ceiling of the next idea; stop when it is small
- [ ] 8. Integrate, re-verify routing and serving-shaped inputs
```

### 1. Correctness harness

- Write a plain PyTorch reference that states the semantics.
- Cases: sizes not multiple of the tile, several requests (packed and interleaved),
  invalid ids (-1), unmapped pages, capped lengths, nothing visible, one large real
  shape.
- Fill outputs with sentinels (`NaN`, `-7`) so unwritten entries show up. If the
  kernel legitimately skips regions (e.g. past a causal horizon the consumer never
  reads), compare only the region the consumer reads and report the skipped fraction.
- Add a `--case N` switch so one case can be run with prints.

### 2. Timing

```bash
rocprofv3 --kernel-trace --output-format csv -d /tmp/rp -o run -- python bench.py
python scripts/rp_kernel_times.py /tmp/rp --match my_kernel --split <num_shapes>
```

- The bench script loops shapes in a fixed order, runs a correctness check, warmup
  (~20) and timed iterations (~100) per shape, with outputs preallocated.
- Shapes: real serving shapes, plus one that stresses each feature (e.g. causal ramp
  for early exit, interleaved requests for per-request passes).
- Always time the reference kernels (existing FlyDSL/Triton/CK) on the same shapes.

### 3. Final ISA

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/dump python test.py --case 0
python scripts/isa_stats.py /tmp/dump --threads 256
```

Read, in order:
- **MFMA count** equal to the reference? Then the gap is data movement / overhead.
- **Load/store widths**: `buffer_load_dword` (4 B) vs `dwordx4` (16 B); `ds_read_b64`
  vs `ds_read_b128`; LDS DMA loads (`... lds`) and their `s_mov_b32 m0`.
- **Registers and occupancy** (CDNA3): 512 VGPR+AGPR per SIMD lane, granule 8,
  waves/SIMD <= 8; 64 KB LDS per CU; a 256-thread block puts one wave on each SIMD.
  `isa_stats.py` prints which of registers or LDS limits occupancy. Lowering the
  non-limiting resource does not help.
- **Spills** (`scratch_*`, `vgpr_spill_count`) must be 0.
- Use the ISA to confirm an intended change happened (e.g. loads became `dwordx4`).

### 4. Hardware counters

```bash
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_INSTS_VMEM SQ_WAVE_CYCLES \
    --output-format csv -d /tmp/pmc -o run -- python bench.py my_kernel
python scripts/rp_counters.py /tmp/pmc --match my_kernel
```

Compare the same shape across kernels and read ratios (e.g. conflicts 15x, VMEM 2.5x).

### 5. Hypotheses and ablations

- List candidates with their evidence, ranked by expected gain.
- Implement each as a separate variant (a small generator script that patches the
  current file keeps variants in sync); test, time, and inspect ISA for each.
- Accept the measurement over the ranking: in the case study LDS padding removed 15x
  bank conflicts yet gained 10%, while wide Q loads + fewer registers gained 2.4x.
- When a gain shows up somewhere unexpected, find out why before moving on (a
  removed serial LDS scan sped up every block, not only skipped ones).

### 6. Isolate fixed costs

Construct inputs that turn one cost on or off:
- everything skipped / nothing visible -> per-block fixed overhead;
- an empty kernel with the same grid and block -> launch floor
  (`scripts/empty_grid_floor.py`, GPU time from rocprofv3);
- all visible vs causal ramp -> compute vs skipping;
- one vs many requests per block -> per-pass reloads.

### 7. Ceiling before effort

Before an optimization, bound its gain: e.g. skipped-block overhead 74 us minus a
16 us floor, times the skipped fraction, is <= 6% of the causal-prefill time, so it
was not worth a redesign. Stop when the next ceiling is small.

### 8. Integration

- Route to FlyDSL only where JIT launches are allowed (not under CUDA graph capture).
- Check the real caller's dtypes/strides (int64 positions, strided page tables);
  support or convert them, never let a `supports()` gate silently reject production
  inputs.
- Verify routing with counting wrappers and correctness through the integrated API.
- Run the project's formatters/linters (e.g. `black`, `ruff check --no-cache`).

## Writing FlyDSL for speed (patterns that paid off)

- Hoist loads of data that does not change across a runtime loop out of it.
- Use the same K permutation for both MFMA operands so each lane issues 16-byte
  loads (see pitfalls reference).
- Keep one accumulator live: loop order tile -> head -> k-step instead of holding an
  accumulator per (head, tile).
- Compute block-wide metadata (min/max/range) with a wave `shuffle_xor` reduction in
  one wave and broadcast a few values through LDS, instead of every thread scanning a
  table.
- Skip whole blocks whose output the consumer never reads, under a block-uniform
  `if` that also covers the loads.

## Utility scripts (execute)

- `scripts/rp_kernel_times.py` - median GPU time per kernel from a kernel-trace CSV,
  optionally split into per-shape groups.
- `scripts/rp_counters.py` - per-kernel counter values from a `--pmc` CSV.
- `scripts/isa_stats.py` - instruction mix, registers, LDS and occupancy estimate from
  `FLYDSL_DUMP_IR` output.
- `scripts/empty_grid_floor.py` - empty kernels on 4096 / 16384 blocks for the launch
  floor; run under rocprofv3 and read with `rp_kernel_times.py --match empty`.

## Additional resources

- FlyDSL semantics and API traps (tracing, carried variables, layouts, MFMA fragments,
  buffer/LDS access, DMA): [references/flydsl-pitfalls.md](references/flydsl-pitfalls.md)
- A complete optimization log with numbers and decisions:
  [references/case-study-qsa-logits.md](references/case-study-qsa-logits.md)
