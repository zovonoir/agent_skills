# Case study: QSA paged MQA logits scorer on MI308X

Kernel: per query row, 4 index heads x 128 dims (bf16) scored against paged compressed
keys (1 head x 128), ReLU per head, sum over heads, scale, `-inf` outside the causal
horizon. Block = 256 threads = 4 waves, 64 rows x 64 columns (4 pages of 16), bf16 MFMA
16x16x16. Numbers are pure GPU time (rocprofv3, median) in ms.

| Step | 1040 x 32256 all visible | 8192 x 4096 all visible | 16384 x 4096 causal | 8192 x 32256 causal | What changed |
|---|---|---|---|---|---|
| Triton CUDA-core original | 6.44 | 6.53 | 13.10 | 51.31 | reference point |
| layout-algebra baseline | 1.33 | 1.28 | 2.55 | 10.19 | first correct version |
| + early exit | 1.27 | 1.22 | 1.70 | 5.75 | skip column blocks past every row's visible count |
| + Q wide loads once, 1 accumulator | 0.54 | 0.51 | 0.73 | 2.57 | K-permuted views, Q hoisted out of the request loop, page->head->k loop order |
| + K register path + LDS padding | 0.53 | 0.51 | 0.71 | 2.52 | BufferCopy128b + ds_write_b128, row stride 136 |
| + wave-0 shuffle reduction, Q load under the skip `if` | 0.40 | 0.37 | 0.51 | 1.77 | block metadata without a 64-step serial LDS scan |

Other kernels on the same shapes: previous hand-written FlyDSL scorer 0.47 / 0.49 /
0.98 / 3.83; Triton MFMA scorer (16 rows x 128 cols per program, early exit) 0.51 /
0.55 / 0.62 / 2.34.

## How each decision was made

1. **Baseline vs reference kernel.** Same MFMA count (128 per wave per pass); ISA showed
   84 `buffer_load_dword` vs 20 `buffer_load_dwordx4`, 236 vs 121 VGPRs. Counters:
   LDS bank conflicts 15x, VMEM instructions 2.5x.
2. **Padding alone (hypothesis: bank conflicts)** cut conflicts 15x but time only 10%.
   Conflicts were not the bottleneck; the ranking was revised from measurements.
3. **Q loads + registers (hypothesis: VMEM count and occupancy).** K-permuted views
   turned Q into 16 `dwordx4` loads, 236 -> 144 VGPRs; time fell to 42%.
4. **K path.** Register path vs 4-byte DMA and padding: within 2%. At 128 VGPRs the
   kernel was LDS-limited (17-18 KB/block -> 3 blocks/CU), so fewer registers did not
   raise occupancy.
5. **Early exit.** Top-k reads each row only over `[0, visible)`, so blocks with
   `col0 >= max(visible)` write nothing. The empty-loop trick (`req_high = req_low - 1`)
   skipped the work but not the Q load placed before the loop; wrapping Q load + loop
   in a block-uniform `if` fixed that.
6. **Metadata reduction.** The expected gain was on skipped blocks; the measured gain
   came from every block: removing the 64-step serial scan of LDS tags sped up
   all-visible shapes by 27%.
7. **Stopping.** All-blocks-skipped cost 74 us vs a 16 us empty-kernel floor, i.e. at
   most ~6% on causal prefill. A pos-only bound computed per wave made it worse
   (74 -> 104 us) and was reverted.

## Mistakes worth avoiding

- CUDA-event timing of a short kernel reported 0.186 ms where the GPU time was 74 us.
- Two FlyDSL kernels named `qsa_logits_fly_kernel_0` merged in the trace.
- `flat_divide` with a 1-mode tiler on a 2-D tensor produced a wrong shape that still
  gave correct results, hiding the bug until the layout was printed.
- A `supports()` gate requiring int32 index tensors would never fire in serving
  (positions are int64); the wrapper converts instead.
