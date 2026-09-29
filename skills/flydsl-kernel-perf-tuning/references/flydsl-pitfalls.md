# FlyDSL pitfalls (flydsl 0.3.x, gfx942 / MI308X)

Each item was hit and verified while writing a kernel; the fix is the part to copy.

## Tracing and control flow

- **Python `if` / `for` on runtime values become `scf.if` / `scf.for`.** A plain
  Python constant condition (`if MODE == "x":`) is also rewritten; use
  `if fx.const_expr(...)` or pick the branch outside the kernel.
- **Loop / branch carried variables.** For a runtime `for`/`if`, FlyDSL carries every
  name that is (a) assigned in the body and already defined before it, or (b) used as
  `name.method(...)` in the body while defined before it. Objects such as `ThrMma`,
  `ThrCopy`, `TiledCopy`, tensor views cannot be carried and fail with errors like
  `ThrMma.__init__() missing 1 required positional argument: 'thr_idx'`.
  Fix: do `make_fragment_*`, `retile`, `partition_*`, `get_slice` before the loop / if;
  inside, only index (`frag[i]`, `view[None, None, p]`), call `fx.*` functions, and call
  methods on names first assigned inside the body. Calls on subscripts
  (`acc[0].fill(...)`) are safe. Give loop variables inside the body names not used
  before it (a top-level `for h in range_constexpr` makes `h` an outer name).
- **Kernel arguments reassigned inside an `if`** (`q = fx.make_view(q, ...)`) are
  carried too; build views before the branch.
- **`and` / `or` / `not` on runtime booleans raise** `cannot evaluate dynamic 'Boolean'
  as Python bool during tracing`. Use `&`, `|`, and parenthesize comparisons:
  `(a >= 0) & (a < n)`; `a >= 0 & a < n` parses as a chained comparison.
- **`x.select(a, b)` evaluates both `a` and `b`.** A load in either operand always
  executes: clamp the index first (`view[ok.select(i, 0)]`), then select the value.
  `Vector.select` is lane-wise; a scalar condition with vector operands picks the
  whole vector.
- **`print(...)` in a kernel runs once at trace time** (shows `?` for dynamic sizes);
  `fx.printf(...)` runs on the GPU per thread and prints runtime layouts. Guard it with
  `if tid == 0 and bid == 0` and remove it before benchmarking.
- Barriers inside a runtime `if` are fine only if the condition is block-uniform.

## Layout algebra

- **The tiler must have as many modes as the tensor.** `flat_divide(x[100,128], (64,))`
  treats `x` as 1-D and yields `(64,2,100)` (the 128 columns are lost; later code only
  works because addresses still follow strides). Write `flat_divide(x, (64,128))`
  -> `(64,128,2,1)` and index `[None, None, blk, 0]`. `fx.printf` the layout.
- `flat_divide` of a runtime-shaped tensor keeps size-1 modes, a static one may drop
  them, so mode counts can differ; always print before indexing.
- `make_view(get_iter(t), make_layout(N, 1))` flattens a padded LDS tile incorrectly;
  keep the 2-D layout with its stride instead.
- A tiled MMA over one wave (`make_layout((1,1,1), ...)`) needs `thr_slice(lane_id)`
  (0..63); `tid` >= 64 is treated as tiling along K and shifts offsets by 16/32/48
  elements for waves 1-3. Same for `make_tiled_copy_A/B/C(...).get_slice(lane_id)`.
- `make_tiled_copy_A/B/C(atom, tiled_mma)` takes the TiledMma, not the ThrMma.
  `make_fragment_B` is a `ThrMma` method; `fx.make_fragment_B` does not exist.
- MFMA 16x16x16 bf16 C fragment: element `j` of lane `l` is row `4*(l//16) + j`,
  column `l % 16`.
- **K permutation for wide loads.** Standard MFMA K layout makes each lane read 8
  bytes per k-step. Using the same permuted K order for A and B keeps the dot product
  and lets a lane read 32 contiguous bf16: view a `[16,128]` operand as
  `(16,(4,4),8):(row_stride,(1,32),4)`, i.e. k-step `s`, lane group `g` covers
  dims `32g + 4s .. +3`.

## Memory access

- `fx.recast_iter(fx.Int8, fx.get_iter(t))` + `fx.rocdl.make_buffer_ptr(...,
  num_records_bytes=...)` gives a byte-addressed buffer resource: loads past the size
  read 0, stores past it are dropped. Byte offset `0x7FFFFFFF` is the branch-free way
  to mask a store.
- `fx.add_offset(ptr, n)` / `ptr + n` count **elements** of the pointer type; recast to
  `Int8` for byte offsets. Recasting back needs the long form with alignment,
  `fx.recast_iter(fx.PointerType.get(ty.ir_type, fx.AddressSpace.Shared, 16), p)`.
- `fx.ptrtoint(fx.get_iter(t))` equals `t.data_ptr()` on the host.
- `BufferCopy(bits, cache_modifier=0)`: the argument is **bits**; the second argument is
  a cache policy, not an element count.
- `memref_store_vec` needs a `Vector`; wrap scalars with
  `fx.Vector.from_elements([x], dtype=fx.Int32)`.
- Global -> LDS DMA (`BufferCopyLDS*`): 16-byte DMA exists only on gfx950; gfx942 has
  4-byte (`BufferCopyLDS32b`). One DMA instruction writes 64 lanes x 4 B = 256
  contiguous LDS bytes per wave, so with a padded LDS row map one wave instruction to
  one 256-byte row. Correct per-thread mapping for `[16,128]` bf16: element pair
  `(2t, 2t+1)`, layout `(2,256):(1,2)`, not `(2,256):(256,1)`.
- Buffer descriptors describe global memory only; LDS uses `ptr_load` / `ptr_store`
  on a shared pointer (`lds.buf.ptr`) or layout views of the struct array.
- LDS struct fields must be declared with the right dtype (`fx.Array[fx.Int32, N, 16]`
  for ids; bf16 silently rounds ids above 256).
- `@fx.struct` with `from __future__ import annotations` fails (string annotations);
  build the class with `type(...)` and real annotation objects.

## Synchronization

- `ds_read` -> MFMA inside one wave needs no barrier; the compiler inserts
  `s_waitcnt lgkmcnt`. Barriers are needed when a wave reads LDS written by another
  wave, and before overwriting LDS other waves may still read.
- A branch that is exactly one full wave (`if tid < 64:`) can use `shuffle_xor(off, 64)`
  wave reductions; all 64 lanes are active.

## Integration

- FlyDSL JIT launches are not captured into CUDA graphs: route to FlyDSL only in eager
  paths (e.g. prefill) and check `torch.cuda.is_current_stream_capturing()`.
- Serving inputs may differ from unit tests (int64 positions, page tables sliced out of
  a wider buffer). Either support them in the kernel or convert in the wrapper; a
  `supports()` gate that rejects them silently disables the kernel in production.
