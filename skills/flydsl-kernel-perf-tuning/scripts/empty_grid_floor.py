"""Launch floor: GPU time of an empty FlyDSL kernel with a given grid of 256-thread blocks.

The printed CUDA-event numbers are dominated by host launch overhead; read the GPU time with
    rocprofv3 --kernel-trace --output-format csv -d /tmp/rp_empty -o run -- python empty_grid_floor.py
    python rp_kernel_times.py /tmp/rp_empty --match empty
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

_Big = fx.struct(type("_Big", (), {"__annotations__": {"buf": fx.Array[fx.Int32, 18432 // 4, 16]}}))


@flyc.kernel(known_block_size=(256, 1, 1))
def empty_kernel(out: fx.Tensor, n: fx.Int32):
    tid = fx.thread_idx.x
    if tid == 1000:  # never true: keeps the argument live
        fx.ptr_store(fx.Int32(n), fx.get_iter(out))


@flyc.kernel(known_block_size=(256, 1, 1))
def empty_lds_kernel(out: fx.Tensor, n: fx.Int32):
    tid = fx.thread_idx.x
    lds = fx.SharedAllocator().allocate(_Big).peek()
    buf = lds.buf.view(fx.make_layout(18432 // 4, 1))
    if tid == 1000:
        buf[0] = n
        fx.ptr_store(fx.Int32(buf[1]), fx.get_iter(out))


@flyc.jit
def run_empty(out: fx.Tensor, n: fx.Int32, grid: fx.Int32, stream: fx.Stream):
    empty_kernel(out, n).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)


@flyc.jit
def run_empty_lds(out: fx.Tensor, n: fx.Int32, grid: fx.Int32, stream: fx.Stream):
    empty_lds_kernel(out, n).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)


def time_ms(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


if __name__ == "__main__":
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    st = torch.cuda.current_stream()
    for grid in (4096, 16384):
        a = time_ms(lambda: run_empty(out, 1, grid, st))
        b = time_ms(lambda: run_empty_lds(out, 1, grid, st))
        print(f"grid={grid:6d} x 256 threads: empty {a * 1e3:6.1f} us   empty + 18KB LDS {b * 1e3:6.1f} us")
