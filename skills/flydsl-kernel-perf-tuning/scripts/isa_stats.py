#!/usr/bin/env python3
"""Instruction mix, register/LDS usage and occupancy estimate of a FlyDSL kernel's final ISA.

Dump the ISA first (inside the FlyDSL container):
    FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/dump python my_test.py --case 0
Then:
    python isa_stats.py /tmp/dump                 # every kernel found under the dump dir
    python isa_stats.py /tmp/dump/<kernel>/22_final_isa.s --threads 256

Occupancy is a CDNA3 (gfx942, MI300/MI308) estimate: 512 VGPR+AGPR per SIMD lane in
granules of 8, at most 8 waves per SIMD, 64 KB LDS per CU, a block's waves spread over
the CU's 4 SIMDs. Treat it as a guide; the hardware has extra constraints.
"""

import argparse
import collections
import glob
import math
import os
import re

GROUPS = [
    ("mfma", r"v_mfma\w*"),
    ("global/buffer load", r"(?:buffer|global)_load\w*"),
    ("global/buffer store", r"(?:buffer|global)_store\w*"),
    ("buffer atomic", r"(?:buffer|global)_atomic\w*"),
    ("lds read", r"ds_read\w*"),
    ("lds write", r"ds_write\w*"),
    ("barrier", r"s_barrier"),
    ("waitcnt", r"s_waitcnt"),
    ("m0 writes", r"s_mov_b32 m0"),
    ("select", r"v_cndmask_b32"),
    ("scratch (spill)", r"scratch_\w+"),
]


def meta(text: str, key: str):
    m = re.search(rf"\.{key}:\s*(\d+)", text) or re.search(rf"\.amdhsa_{key}\s+(\d+)", text)
    return int(m.group(1)) if m else None


def occupancy(regs_total: int, lds: int, threads: int) -> str:
    regs = int(math.ceil(regs_total / 8) * 8) or 8
    by_regs = min(8, 512 // regs)
    waves_per_block = max(1, threads // 64)
    blocks_by_lds = 65536 // lds if lds else 10**9
    by_lds = min(8, (blocks_by_lds * waves_per_block) // 4)
    limit = "registers" if by_regs <= by_lds else "LDS"
    return f"~{min(by_regs, by_lds)} waves/SIMD (registers allow {by_regs}, LDS allows {by_lds}; limited by {limit})"


def report(path: str, threads: int) -> None:
    text = open(path).read()
    body = [ln.strip() for ln in text.splitlines() if ln.startswith("\t") and not ln.strip().startswith((".", ";"))]
    counts = collections.Counter()
    detail = collections.Counter()
    for ln in body:
        op = ln.split()[0] if ln.split() else ""
        for label, pat in GROUPS:
            if re.fullmatch(pat, op) or (label == "m0 writes" and ln.startswith("s_mov_b32 m0")):
                counts[label] += 1
                if label in ("global/buffer load", "global/buffer store", "lds read", "lds write", "mfma"):
                    detail[op + (" (lds dma)" if " lds" in ln and "load" in op else "")] += 1
                break
    # .vgpr_count already covers arch VGPRs (up to accum_offset) plus AGPRs
    vgpr_total = meta(text, "vgpr_count") or 0
    agpr = meta(text, "agpr_count") or 0
    accum_offset = meta(text, "accum_offset") or vgpr_total
    regs_total = max(vgpr_total, accum_offset + agpr)
    sgpr = meta(text, "sgpr_count")
    lds = meta(text, "group_segment_fixed_size") or 0
    spill = meta(text, "vgpr_spill_count")
    print(f"== {path}")
    print(f"   vgpr+agpr={regs_total} (agpr={agpr}) sgpr={sgpr} lds={lds}B spill={spill}")
    print(f"   {occupancy(regs_total, lds, threads)}")
    for label, _ in GROUPS:
        if counts[label]:
            print(f"   {label:20s} {counts[label]:5d}")
    for op, n in sorted(detail.items(), key=lambda kv: -kv[1]):
        print(f"      {op:32s} {n:5d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="FLYDSL_DUMP_DIR, a kernel subdir, or a 22_final_isa.s file")
    ap.add_argument("--threads", type=int, default=256, help="threads per block (for the occupancy estimate)")
    a = ap.parse_args()
    files = [a.path] if os.path.isfile(a.path) else sorted(glob.glob(os.path.join(a.path, "**", "*final_isa.s"), recursive=True))
    if not files:
        raise SystemExit(f"no *final_isa.s under {a.path}")
    for f in files:
        report(f, a.threads)


if __name__ == "__main__":
    main()
