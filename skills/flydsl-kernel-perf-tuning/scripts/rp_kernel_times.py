#!/usr/bin/env python3
"""Median pure-GPU kernel time per kernel name from a rocprofv3 --kernel-trace CSV.

Usage:
    rocprofv3 --kernel-trace --output-format csv -d /tmp/rp -o run -- python bench.py
    python rp_kernel_times.py /tmp/rp                      # every kernel, median over all launches
    python rp_kernel_times.py /tmp/rp --match qsa_ --split 5
        # launches of each matching kernel are split, in launch order, into 5 equal
        # groups (one per benchmarked shape) and the median of each group is printed

A kernel name that also matches a longer pattern is still reported under its own full
name, so give kernels distinct names (two FlyDSL kernels both called foo_kernel_0 merge).
Register counts are printed as rocprofv3 reports them; use isa_stats.py for occupancy.
"""

import argparse
import collections
import csv
import glob
import os
import sys


def find_csv(path: str) -> str:
    if os.path.isfile(path):
        return path
    hits = glob.glob(os.path.join(path, "**", "*kernel_trace.csv"), recursive=True)
    if not hits:
        sys.exit(f"no *kernel_trace.csv under {path}")
    return hits[0]


def median(v):
    v = sorted(v)
    return v[len(v) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="rocprofv3 output dir or kernel_trace.csv")
    ap.add_argument("--match", default="", help="substring a kernel name must contain")
    ap.add_argument("--split", type=int, default=1, help="split each kernel's launches into N ordered groups")
    a = ap.parse_args()

    times = collections.OrderedDict()
    resources = {}
    for r in csv.DictReader(open(find_csv(a.path))):
        name = r["Kernel_Name"]
        if a.match not in name:
            continue
        times.setdefault(name, []).append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
        resources[name] = (
            r.get("VGPR_Count", r.get("Arch_VGPR_Count", "?")),
            r.get("Accum_VGPR_Count", "?"),
            r.get("LDS_Block_Size", r.get("Group_Segment_Size", "?")),
        )

    if not times:
        sys.exit("no matching kernels")
    for name, v in times.items():
        vgpr, agpr, lds = resources[name]
        head = f"{name[:60]:60s} n={len(v):5d} vgpr={vgpr} agpr={agpr} lds={lds}"
        if a.split <= 1:
            print(f"{head}  median {median(v):9.1f} us")
            continue
        per = len(v) // a.split
        groups = [median(v[i * per:(i + 1) * per]) for i in range(a.split)] if per else []
        print(f"{head}  median us per group: " + ", ".join(f"{g:.1f}" for g in groups))


if __name__ == "__main__":
    main()
