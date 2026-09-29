#!/usr/bin/env python3
"""Per-kernel hardware counter values from a rocprofv3 --pmc CSV.

Usage:
    rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS SQ_INSTS_VMEM SQ_WAVE_CYCLES \
        --output-format csv -d /tmp/pmc -o run -- python bench.py
    python rp_counters.py /tmp/pmc --match qsa_            # first launch of each kernel
    python rp_counters.py /tmp/pmc --match qsa_ --launch -1   # last launch

Compare ratios between two kernels on the same shape; absolute values are hard to read.
Counters that do not fit in one pass need separate rocprofv3 runs.
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
    hits = glob.glob(os.path.join(path, "**", "*counter_collection.csv"), recursive=True)
    if not hits:
        sys.exit(f"no *counter_collection.csv under {path}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="rocprofv3 output dir or counter_collection.csv")
    ap.add_argument("--match", default="", help="substring a kernel name must contain")
    ap.add_argument("--launch", type=int, default=0, help="which launch of each kernel to report (python index)")
    a = ap.parse_args()

    # (kernel, counter) -> per-dispatch values; a dispatch may report one row per XCD/SE, so sum by dispatch id
    per_dispatch = collections.defaultdict(lambda: collections.defaultdict(float))
    order = collections.defaultdict(list)
    for r in csv.DictReader(open(find_csv(a.path))):
        name = r["Kernel_Name"]
        if a.match not in name:
            continue
        key = (name, r["Counter_Name"])
        did = r.get("Dispatch_Id", r.get("Correlation_Id", "0"))
        if did not in per_dispatch[key]:
            order[key].append(did)
        per_dispatch[key][did] += float(r["Counter_Value"])

    if not per_dispatch:
        sys.exit("no matching kernels")
    for (name, counter), vals in sorted(per_dispatch.items()):
        ids = order[(name, counter)]
        print(f"{name[:60]:60s} {counter:24s} {vals[ids[a.launch]]:.3e}  (dispatches={len(ids)})")


if __name__ == "__main__":
    main()
