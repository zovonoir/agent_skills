#!/usr/bin/env python3
"""Export a ROCpd unified database on one shared time origin.

Per-rank normalization subtracts each process's own first event, which injects
the gap between process origins into every kernel on that rank's timeline and
makes synchronizing collectives look staggered. Everything here is normalized
against the earliest event across all processes instead.
"""

import bisect
import csv
import sqlite3
import sys

COMM_PATTERNS = [
    ("ncclDevKernel", "%ncclDevKernel%"),
    ("ar_ll128", "%ar_ll128%"),
    ("cross_device_reduce", "%cross_device_reduce%"),
    ("allgather_vec", "%allgather_vec%"),
]


def origins(con):
    per_rank = {}
    for (pid,) in con.execute("SELECT DISTINCT pid FROM kernels ORDER BY pid"):
        row = con.execute(
            """
            SELECT MIN(t) FROM (
                SELECT MIN(start) AS t FROM regions WHERE pid = :pid
                UNION ALL
                SELECT MIN(start) AS t FROM kernels WHERE pid = :pid
            )
            """,
            {"pid": pid},
        ).fetchone()
        per_rank[pid] = row[0]
    return per_rank, min(per_rank.values())


def export_kernels(con, per_rank, global_origin, path):
    ranks = {pid: index for index, pid in enumerate(sorted(per_rank))}
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "pid",
                "gpu_index",
                "dispatch_id",
                "name",
                "start_ms",
                "end_ms",
                "duration_us",
                "start_ms_per_rank_origin",
                "grid_x",
                "workgroup_x",
            ]
        )
        rows = con.execute(
            """
            SELECT pid, agent_log_index, dispatch_id, name, start, end, duration,
                   grid_x, workgroup_x
            FROM kernels ORDER BY start
            """
        )
        count = 0
        for pid, gpu, dispatch, name, start, end, dur, gx, wx in rows:
            writer.writerow(
                [
                    ranks[pid],
                    pid,
                    gpu,
                    dispatch,
                    name,
                    f"{(start - global_origin) / 1e6:.6f}",
                    f"{(end - global_origin) / 1e6:.6f}",
                    f"{dur / 1e3:.3f}",
                    f"{(start - per_rank[pid]) / 1e6:.6f}",
                    gx,
                    wx,
                ]
            )
            count += 1
    return count


def export_comm_pairs(con, per_rank, global_origin, path):
    pids = sorted(per_rank)
    left, right = pids[0], pids[1]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "pair_index",
                "rank0_start_ms",
                "rank1_start_ms",
                "delta_us",
                "rank0_dur_us",
                "rank1_dur_us",
                "intervals_overlap",
                "delta_us_if_per_rank_origin",
            ]
        )
        for family, pattern in COMM_PATTERNS:
            query = (
                "SELECT start, end, duration FROM kernels "
                "WHERE pid = ? AND name LIKE ? ORDER BY start"
            )
            a = con.execute(query, (left, pattern)).fetchall()
            b = con.execute(query, (right, pattern)).fetchall()
            starts_b = [item[0] for item in b]
            for index, (s0, e0, d0) in enumerate(a):
                near = bisect.bisect_left(starts_b, s0)
                best = None
                for j in range(max(0, near - 3), min(len(b), near + 4)):
                    gap = abs(b[j][0] - s0)
                    if best is None or gap < best[0]:
                        best = (gap, j)
                if best is None:
                    continue
                s1, e1, d1 = b[best[1]]
                writer.writerow(
                    [
                        family,
                        index,
                        f"{(s0 - global_origin) / 1e6:.6f}",
                        f"{(s1 - global_origin) / 1e6:.6f}",
                        f"{(s0 - s1) / 1e3:.3f}",
                        f"{d0 / 1e3:.3f}",
                        f"{d1 / 1e3:.3f}",
                        "yes" if (s0 < e1 and s1 < e0) else "no",
                        f"{((s0 - per_rank[left]) - (s1 - per_rank[right])) / 1e3:.3f}",
                    ]
                )


def main(db, kernel_csv, comm_csv):
    con = sqlite3.connect(db)
    per_rank, global_origin = origins(con)
    for pid, value in sorted(per_rank.items()):
        print(
            f"pid={pid} origin={value} offset_from_global={(value - global_origin) / 1e3:.3f} us"
        )
    count = export_kernels(con, per_rank, global_origin, kernel_csv)
    export_comm_pairs(con, per_rank, global_origin, comm_csv)
    print(f"wrote {count} kernel rows to {kernel_csv}")
    print(f"wrote comm pair table to {comm_csv}")
    con.close()


if __name__ == "__main__":
    main(*sys.argv[1:4])
