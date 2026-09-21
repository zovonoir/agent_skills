#!/usr/bin/env python3
"""Merge per-rank ROCpd databases into one multi-GPU database.

rocpd 1.1.0 has no `merge` subcommand. Concrete tables carry a GUID suffix, so
they can coexist; only the unsuffixed base views need to be rebuilt as
UNION ALL over every rank's concrete table.

Timestamps are already on one system clock domain and need no correction. The
merge does add a zero-work anchor region to every rank that starts late, so
that viewers which zero each process against its own first event still place
all ranks on one origin. Without it a synchronizing collective looks staggered
by the gap between process start times.
"""

import re
import shutil
import sqlite3
import sys

UUID_SUFFIX = re.compile(r"^(rocpd_.+)_([0-9a-f]{8}(?:_[0-9a-f]{4}){3}_[0-9a-f]{12})$")


def concrete_tables(con, schema="main"):
    rows = con.execute(
        f"SELECT name FROM {schema}.sqlite_master WHERE type='table'"
    ).fetchall()
    out = {}
    for (name,) in rows:
        m = UUID_SUFFIX.match(name)
        if m:
            out.setdefault(m.group(1), []).append(name)
    return out


ANCHOR_NAME = "ROCPD_MERGE_TIME_ANCHOR"
ANCHOR_DURATION_NS = 1000


def uuid_suffixes(con):
    found = []
    for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_region_%'"
    ):
        match = UUID_SUFFIX.match(name)
        if match:
            found.append(match.group(2))
    return sorted(found)


def anchor_time_origin(con):
    """Give every rank the same first-event timestamp.

    A viewer that subtracts each process's own minimum cannot otherwise know
    that the ranks share a clock. Shifting timestamps does not help, because
    that subtraction moves the origin by the same amount.
    """
    suffixes = uuid_suffixes(con)
    origins = {}
    for suffix in suffixes:
        low = con.execute(
            f'SELECT MIN(start) FROM "rocpd_region_{suffix}"'
        ).fetchone()[0]
        kernel_low = con.execute(
            f'SELECT MIN(start) FROM "rocpd_kernel_dispatch_{suffix}"'
        ).fetchone()[0]
        origins[suffix] = min(v for v in (low, kernel_low) if v is not None)

    global_origin = min(origins.values())
    added = []
    for suffix, origin in origins.items():
        if origin <= global_origin:
            continue
        template = con.execute(
            f'SELECT nid, pid, tid FROM "rocpd_region_{suffix}" ORDER BY start LIMIT 1'
        ).fetchone()
        guid = con.execute(
            f'SELECT guid FROM "rocpd_region_{suffix}" LIMIT 1'
        ).fetchone()[0]

        con.execute(
            f'INSERT INTO "rocpd_string_{suffix}" (guid, string) VALUES (?, ?)',
            (guid, ANCHOR_NAME),
        )
        name_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        con.execute(
            f'INSERT INTO "rocpd_event_{suffix}" (guid, category_id) VALUES (?, ?)',
            (guid, name_id),
        )
        event_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        con.execute(
            f'INSERT INTO "rocpd_region_{suffix}" '
            "(guid, nid, pid, tid, start, end, name_id, event_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                guid,
                template[0],
                template[1],
                template[2],
                global_origin,
                global_origin + ANCHOR_DURATION_NS,
                name_id,
                event_id,
            ),
        )
        added.append((suffix, (origin - global_origin) / 1e3))
    con.commit()
    return global_origin, added


def main(inputs, output):
    shutil.copyfile(inputs[0], output)
    con = sqlite3.connect(output)
    con.execute("PRAGMA foreign_keys=OFF")

    for idx, src in enumerate(inputs[1:], start=1):
        alias = f"src{idx}"
        con.execute(f"ATTACH DATABASE ? AS {alias}", (src,))
        for _, tables in sorted(concrete_tables(con, alias).items()):
            for table in tables:
                ddl = con.execute(
                    f"SELECT sql FROM {alias}.sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()[0]
                con.execute(ddl)
                con.execute(f'INSERT INTO main."{table}" SELECT * FROM {alias}."{table}"')
        con.commit()
        con.execute(f"DETACH DATABASE {alias}")

    for base, tables in sorted(concrete_tables(con).items()):
        if not con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='view' AND name=?", (base,)
        ).fetchone():
            continue
        union = "\nUNION ALL\n".join(f'SELECT * FROM "{t}"' for t in sorted(tables))
        con.execute(f'DROP VIEW "{base}"')
        con.execute(f'CREATE VIEW "{base}" AS\n{union}')
    con.commit()

    global_origin, added = anchor_time_origin(con)
    print(f"global origin {global_origin}")
    for suffix, offset_us in added:
        print(f"anchored {suffix} (was {offset_us:.3f} us late)")
    if not added:
        print("no anchor needed; all ranks already share an origin")

    con.execute("VACUUM")
    print("integrity", con.execute("PRAGMA integrity_check").fetchone()[0])
    con.close()


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
