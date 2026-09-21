#!/usr/bin/env python3
"""Make ROCTx marker regions readable in a ROCpd timeline viewer.

rocprofv3 names every ROCTx range after the API that produced it, so a viewer
shows a stack of identical `roctxThreadRangeA` rows. The text passed to
`roctxRangePush` is kept separately, in `rocpd_event.extdata` as a JSON
`message` field, and never reaches the region name.

This rewrites each marker region's name to its own label, which turns that
anonymous stack into a readable call stack. SGLang's layerwise hook pushes a
Python dict repr, so the module path is lifted out of it when present.
"""

import json
import re
import shutil
import sqlite3
import sys

# The dict repr arrives SQL-escaped, so single quotes may be doubled.
MODULE_PATTERN = re.compile(r"'{1,2}Module'{1,2}\s*:\s*'{1,2}([^']+)'{1,2}")
MAX_RAW_LABEL = 120


def marker_uuids(con):
    found = []
    for (name,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_region_%'"
    ):
        found.append(name[len("rocpd_region_") :])
    return sorted(found)


def label_for(message):
    matched = MODULE_PATTERN.search(message)
    if matched:
        return matched.group(1)
    collapsed = " ".join(message.split())
    return collapsed[:MAX_RAW_LABEL] if collapsed else None


def relabel(con, suffix):
    strings = f"rocpd_string_{suffix}"
    regions = f"rocpd_region_{suffix}"
    events = f"rocpd_event_{suffix}"

    guid = con.execute(f'SELECT guid FROM "{regions}" LIMIT 1').fetchone()[0]
    rows = con.execute(
        f"""
        SELECT r.id, e.extdata FROM "{regions}" r
        JOIN "{events}" e ON e.id = r.event_id
        WHERE (SELECT string FROM "{strings}" WHERE id = e.category_id) LIKE 'MARKER%'
        """
    ).fetchall()

    cache = {}
    updates = []
    skipped = 0
    for region_id, extdata in rows:
        try:
            message = json.loads(extdata or "{}").get("message", "")
        except json.JSONDecodeError:
            message = ""
        label = label_for(message)
        if not label:
            skipped += 1
            continue
        if label not in cache:
            con.execute(
                f'INSERT OR IGNORE INTO "{strings}" (guid, string) VALUES (?, ?)',
                (guid, label),
            )
            cache[label] = con.execute(
                f'SELECT id FROM "{strings}" WHERE string = ?', (label,)
            ).fetchone()[0]
        updates.append((cache[label], region_id))

    con.executemany(f'UPDATE "{regions}" SET name_id = ? WHERE id = ?', updates)
    return len(rows), len(updates), len(cache), skipped


def main(source, output):
    if source != output:
        shutil.copyfile(source, output)
    con = sqlite3.connect(output)
    for suffix in marker_uuids(con):
        total, done, distinct, skipped = relabel(con, suffix)
        print(
            f"{suffix[:8]}: {done}/{total} marker regions relabeled, "
            f"{distinct} distinct labels, {skipped} without text"
        )
    con.commit()
    print("integrity", con.execute("PRAGMA integrity_check").fetchone()[0])
    con.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
