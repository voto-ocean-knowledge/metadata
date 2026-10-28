#!/usr/bin/env python3
"""
Extract global attributes from NetCDF files into a flat metadata table.

Each row in the output table holds the global attributes of one file, plus a
few bookkeeping columns (prefixed with "_"). On subsequent runs only files
that are new or have changed (size or modification time) are re-read; rows of
deleted files are removed. New global attributes simply become new columns.

Usage:
    python extract_nc_metadata.py /data/netcdf_root -o metadata.csv
    python extract_nc_metadata.py /data/netcdf_root -o metadata.parquet --workers 8
    python extract_nc_metadata.py /data/netcdf_root -o metadata.csv --force

Requirements: netCDF4, numpy, pandas (+ pyarrow for .parquet output)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
log = logging.getLogger("nc_metadata")

# Bookkeeping columns. Prefixed with "_" to avoid clashing with attribute names.
PATH_COL = "_file_path"          # path relative to the root folder
SIZE_COL = "_file_size"          # bytes
MTIME_COL = "_file_mtime_ns"     # modification time (ns since epoch)
EXTRACTED_COL = "_extracted_at"  # UTC timestamp of extraction
ERROR_COL = "_error"             # error message if the file could not be read
BOOKKEEPING_COLS = [PATH_COL, SIZE_COL, MTIME_COL, EXTRACTED_COL, ERROR_COL]

DEFAULT_PATTERNS = "*.nc"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def to_scalar(value):
    """Convert a NetCDF attribute value into something that fits in one table cell."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return to_scalar(value.item())
        return json.dumps(value.tolist())  # multi-valued attributes -> JSON string
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_global_attrs(path: Path) -> dict:
    """Read all global attributes from a NetCDF file."""
    attrs = {}
    with xr.open_dataset(path) as ds:
        for name, value in ds.attrs.items():
            key = f"attr{name}" if name in BOOKKEEPING_COLS else name
            attrs[key] = value
        dataset_id = ds.attrs['id']
        id_parts = ds.attrs['id'].split('_')
        attrs['data_mode'] = id_parts[-1]
        attrs['erddap_url'] = f"http://136.243.54.252:8082/erddap/tabledap/{dataset_id}.html"

        for name, variable in ds.variables.items():
            if name[:6] != 'SENSOR':
                continue
            sensor_type = '_'.join(name.split('_')[:-1])
            for key, val in variable.attrs.items():
                attrs[f"{sensor_type}-{key}"] = val


    return attrs


def process_file(job: tuple[str, str, int, int]) -> dict:
    """Build one table row for a file. Top-level function so it can be pickled."""
    abs_path, rel_path, size, mtime_ns = job
    row = {
        PATH_COL: abs_path,
        SIZE_COL: size,
        MTIME_COL: mtime_ns,
        EXTRACTED_COL: datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ERROR_COL: None,
    }
    try:
        row.update(extract_global_attrs(Path(abs_path)))
    except Exception as exc:  # keep going; record the failure in the table
        row[ERROR_COL] = f"{type(exc).__name__}: {exc}"
    return row


# --------------------------------------------------------------------------- #
# File discovery and table I/O
# --------------------------------------------------------------------------- #
def scan_files(roots: list[Path], pattern: str) -> dict[str, tuple[str, int, int]]:
    """Recursively find NetCDF files. Returns {relative_path: (abs_path, size, mtime_ns)}."""
    found = {}
    for root in roots:
        for p in root.rglob(pattern):
            if p.is_file():
                st = p.stat()
                found[p.relative_to(root).as_posix()] = (str(p), st.st_size, st.st_mtime_ns)
    return found


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=BOOKKEEPING_COLS)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    # keep_default_na=False so attribute values like "NA" or "None" stay as text
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Write atomically so an interrupted run never corrupts the existing table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        out = df.copy()
        # Parquet needs one type per column; stringify columns with mixed types
        for col in out.columns:
            if out[col].dtype == object and len({type(v) for v in out[col].dropna()}) > 1:
                out[col] = out[col].map(lambda v: v if pd.isna(v) else str(v))
        out.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, path)




# --------------------------------------------------------------------------- #
# Incremental update
# --------------------------------------------------------------------------- #
def update_metadata(root: list[Path], table_path: Path, patterns: str,
                    workers: int = 1, force: bool = False) -> pd.DataFrame:
    existing = load_table(table_path)
    on_disk = scan_files(root, patterns)
    log.info("Found %d NetCDF files under %s", len(on_disk), root)

    # Fingerprints from the previous run: {path: (size, mtime)}
    known, had_error = {}, set()
    if not existing.empty:
        known = {
            p: (int(s), int(m))
            for p, s, m in zip(existing[PATH_COL], existing[SIZE_COL], existing[MTIME_COL])
        }
        had_error = set(existing.loc[existing[ERROR_COL].notna(), PATH_COL])

    jobs, n_new, n_changed = [], 0, 0
    for rel, (abs_path, size, mtime) in sorted(on_disk.items()):
        if rel not in known:
            n_new += 1
        elif force or known[rel] != (size, mtime) or rel in had_error:
            n_changed += 1
        else:
            continue
        jobs.append((abs_path, rel, size, mtime))

    deleted = set(known) - set(on_disk)
    log.info("New: %d | changed/retry: %d | unchanged: %d | deleted: %d",
             n_new, n_changed, len(on_disk) - len(jobs), len(deleted))

    if not jobs and not deleted:
        log.info("Nothing to update.")
        return existing

    # Extract
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            new_rows = list(pool.map(process_file, jobs, chunksize=8))
    else:
        new_rows = [process_file(j) for j in jobs]

    n_errors = sum(r[ERROR_COL] is not None for r in new_rows)
    if n_errors:
        log.warning("%d file(s) could not be read; see the %s column", n_errors, ERROR_COL)

    # Merge: drop stale/deleted rows, append fresh ones. Unknown attributes
    # automatically become new columns (NaN for files that lack them).
    replaced = {r[PATH_COL] for r in new_rows} | deleted
    kept = existing[~existing[PATH_COL].isin(replaced)]
    frames = [f for f in (kept, pd.DataFrame(new_rows)) if not f.empty]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else existing.iloc[0:0]

    # Column order: bookkeeping, then previously known attributes, then new ones
    old_attrs = [c for c in existing.columns if c not in BOOKKEEPING_COLS]
    new_attrs = sorted(c for c in combined.columns
                       if c not in BOOKKEEPING_COLS and c not in old_attrs)
    if new_attrs:
        log.info("New attribute column(s): %s", ", ".join(new_attrs))
    attr_cols = [c for c in old_attrs + new_attrs if c in combined.columns]
    combined = combined.reindex(columns=BOOKKEEPING_COLS + attr_cols)

    # Remove attribute columns no remaining file has (e.g. after deletions)
    empty = [c for c in attr_cols if combined[c].isna().all()]
    combined = combined.drop(columns=empty)

    # convert_dtypes keeps integer attributes as integers even when some files lack them
    combined = combined.sort_values(PATH_COL, ignore_index=True).convert_dtypes()
    save_table(combined, table_path)
    log.info("Wrote %d rows x %d columns to %s", *combined.shape, table_path)
    return combined


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, help="Root folder containing NetCDF files (searched recursively)")
    ap.add_argument("-o", "--output", type=Path, default=Path("nc_metadata.csv"),
                    help="Output table (.csv or .parquet). Default: nc_metadata.csv")
    ap.add_argument("-p", "--pattern", action="append", dest="patterns",
                    help=f"Filename glob, repeatable. Default: {' '.join(DEFAULT_PATTERNS)}")
    ap.add_argument("-w", "--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="Parallel worker processes (default: up to 4)")
    ap.add_argument("--force", action="store_true", help="Re-extract all files, ignoring change detection")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not args.root.is_dir():
        log.error("Root folder does not exist: %s", args.root)
        return 1
    update_metadata(args.root.resolve(), args.output, args.patterns or DEFAULT_PATTERNS,
                    workers=args.workers, force=args.force)
    return 0

def run_local():
    logf = '/data/log/metadata_tables.log'
    logging.basicConfig(filename=logf,
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    log.info("Started")
    update_metadata([Path("/data/OG_nrt"), Path("/data/OG_complete")], Path("/data/meta/meta_og1.csv"), '*.nc', workers=8, force=True)
    log.info("Complete")


if __name__ == "__main__":
    run_local()