"""Pre-build per-run VTU caches for timestep_stride=1 on a local machine.

Each run's cache is built and written to disk independently, so peak RAM is
one run at a time (~240 MB) rather than all 111 training runs simultaneously
(which is what causes the SLURM OOM during training init).

Once caches exist on the HPC the cropped_stride1 job will skip rebuilding and
load directly — but note the cache files are large (~240 MB × 139 runs ≈ 33 GB
total) so make sure the destination has enough disk space before copying.

Usage
-----
    uv run python build_stride1_cache.py --data_dir RUN-2
    uv run python build_stride1_cache.py --data_dir RUN-2 --cache_dir /tmp/cache
    uv run python build_stride1_cache.py --data_dir RUN-2 --workers 16 --dry_run
"""

import argparse
import os
import sys
import time

# Make sure the dataset module is importable when run from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples", "cfd", "needle_tissue_cropped"))

from dataset import (  # noqa: E402
    _atomic_torch_save,
    _group_vtu_by_run,
    _process_all_frames,
)


def build_caches(
    data_dir: str,
    cache_dir: str,
    timestep_stride: int = 1,
    workers: int = 8,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Build (or skip) per-run caches for all runs found in *data_dir*."""
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Scanning {data_dir} for VTU files (stride={timestep_stride})...")
    run_files = _group_vtu_by_run(data_dir, timestep_stride)
    if not run_files:
        print("ERROR: no multi-run VTU files found.  Check data_dir.")
        sys.exit(1)

    run_ids = sorted(run_files.keys(), key=int)
    print(f"Found {len(run_ids)} runs: {run_ids[0]} … {run_ids[-1]}")

    skipped = built = 0
    total_t0 = time.time()

    for run_id in run_ids:
        vtu_files = run_files[run_id]
        n_frames = len(vtu_files)
        cache_path = os.path.join(cache_dir, f"preprocessed_cache_RUN-{run_id}.pt")

        if os.path.exists(cache_path) and not force:
            print(f"  RUN-{run_id:>4}: cache exists ({n_frames} frames) — skipping")
            skipped += 1
            continue

        if dry_run:
            print(f"  RUN-{run_id:>4}: would build cache ({n_frames} frames)")
            continue

        print(f"\n  RUN-{run_id:>4}: building cache ({n_frames} frames) ...")
        t0 = time.time()
        cache = _process_all_frames(vtu_files, num_workers=workers)
        _atomic_torch_save(cache, cache_path)
        elapsed = time.time() - t0
        size_mb = os.path.getsize(cache_path) / 1e6
        print(f"  RUN-{run_id:>4}: done in {elapsed:.0f}s  ({size_mb:.0f} MB → {cache_path})")
        built += 1

    total_elapsed = time.time() - total_t0
    print(
        f"\nDone.  Built={built}  Skipped={skipped}  "
        f"Total time={total_elapsed:.0f}s"
    )
    if not dry_run and built > 0:
        total_size_mb = sum(
            os.path.getsize(os.path.join(cache_dir, f))
            for f in os.listdir(cache_dir)
            if f.startswith("preprocessed_cache_RUN-") and f.endswith(".pt")
        ) / 1e6
        print(f"Cache directory size: {total_size_mb:.0f} MB  ({cache_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data_dir",
        default="RUN-2",
        help="Directory containing *-RUN-N_T.vtu files (default: RUN-2)",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Where to write cache files.  Defaults to --data_dir.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Timestep stride when selecting frames per run (default: 1 = all frames)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="ThreadPoolExecutor workers for parallel VTU loading (default: 8)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild caches even if they already exist on disk",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else data_dir

    if not os.path.isdir(data_dir):
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(1)

    build_caches(
        data_dir=data_dir,
        cache_dir=cache_dir,
        timestep_stride=args.stride,
        workers=args.workers,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
