"""
Fix quaternion sign ambiguity in hand pretrain data.

Quaternions q and -q represent the same rotation. When delta rotations are stored
as quaternions, qw should be close to +1 for small rotations. However, ~0.63% of
frames have qw near -1 due to sign ambiguity, creating a bimodal distribution that
destabilizes training (NaN/inf loss).

This script:
1. Iterates all parquet files across all shards
2. For each frame, if qw < 0, negates the entire quaternion (qx,qy,qz,qw)
3. Applies to both action and observation.state columns
4. Rewrites parquet files in-place
5. Recomputes stats_gr00t.json and episodes_stats.jsonl for each shard
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ACTION_LEFT_QUAT_DIMS = [3, 4, 5, 6]    # left qx, qy, qz, qw
ACTION_LEFT_QW_DIM = 6
ACTION_RIGHT_QUAT_DIMS = [11, 12, 13, 14]  # right qx, qy, qz, qw
ACTION_RIGHT_QW_DIM = 14

STATE_LEFT_QUAT_DIMS = [3, 4, 5, 6]
STATE_LEFT_QW_DIM = 6
STATE_RIGHT_QUAT_DIMS = [11, 12, 13, 14]
STATE_RIGHT_QW_DIM = 14


def fix_quaternion_sign(arr: np.ndarray, quat_dims: list[int], qw_dim: int) -> tuple[np.ndarray, int]:
    """Negate quaternion components where qw < 0 to ensure sign consistency.
    
    Returns the fixed array and count of flipped frames.
    """
    qw = arr[:, qw_dim]
    flip_mask = qw < 0
    n_flipped = flip_mask.sum()
    if n_flipped > 0:
        arr[flip_mask][:, quat_dims] *= -1
        for dim in quat_dims:
            arr[flip_mask, dim] *= -1
    return arr, int(n_flipped)


def fix_parquet_file(parquet_path: Path, dry_run: bool = False) -> dict:
    """Fix quaternion signs in a single parquet file. Returns fix statistics."""
    df = pd.read_parquet(parquet_path)
    stats = {"action_flips": 0, "state_flips": 0, "total_frames": len(df)}

    for col, left_quat, left_qw, right_quat, right_qw, stat_key in [
        ("action", ACTION_LEFT_QUAT_DIMS, ACTION_LEFT_QW_DIM,
         ACTION_RIGHT_QUAT_DIMS, ACTION_RIGHT_QW_DIM, "action_flips"),
        ("observation.state", STATE_LEFT_QUAT_DIMS, STATE_LEFT_QW_DIM,
         STATE_RIGHT_QUAT_DIMS, STATE_RIGHT_QW_DIM, "state_flips"),
    ]:
        if col not in df.columns:
            continue
        arr = np.stack(df[col].values).astype(np.float32)
        _, n1 = fix_quaternion_sign(arr, left_quat, left_qw)
        _, n2 = fix_quaternion_sign(arr, right_quat, right_qw)
        stats[stat_key] = n1 + n2

        if (n1 + n2) > 0 and not dry_run:
            df[col] = list(arr)

    if not dry_run and (stats["action_flips"] > 0 or stats["state_flips"] > 0):
        df.to_parquet(parquet_path, index=False)

    return stats


def compute_episode_stats(df: pd.DataFrame, episode_index: int) -> dict:
    """Compute per-episode statistics matching the episodes_stats.jsonl format."""
    ep_stats = {}
    for col in df.columns:
        try:
            arr = np.stack(df[col].values).astype(np.float64)
        except (ValueError, TypeError):
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Skip image columns
        if arr.ndim > 2:
            ep_stats[col] = {
                "min": [[[0.0]], [[0.0]], [[0.0]]],
                "max": [[[1.0]], [[1.0]], [[1.0]]],
                "mean": [[[0.5]], [[0.5]], [[0.5]]],
                "std": [[[0.0]], [[0.0]], [[0.0]]],
                "count": [len(df)],
            }
            continue

        ep_stats[col] = {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "count": [len(df)],
        }
    return {"episode_index": episode_index, "stats": ep_stats}


def compute_shard_stats(shard_path: Path) -> dict:
    """Compute aggregated shard-level statistics (stats_gr00t.json format)."""
    parquet_files = sorted(shard_path.glob("data/*/*.parquet"))
    if not parquet_files:
        return {}

    all_data = {}
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for col in df.columns:
            try:
                arr = np.stack(df[col].values).astype(np.float64)
            except (ValueError, TypeError):
                continue
            if col not in all_data:
                all_data[col] = []
            all_data[col].append(arr)

    stats = {}
    for col, data_list in all_data.items():
        arr = np.concatenate(data_list, axis=0)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim > 2:
            continue

        stats[col] = {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return stats


def process_shard(shard_path: Path, dry_run: bool = False, recompute_stats: bool = True) -> dict:
    """Process a single shard: fix quaternions and recompute stats."""
    parquet_files = sorted(shard_path.glob("data/*/*.parquet"))
    shard_stats = {
        "shard": shard_path.name,
        "total_files": len(parquet_files),
        "total_action_flips": 0,
        "total_state_flips": 0,
        "total_frames": 0,
    }

    for pf in parquet_files:
        file_stats = fix_parquet_file(pf, dry_run=dry_run)
        shard_stats["total_action_flips"] += file_stats["action_flips"]
        shard_stats["total_state_flips"] += file_stats["state_flips"]
        shard_stats["total_frames"] += file_stats["total_frames"]

    if not dry_run and recompute_stats:
        # Recompute stats_gr00t.json
        gr00t_stats = compute_shard_stats(shard_path)
        stats_path = shard_path / "meta" / "stats_gr00t.json"
        with open(stats_path, "w") as f:
            json.dump(gr00t_stats, f, indent=4)

        # Recompute episodes_stats.jsonl
        episodes_stats_path = shard_path / "meta" / "episodes_stats.jsonl"
        with open(episodes_stats_path, "w") as f:
            for pf in sorted(shard_path.glob("data/*/*.parquet")):
                ep_idx = int(pf.stem.split("_")[-1])
                df = pd.read_parquet(pf)
                ep_stat = compute_episode_stats(df, ep_idx)
                f.write(json.dumps(ep_stat) + "\n")

    return shard_stats


def main():
    parser = argparse.ArgumentParser(description="Fix quaternion sign ambiguity in hand pretrain data")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/nas/guchenyang/Data/Hand-Data/Pretrain-0207-Clean",
        help="Root directory of hand pretrain data",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only report issues without modifying data")
    parser.add_argument("--no_recompute_stats", action="store_true", help="Skip stats recomputation")
    parser.add_argument("--shards", type=str, default=None, help="Comma-separated shard indices to process (e.g., '0,1,2'). Default: all")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: data root {data_root} does not exist")
        sys.exit(1)

    shard_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("shard_")])
    if args.shards is not None:
        indices = [int(x) for x in args.shards.split(",")]
        shard_dirs = [d for d in shard_dirs if int(d.name.split("_")[1]) in indices]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(shard_dirs)} shards from {data_root}")
    print(f"Recompute stats: {not args.no_recompute_stats}")
    print()

    total_flips_action = 0
    total_flips_state = 0
    total_frames = 0

    for shard_dir in tqdm(shard_dirs, desc="Processing shards"):
        stats = process_shard(
            shard_dir,
            dry_run=args.dry_run,
            recompute_stats=not args.no_recompute_stats,
        )
        total_flips_action += stats["total_action_flips"]
        total_flips_state += stats["total_state_flips"]
        total_frames += stats["total_frames"]

        if stats["total_action_flips"] > 0 or stats["total_state_flips"] > 0:
            tqdm.write(
                f"  {stats['shard']}: {stats['total_files']} files, "
                f"{stats['total_frames']} frames, "
                f"action_flips={stats['total_action_flips']}, "
                f"state_flips={stats['total_state_flips']}"
            )

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Total frames: {total_frames}")
    print(f"  Total action quaternion flips: {total_flips_action} ({100*total_flips_action/max(total_frames,1):.2f}%)")
    print(f"  Total state quaternion flips: {total_flips_state} ({100*total_flips_state/max(total_frames,1):.2f}%)")


if __name__ == "__main__":
    main()
