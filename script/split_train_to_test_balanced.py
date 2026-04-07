import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move half of the samples from datasets_tmp/Train_files to datasets_tmp/Test_files while keeping class buckets balanced."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("datasets_tmp/Train_files"),
        help="Source directory containing paired .avi/.txt samples.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("datasets_tmp/Test_files"),
        help="Target directory for moved samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic selection order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned moves without touching files.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    return parser.parse_args()


def collect_samples(train_dir: Path):
    samples = {}
    for avi_path in sorted(train_dir.glob("*.avi")):
        stem = avi_path.stem
        txt_path = train_dir / f"{stem}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing label/metadata pair for sample: {avi_path}")

        parts = stem.split("_")
        if len(parts) != 4:
            raise ValueError(f"Unexpected sample name format: {stem}")

        samples[stem] = {
            "stem": stem,
            "parts": parts,
            "avi": avi_path,
            "txt": txt_path,
        }
    return samples


def plan_split(samples, seed):
    rng = random.Random(seed)

    by_major = defaultdict(list)
    for sample in samples.values():
        by_major[sample["parts"][0]].append(sample)

    selected = []
    stats = {}

    for major_key, major_samples in sorted(by_major.items()):
        target = len(major_samples) // 2
        by_prefix = defaultdict(list)
        for sample in major_samples:
            prefix = "_".join(sample["parts"][:-1])
            by_prefix[prefix].append(sample)

        prefixes = sorted(by_prefix)
        rng.shuffle(prefixes)

        prefix_quota = {}
        assigned = 0
        for prefix in prefixes:
            quota = len(by_prefix[prefix]) // 2
            prefix_quota[prefix] = quota
            assigned += quota

        remaining = target - assigned
        if remaining < 0:
            raise ValueError(f"Internal quota error for bucket {major_key}")

        for prefix in prefixes[:remaining]:
            prefix_quota[prefix] += 1

        chosen = []
        for prefix in prefixes:
            group = sorted(by_prefix[prefix], key=lambda item: item["stem"])
            rng.shuffle(group)
            chosen.extend(group[:prefix_quota[prefix]])

        if len(chosen) != target:
            raise ValueError(
                f"Failed to create balanced plan for bucket {major_key}: expected {target}, got {len(chosen)}"
            )

        selected.extend(chosen)
        stats[major_key] = {
            "total": len(major_samples),
            "to_test": len(chosen),
            "remain_train": len(major_samples) - len(chosen),
        }

    return sorted(selected, key=lambda item: item["stem"]), stats


def transfer_file(src: Path, dst: Path, copy_only: bool, dry_run: bool):
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_only:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def execute_plan(selected, test_dir: Path, copy_only: bool, dry_run: bool):
    for sample in selected:
        transfer_file(sample["avi"], test_dir / sample["avi"].name, copy_only, dry_run)
        transfer_file(sample["txt"], test_dir / sample["txt"].name, copy_only, dry_run)


def main():
    args = parse_args()
    train_dir = args.train_dir.resolve()
    test_dir = args.test_dir.resolve()

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory does not exist: {train_dir}")

    samples = collect_samples(train_dir)
    selected, stats = plan_split(samples, args.seed)

    print(f"train_dir       : {train_dir}")
    print(f"test_dir        : {test_dir}")
    print(f"mode            : {'copy' if args.copy else 'move'}")
    print(f"dry_run         : {args.dry_run}")
    print(f"seed            : {args.seed}")
    print(f"sample_total    : {len(samples)}")
    print(f"sample_to_test  : {len(selected)}")
    print(f"sample_remain   : {len(samples) - len(selected)}")
    print("bucket_stats:")
    for major_key, info in stats.items():
        print(
            f"  class_{major_key}: total={info['total']} to_test={info['to_test']} remain_train={info['remain_train']}"
        )

    execute_plan(selected, test_dir, args.copy, args.dry_run)


if __name__ == "__main__":
    main()
