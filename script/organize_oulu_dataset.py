import argparse
import shutil
from pathlib import Path
import re


SOURCE_FOLDERS = {
    "oulu_train_live_OULU-NPU": ("train", "live", "Train_files"),
    "oulu_train_spoof_OULU-NPU": ("train", "attack", "Train_files"),
    "oulu_test_live_OULU-NPU": ("test", "live", "Test_files"),
    "oulu_test_spoof_OULU-NPU": ("test", "attack", "Test_files"),
}

IMAGE_RE = re.compile(r"^crop_(?P<frame>\d+)\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organize datasets_tmp/FAS/OULU-NPU/preposess into the model-ready dataset/OULU layout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets_tmp/FAS/OULU-NPU/preposess"),
        help="Root directory of the preprocessed OULU-NPU data.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("dataset/OULU"),
        help="Target OULU dataset directory used by the model.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place image files into the target layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target images if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned summary without creating files.",
    )
    return parser.parse_args()


def iter_frame_files(video_dir: Path):
    for path in sorted(video_dir.iterdir()):
        if not path.is_file():
            continue
        match = IMAGE_RE.match(path.name)
        if match:
            yield path, f"{match.group('frame')}.{match.group('ext').lower()}"


def place_file(src: Path, dst: Path, mode: str, overwrite: bool, dry_run: bool):
    if dst.exists():
        if not overwrite:
            return False
        if not dry_run:
            dst.unlink()

    if dry_run:
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.hardlink_to(src.resolve())
    return True


def organize_split(source_root: Path, target_root: Path, mode: str, overwrite: bool, dry_run: bool):
    stats = {
        "videos_total": 0,
        "videos_ok": 0,
        "videos_skipped": 0,
        "frames_done": 0,
    }
    skipped = []

    for folder_name, (phase, label, inner_name) in SOURCE_FOLDERS.items():
        video_root = source_root / folder_name / inner_name
        if not video_root.exists():
            skipped.append((folder_name, "missing_source"))
            continue

        for video_dir in sorted(path for path in video_root.iterdir() if path.is_dir()):
            stats["videos_total"] += 1
            frames = list(iter_frame_files(video_dir))
            if not frames:
                stats["videos_skipped"] += 1
                skipped.append((video_dir.name, "no_crop_images"))
                continue

            target_video_dir = target_root / phase / label / video_dir.name
            created = 0
            for src_image, target_name in frames:
                dst_image = target_video_dir / target_name
                if place_file(src_image, dst_image, mode, overwrite, dry_run):
                    created += 1

            stats["videos_ok"] += 1
            stats["frames_done"] += created

    return stats, skipped


def main():
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    stats, skipped = organize_split(
        source_root=source,
        target_root=target,
        mode=args.mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    print(f"source      : {source}")
    print(f"target      : {target}")
    print(f"mode        : {args.mode}")
    print(f"dry_run     : {args.dry_run}")
    print(f"videos_total: {stats['videos_total']}")
    print(f"videos_ok   : {stats['videos_ok']}")
    print(f"videos_skip : {stats['videos_skipped']}")
    print(f"frames_done : {stats['frames_done']}")
    if skipped:
        print("skipped_details:")
        for name, reason in skipped[:20]:
            print(f"  - {name}: {reason}")
        if len(skipped) > 20:
            print(f"  - ... {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
