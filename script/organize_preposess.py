import argparse
import re
import shutil
from pathlib import Path


CASIA_DIR_RE = re.compile(
    r"^casia_(?P<split>train|test)_(?P<label>live|spoof)_(?P<subject>\d+)_(?P<scene>[A-Za-z]+)_(?P<take>\d+)$"
)
IMAGE_RE = re.compile(r"^crop_(?P<frame>\d+)\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize ./datasets/preposess into the README datasets layout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/preposess"),
        help="Source directory containing preprocessed video folders.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("datasets"),
        help="Root directory for the organized datasets.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place frames into the target layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target images if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned work without creating files.",
    )
    return parser.parse_args()


def parse_casia_dir_name(dir_name: str):
    match = CASIA_DIR_RE.match(dir_name)
    if not match:
        return None

    split = match.group("split")
    label = "live" if match.group("label") == "live" else "attack"
    video_name = f"{match.group('subject')}_{match.group('scene')}_{match.group('take')}"

    return {
        "dataset_name": "CASIA",
        "split": split,
        "label": label,
        "video_name": video_name,
    }


def iter_crop_images(video_dir: Path):
    images = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file():
            continue
        match = IMAGE_RE.match(path.name)
        if match:
            images.append((path, f"{match.group('frame')}.{match.group('ext').lower()}"))
    return images


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
        src_hardlink = src.resolve()
        dst.hardlink_to(src_hardlink)
    return True


def organize_video_dir(video_dir: Path, target_root: Path, mode: str, overwrite: bool, dry_run: bool):
    parsed = parse_casia_dir_name(video_dir.name)
    if parsed is None:
        return {
            "status": "skipped",
            "reason": "unsupported_name",
            "video_dir": video_dir,
            "frames": 0,
        }

    crop_images = iter_crop_images(video_dir)
    if not crop_images:
        return {
            "status": "skipped",
            "reason": "no_crop_images",
            "video_dir": video_dir,
            "frames": 0,
        }

    target_video_dir = (
        target_root
        / parsed["dataset_name"]
        / parsed["split"]
        / parsed["label"]
        / parsed["video_name"]
    )

    created = 0
    for src_image, target_name in crop_images:
        dst_image = target_video_dir / target_name
        if place_file(src_image, dst_image, mode, overwrite, dry_run):
            created += 1

    return {
        "status": "ok",
        "reason": "",
        "video_dir": video_dir,
        "frames": created,
        "target_video_dir": target_video_dir,
    }


def main():
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    stats = {
        "videos_total": 0,
        "videos_ok": 0,
        "videos_skipped": 0,
        "frames_created": 0,
    }
    skipped = []

    for video_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        stats["videos_total"] += 1
        result = organize_video_dir(
            video_dir=video_dir,
            target_root=target,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

        if result["status"] == "ok":
            stats["videos_ok"] += 1
            stats["frames_created"] += result["frames"]
        else:
            stats["videos_skipped"] += 1
            skipped.append(result)

    print(f"source      : {source}")
    print(f"target      : {target}")
    print(f"mode        : {args.mode}")
    print(f"dry_run     : {args.dry_run}")
    print(f"videos_total: {stats['videos_total']}")
    print(f"videos_ok   : {stats['videos_ok']}")
    print(f"videos_skip : {stats['videos_skipped']}")
    print(f"frames_done : {stats['frames_created']}")

    if skipped:
        print("skipped_details:")
        for item in skipped[:20]:
            print(f"  - {item['video_dir'].name}: {item['reason']}")
        if len(skipped) > 20:
            print(f"  - ... {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
