import argparse
import os
import random
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torchvision import transforms


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


PROTOCOLS = OrderedDict(
    {
        "O_M_to_C": {"seen": ["OULU", "MSU"], "unseen": ["CASIA"]},
        "O_C_to_M": {"seen": ["OULU", "CASIA"], "unseen": ["MSU"]},
        "C_M_to_O": {"seen": ["CASIA", "MSU"], "unseen": ["OULU"]},
    }
)

METHODS = OrderedDict(
    {
        "gdfas": {
            "name": "GD-FAS",
            "checkpoint": os.path.join("results", "{protocol}", "{protocol}_best.pth"),
        },
        "coop": {
            "name": "GD-FAS + CoOp",
            "checkpoint": os.path.join(
                "results_coop", "results", "test", "{protocol}_best.pth"
            ),
        },
    }
)

DOMAIN_COLORS = {
    "OULU": "#1f77b4",
    "CASIA": "#ff7f0e",
    "MSU": "#2ca02c",
}

LABEL_MARKERS = {
    0: "x",
    1: ".",
}

LABEL_NAMES = {
    0: "attack",
    1: "live",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA visualization for CLIP image encoder features."
    )
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--output_dir", default=os.path.join("visualizations", "pca_clip"))
    parser.add_argument(
        "--protocol",
        action="append",
        choices=list(PROTOCOLS.keys()),
        help="Protocol to visualize. Can be repeated. Default: all protocols.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=list(METHODS.keys()),
        help="Method to visualize. Can be repeated. Default: both methods.",
    )
    parser.add_argument("--videos_per_class", type=int, default=80)
    parser.add_argument("--frames_per_video", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--pca_trim_rate",
        type=float,
        default=0.3,
        help=(
            "Fraction of PCA-coordinate tail points to trim for plotting and "
            "boundary fitting. Default 0.3 keeps the central 70% points."
        ),
    )
    parser.add_argument(
        "--axis_padding",
        type=float,
        default=0.04,
        help="Padding ratio around visible PCA points.",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Save high-dimensional features and PCA coordinates.",
    )
    return parser.parse_args()


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def list_video_dirs(data_root, domain, label_name):
    label_dir = os.path.join(data_root, domain, "test", label_name)
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Missing dataset directory: {label_dir}")
    video_dirs = [
        os.path.join(label_dir, name)
        for name in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, name))
    ]
    return sorted(video_dirs)


def list_image_files(video_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [
        os.path.join(video_dir, name)
        for name in sorted(os.listdir(video_dir))
        if os.path.splitext(name.lower())[1] in extensions
    ]


def choose_evenly(items, count):
    if count <= 0 or len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count)
    return [items[int(round(index))] for index in indices]


def build_samples(data_root, domains, videos_per_class, frames_per_video, seed):
    rng = random.Random(seed)
    samples = []

    for domain in domains:
        for label_name, label in [("attack", 0), ("live", 1)]:
            video_dirs = list_video_dirs(data_root, domain, label_name)
            rng.shuffle(video_dirs)
            video_dirs = sorted(video_dirs[:videos_per_class])

            for video_dir in video_dirs:
                frame_paths = list_image_files(video_dir)
                if not frame_paths:
                    continue
                frame_paths = choose_evenly(frame_paths, frames_per_video)
                samples.append(
                    {
                        "domain": domain,
                        "label": label,
                        "video": video_dir,
                        "frames": frame_paths,
                    }
                )

    return samples


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model.eval()
    model.to(device)
    return model


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_clip_features(model, samples, transform, batch_size, device):
    features = []
    labels = []
    domains = []
    videos = []

    with torch.no_grad():
        for sample in samples:
            frame_features = []
            for frame_batch in batched(sample["frames"], batch_size):
                images = []
                for frame_path in frame_batch:
                    with Image.open(frame_path) as image:
                        images.append(transform(image.convert("RGB")))

                images = torch.stack(images, dim=0).to(device)
                image_features = model.model.encode_image(images)
                image_features = image_features.float()
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                frame_features.append(image_features.cpu())

            video_feature = torch.cat(frame_features, dim=0).mean(dim=0)
            video_feature = video_feature / video_feature.norm().clamp_min(1e-12)
            features.append(video_feature.numpy())
            labels.append(sample["label"])
            domains.append(sample["domain"])
            videos.append(sample["video"])

    return {
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "domains": np.asarray(domains),
        "videos": np.asarray(videos),
    }


def fit_protocol_pca(group_method_features):
    all_features = np.concatenate(
        [
            payload["features"]
            for method_features in group_method_features.values()
            for payload in method_features.values()
        ],
        axis=0,
    )
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_features)
    coords = {}
    for group_name, method_features in group_method_features.items():
        coords[group_name] = {
            method: pca.transform(payload["features"])
            for method, payload in method_features.items()
        }
    return pca, coords


def pca_quantile_mask(coords, trim_rate):
    if trim_rate <= 0:
        return np.ones(coords.shape[0], dtype=bool)
    if trim_rate >= 1:
        raise ValueError("--pca_trim_rate must be smaller than 1.")

    ranks = np.empty_like(coords, dtype=np.float64)
    for dim in range(coords.shape[1]):
        order = np.argsort(coords[:, dim])
        ranks[order, dim] = np.linspace(0.0, 1.0, coords.shape[0])

    tail_score = np.max(np.abs(ranks - 0.5), axis=1)
    threshold = np.quantile(tail_score, 1.0 - trim_rate)
    return tail_score <= threshold


def line_from_logistic(model, x_min, x_max, y_min, y_max):
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    xs = np.linspace(x_min, x_max, 200)

    if abs(coef[1]) < 1e-8:
        if abs(coef[0]) < 1e-8:
            return None, None
        x = -intercept / coef[0]
        return np.asarray([x, x]), np.asarray([y_min, y_max])

    ys = -(coef[0] * xs + intercept) / coef[1]
    keep = (ys >= y_min) & (ys <= y_max)
    if keep.sum() < 2:
        return None, None
    return xs[keep], ys[keep]


def draw_boundary(ax, points, labels, color, linestyle, linewidth, label):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(points, labels)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xs, ys = line_from_logistic(clf, x_min, x_max, y_min, y_max)
    if xs is None:
        return
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def plot_method(
    coords,
    labels,
    domains,
    method_name,
    protocol,
    group_name,
    group_domains,
    output_path,
    dpi,
    pca_trim_rate,
    axis_padding,
):
    visible_mask = pca_quantile_mask(coords, pca_trim_rate)
    if visible_mask.sum() >= 4 and len(np.unique(labels[visible_mask])) == 2:
        coords = coords[visible_mask]
        labels = labels[visible_mask]
        domains = domains[visible_mask]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    for domain in group_domains:
        domain_mask = domains == domain
        for label in [0, 1]:
            mask = domain_mask & (labels == label)
            if not mask.any():
                continue
            marker = LABEL_MARKERS[label]
            size = 44 if label == 0 else 70
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=DOMAIN_COLORS[domain],
                marker=marker,
                s=size,
                linewidths=1.2 if label == 0 else 0.0,
                alpha=0.78,
                label=f"{domain} {LABEL_NAMES[label]}",
            )

    x_span = coords[:, 0].max() - coords[:, 0].min()
    y_span = coords[:, 1].max() - coords[:, 1].min()
    pad_x = max(x_span * axis_padding, 1e-6)
    pad_y = max(y_span * axis_padding, 1e-6)
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    if group_name == "seen":
        for domain in group_domains:
            mask = domains == domain
            draw_boundary(
                ax,
                coords[mask],
                labels[mask],
                DOMAIN_COLORS[domain],
                "-",
                1.8,
                f"{domain} boundary",
            )
        draw_boundary(
            ax,
            coords,
            labels,
            "black",
            "--",
            2.1,
            "overall boundary",
        )
    else:
        color = DOMAIN_COLORS[group_domains[0]]
        draw_boundary(
            ax,
            coords,
            labels,
            color,
            "-",
            2.0,
            f"{group_domains[0]} boundary",
        )

    ax.set_title(f"{protocol} {group_name} - {method_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_npz(
    output_path,
    pca,
    method_features,
    method_coords,
    protocol,
    group_name,
    pca_trim_rate,
):
    payload = {
        "protocol": protocol,
        "group": group_name,
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
    }
    for method, features in method_features.items():
        prefix = method
        payload[f"{prefix}_features"] = features["features"]
        payload[f"{prefix}_coords"] = method_coords[method]
        payload[f"{prefix}_visible_mask"] = pca_quantile_mask(
            method_coords[method], pca_trim_rate
        )
        payload[f"{prefix}_labels"] = features["labels"]
        payload[f"{prefix}_domains"] = features["domains"]
        payload[f"{prefix}_videos"] = features["videos"]
    np.savez_compressed(output_path, **payload)


def main():
    args = parse_args()
    if not 0 <= args.pca_trim_rate < 1:
        raise ValueError("--pca_trim_rate must be in [0, 1).")
    if args.axis_padding < 0:
        raise ValueError("--axis_padding must be non-negative.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available.")

    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(REPO_ROOT, data_root)
    data_root = os.path.abspath(data_root)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)
    output_dir = os.path.abspath(output_dir)
    protocols = args.protocol or list(PROTOCOLS.keys())
    methods = args.method or list(METHODS.keys())
    transform = build_transform()

    os.makedirs(output_dir, exist_ok=True)

    for protocol in protocols:
        protocol_dir = os.path.join(output_dir, protocol)
        os.makedirs(protocol_dir, exist_ok=True)

        loaded_models = {}
        for method in methods:
            checkpoint = METHODS[method]["checkpoint"].format(protocol=protocol)
            if not os.path.isabs(checkpoint):
                checkpoint = os.path.join(REPO_ROOT, checkpoint)
            checkpoint = os.path.abspath(checkpoint)
            loaded_models[method] = load_checkpoint(checkpoint, args.device)

        group_samples = {}
        group_method_features = {}
        for group_name in ["seen", "unseen"]:
            group_domains = PROTOCOLS[protocol][group_name]
            group_samples[group_name] = build_samples(
                data_root,
                group_domains,
                args.videos_per_class,
                args.frames_per_video,
                args.seed,
            )
            if not group_samples[group_name]:
                raise RuntimeError(f"No samples found for {protocol} {group_name}.")

            group_method_features[group_name] = {}
            for method, model in loaded_models.items():
                group_method_features[group_name][method] = extract_clip_features(
                    model,
                    group_samples[group_name],
                    transform,
                    args.batch_size,
                    args.device,
                )

        pca, group_method_coords = fit_protocol_pca(group_method_features)

        for group_name in ["seen", "unseen"]:
            group_domains = PROTOCOLS[protocol][group_name]
            method_features = group_method_features[group_name]
            method_coords = group_method_coords[group_name]
            if args.save_npz:
                save_npz(
                    os.path.join(protocol_dir, f"{group_name}_clip_pca_features.npz"),
                    pca,
                    method_features,
                    method_coords,
                    protocol,
                    group_name,
                    args.pca_trim_rate,
                )

            for method in methods:
                output_path = os.path.join(
                    protocol_dir, f"{group_name}_{method}_clip_pca.png"
                )
                plot_method(
                    method_coords[method],
                    method_features[method]["labels"],
                    method_features[method]["domains"],
                    METHODS[method]["name"],
                    protocol,
                    group_name,
                    group_domains,
                    output_path,
                    args.dpi,
                    args.pca_trim_rate,
                    args.axis_padding,
                )


if __name__ == "__main__":
    main()
