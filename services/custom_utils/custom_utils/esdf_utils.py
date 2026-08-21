"""
https://github.com/gershom96/VLA_DataGeneration/blob/main/tests/esdf-test.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_utils.pointcloud_utils import camera_to_base_transform, pointcloud_to_esdf_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the ESDF pipeline from saved pointcloud .npz files and save "
            "side-by-side debug visualizations."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "output",
        help="Root directory containing processed clips with pointclouds/ and images/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "esdf_debug",
        help="Directory where visualization PNGs will be written.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help="Optional path to a single pointcloud .npz file to inspect.",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Only process pointclouds inside a specific clip directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pointcloud files to process.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth pointcloud file after sorting.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display each figure interactively after saving it.",
    )
    parser.add_argument(
        "--frame-preset",
        choices=("identity", "camera_to_base"),
        default="camera_to_base",
        help=(
            "Coordinate transform applied before ESDF creation. "
            "'camera_to_base' assumes saved points are in camera frame "
            "(x right, y down, z forward) and maps them to base frame "
            "(x forward, y left, z up), including camera-height translation."
        ),
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.5,
        help="Camera height above ground in meters for the camera_to_base preset.",
    )
    parser.add_argument("--h-min", type=float, default=0.3, help="Minimum kept height in meters.")
    parser.add_argument("--h-max", type=float, default=2.0, help="Maximum kept height in meters.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Minimum forward extent in meters.")
    parser.add_argument("--x-max", type=float, default=25.0, help="Maximum forward extent in meters.")
    parser.add_argument("--y-min", type=float, default=-10.0, help="Minimum lateral extent in meters.")
    parser.add_argument("--y-max", type=float, default=10.0, help="Maximum lateral extent in meters.")
    parser.add_argument("--resolution", type=float, default=0.10, help="Grid resolution in meters per cell.")
    parser.add_argument("--sensor-x", type=float, default=0.0, help="Sensor x location in map frame.")
    parser.add_argument("--sensor-y", type=float, default=0.0, help="Sensor y location in map frame.")
    parser.add_argument("--robot-radius", type=float, default=0.25, help="Robot footprint radius in meters.")
    parser.add_argument(
        "--disable-ground-alignment",
        action="store_true",
        help="Disable RANSAC-based pitch/roll correction before ESDF creation.",
    )
    parser.add_argument(
        "--ground-smoothing",
        type=float,
        default=0.65,
        help="Exponential smoothing weight for a prior fitted ground normal.",
    )
    parser.add_argument(
        "--ground-candidate-min-forward",
        type=float,
        default=0.5,
        help="Minimum forward distance used when sampling ground-plane candidates in meters.",
    )
    parser.add_argument(
        "--ground-candidate-z-min",
        type=float,
        default=-2.0,
        help="Minimum candidate height in meters for RANSAC ground-plane fitting.",
    )
    parser.add_argument(
        "--ground-candidate-z-max",
        type=float,
        default=2.0,
        help="Maximum candidate height in meters for RANSAC ground-plane fitting.",
    )
    parser.add_argument(
        "--ground-candidate-cell-size",
        type=float,
        default=0.75,
        help="Coarse XY cell size in meters used to retain low ground-plane candidates.",
    )
    parser.add_argument(
        "--ground-ransac-iterations",
        type=int,
        default=120,
        help="Maximum RANSAC iterations used for per-frame ground-plane fitting.",
    )
    parser.add_argument(
        "--ground-ransac-distance-threshold",
        type=float,
        default=0.08,
        help="Inlier threshold in meters for RANSAC ground-plane fitting.",
    )
    parser.add_argument(
        "--ground-max-tilt-deg",
        type=float,
        default=35.0,
        help="Reject fitted ground planes whose normal tilts more than this from vertical.",
    )
    parser.add_argument(
        "--ground-max-correction-deg",
        type=float,
        default=8.0,
        help="Clamp the applied per-frame pitch/roll correction to this many degrees.",
    )
    return parser.parse_args()


def discover_pointcloud_files(
    input_root: Path,
    sample: Path | None,
    clip_id: str | None,
    limit: int | None,
    stride: int,
) -> list[Path]:
    if sample is not None:
        return [sample.resolve()]

    candidates = sorted(path.resolve() for path in input_root.glob("*/pointclouds/*.npz"))
    if clip_id is not None:
        candidates = [path for path in candidates if path.parents[1].name == clip_id]
    if stride > 1:
        candidates = candidates[::stride]
    if limit is not None:
        candidates = candidates[:limit]
    return candidates


def frame_transform(frame_preset: str, camera_height: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if frame_preset == "identity":
        return None, None
    if frame_preset == "camera_to_base":
        return camera_to_base_transform(camera_height=camera_height)
    raise ValueError(f"Unsupported frame preset: {frame_preset}")


def load_rgb_for_sample(pointcloud_path: Path, fallback_rgb: np.ndarray) -> np.ndarray:
    image_path = pointcloud_path.parents[1] / "images" / f"{pointcloud_path.stem}.jpg"
    if image_path.exists():
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))
    return fallback_rgb


def load_sample_metadata(pointcloud_path: Path) -> dict[str, Any] | None:
    metadata_path = pointcloud_path.parents[1] / "metadata.jsonl"
    if not metadata_path.exists():
        return None

    sample_id = pointcloud_path.stem
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("sample_id") == sample_id:
                return record
    return None


def semantic_bev_image(
    occupied_mask: np.ndarray,
    visible_free_mask: np.ndarray,
    unknown_mask: np.ndarray,
) -> np.ndarray:
    semantic = np.ones((*occupied_mask.shape, 3), dtype=np.float32)
    semantic[unknown_mask] = np.asarray([0.55, 0.55, 0.55], dtype=np.float32)
    semantic[visible_free_mask] = np.asarray([0.75, 0.92, 1.0], dtype=np.float32)
    semantic[occupied_mask] = np.asarray([0.88, 0.20, 0.20], dtype=np.float32)
    return semantic


def finite_percentile_abs(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.percentile(np.abs(finite), percentile))
    return max(vmax, 1e-6)


def detection_color(index: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab20")
    return cmap(index % cmap.N)


def color_to_uint8(color: Sequence[float]) -> tuple[int, int, int]:
    return tuple(int(round(255 * float(channel))) for channel in color[:3])


def image_source_size_for_boxes(
    metadata: dict[str, Any] | None,
    full_image_size: Sequence[int] | None,
    rgb: np.ndarray,
) -> tuple[float, float]:
    if metadata is not None:
        image_size = metadata.get("image_size")
        if isinstance(image_size, dict):
            width = float(image_size.get("width", 0) or 0)
            height = float(image_size.get("height", 0) or 0)
            if width > 0 and height > 0:
                return width, height

    if full_image_size is not None and len(full_image_size) >= 2:
        width = float(full_image_size[0])
        height = float(full_image_size[1])
        if width > 0 and height > 0:
            return width, height

    return float(rgb.shape[1]), float(rgb.shape[0])


def scale_bbox_to_display(
    bbox: Sequence[float],
    source_size: tuple[float, float],
    display_shape: tuple[int, int],
    normalized: bool = False,
) -> list[float]:
    disp_h, disp_w = display_shape
    if normalized:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        return [x1 * disp_w, y1 * disp_h, x2 * disp_w, y2 * disp_h]

    src_w, src_h = source_size
    if src_w <= 0 or src_h <= 0:
        return [float(v) for v in bbox]

    scale_x = disp_w / src_w
    scale_y = disp_h / src_h
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def detection_bbox_is_normalized(detection: dict[str, Any], bbox: Sequence[float]) -> bool:
    bbox_format = str(detection.get("bbox_format", "")).strip().lower()
    if bbox_format in {"xyxy_norm", "xyxy_normalized", "normalized_xyxy"}:
        return True

    if len(bbox) != 4:
        return False
    values = [float(v) for v in bbox]
    return all(0.0 <= value <= 1.0 for value in values)


def annotate_rgb_with_detections(
    rgb: np.ndarray,
    detections: Sequence[dict[str, Any]],
    source_size: tuple[float, float],
) -> np.ndarray:
    annotated = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(annotated)

    for index, detection in enumerate(detections):
        bbox = detection.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = scale_bbox_to_display(
            bbox,
            source_size,
            rgb.shape[:2],
            normalized=detection_bbox_is_normalized(detection, bbox),
        )
        if x2 <= x1 or y2 <= y1:
            continue

        color = color_to_uint8(detection_color(index))
        label = detection.get("label") or detection.get("raw_label") or f"obj {index + 1}"
        object_id = detection.get("object_id", index + 1)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2.0, max(0.0, y1 - 14.0)), f"{object_id}: {label}", fill=color)

    return np.asarray(annotated)


def plot_goal_markers(
    ax: plt.Axes,
    detections: Sequence[dict[str, Any]],
) -> None:
    for index, detection in enumerate(detections):
        goal_xy = detection.get("goal_xy_m")
        if not isinstance(goal_xy, (list, tuple)) or len(goal_xy) < 2:
            continue

        goal_x = float(goal_xy[0])
        goal_y = float(goal_xy[1])
        if not np.isfinite(goal_x) or not np.isfinite(goal_y):
            continue

        color = detection_color(index)
        object_id = detection.get("object_id", index + 1)
        ax.scatter([goal_x], [goal_y], marker="x", s=90, c=[color], linewidths=2.0, zorder=5)
        ax.text(goal_x + 0.15, goal_y + 0.15, str(object_id), color=color, fontsize=9, weight="bold", zorder=6)


def metadata_object_goals(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if metadata is None:
        return []

    object_goals = metadata.get("object_goals")
    if isinstance(object_goals, list):
        return [goal for goal in object_goals if isinstance(goal, dict)]

    reference_goals = metadata.get("reference_goals")
    if isinstance(reference_goals, list):
        return [goal for goal in reference_goals if isinstance(goal, dict)]

    detections = metadata.get("detections")
    if isinstance(detections, list):
        return [goal for goal in detections if isinstance(goal, dict)]

    return []


def plot_bool_map(
    ax: plt.Axes,
    mask: np.ndarray,
    extent: Sequence[float],
    title: str,
    sensor_xy: tuple[float, float],
) -> None:
    ax.imshow(mask.astype(np.float32), origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=1.0)
    ax.scatter([sensor_xy[0]], [sensor_xy[1]], marker="x", s=36, c="yellow", linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")


def plot_scalar_map(
    ax: plt.Axes,
    values: np.ndarray,
    extent: Sequence[float],
    title: str,
    sensor_xy: tuple[float, float],
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(values, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter([sensor_xy[0]], [sensor_xy[1]], marker="x", s=36, c="yellow", linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_debug_figure(
    pointcloud_path: Path,
    rgb: np.ndarray,
    metadata: dict[str, Any] | None,
    full_image_size: Sequence[int] | None,
    result: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    extent = [args.x_min, args.x_max, args.y_min, args.y_max]
    sensor_xy = (args.sensor_x, args.sensor_y)
    filtered = result["points_filtered"]
    esdf = result["esdf"]
    clearance = result["clearance"]
    object_goals = metadata_object_goals(metadata)
    source_size = image_source_size_for_boxes(metadata, full_image_size, rgb)
    esdf_scale = finite_percentile_abs(esdf, percentile=99.0)
    clearance_scale = finite_percentile_abs(clearance, percentile=99.0)
    ground_alignment = result.get("ground_alignment") if isinstance(result, dict) else None
    alignment_text = ""
    if isinstance(ground_alignment, dict) and ground_alignment.get("enabled"):
        source = ground_alignment.get("normal_source", "unknown")
        applied_tilt = ground_alignment.get("applied_tilt_deg")
        if applied_tilt is None:
            alignment_text = f" | ground align: {source}"
        else:
            alignment_text = f" | ground align: {source} {float(applied_tilt):.1f} deg"

    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
    fig.suptitle(
        (
            f"{pointcloud_path.parents[1].name}/{pointcloud_path.stem} | "
            f"filtered points: {filtered.shape[0]} | "
            f"frame: {args.frame_preset} | res: {args.resolution:.2f} m"
            f"{alignment_text}"
        ),
        fontsize=14,
    )

    ax = axes[0, 0]
    ax.imshow(annotate_rgb_with_detections(rgb, object_goals, source_size))
    ax.set_title("Egocentric RGB + Object Goals")
    ax.axis("off")

    ax = axes[0, 1]
    if filtered.shape[0] > 0:
        point_count = filtered.shape[0]
        if point_count > 25000:
            sample_idx = np.linspace(0, point_count - 1, 25000).astype(np.int32)
            filtered_plot = filtered[sample_idx]
        else:
            filtered_plot = filtered
        scatter = ax.scatter(
            filtered_plot[:, 0],
            filtered_plot[:, 1],
            c=filtered_plot[:, 2],
            s=1.0,
            alpha=0.45,
            cmap="viridis",
            linewidths=0.0,
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="height z (m)")
    plot_goal_markers(ax, object_goals)
    ax.scatter([sensor_xy[0]], [sensor_xy[1]], marker="x", s=36, c="red", linewidths=1.5)
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_title("Filtered BEV Points + Object Goals")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")

    plot_bool_map(axes[0, 2], result["occupied_mask"], extent, "Occupied", sensor_xy)
    plot_bool_map(axes[0, 3], result["visible_free_mask"], extent, "Visible Free", sensor_xy)
    plot_bool_map(axes[1, 0], result["unknown_mask"], extent, "Unknown", sensor_xy)

    ax = axes[1, 1]
    ax.imshow(semantic_bev_image(result["occupied_mask"], result["visible_free_mask"], result["unknown_mask"]), origin="lower", extent=extent)
    ax.scatter([sensor_xy[0]], [sensor_xy[1]], marker="x", s=36, c="yellow", linewidths=1.5)
    ax.set_title("Known / Unknown Semantics")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")

    plot_scalar_map(
        axes[1, 2],
        esdf,
        extent,
        "ESDF (m)",
        sensor_xy,
        cmap="coolwarm",
        vmin=-esdf_scale,
        vmax=esdf_scale,
    )
    plot_scalar_map(
        axes[1, 3],
        clearance,
        extent,
        "Clearance (m)",
        sensor_xy,
        cmap="coolwarm",
        vmin=-clearance_scale,
        vmax=clearance_scale,
    )

    # Render the Matplotlib figure into an RGB NumPy array.
    fig.canvas.draw()
    image_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    return image_rgb

def build_points_input(points_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points_input = points_map.astype(np.float32, copy=True)
    invalid = ~mask.astype(bool)
    points_input[invalid] = np.nan
    return points_input


def output_path_for_sample(pointcloud_path: Path, output_dir: Path) -> Path:
    clip_id = pointcloud_path.parents[1].name
    return output_dir / clip_id / f"{pointcloud_path.stem}_esdf_debug.png"


def process_sample(
    pointcloud_path: Path,
    args: argparse.Namespace,
    ground_normal_prior: np.ndarray | None = None,
) -> tuple[Path, np.ndarray | None]:
    with np.load(pointcloud_path) as sample:
        points_map = sample["points_map"]
        mask = sample["mask"]
        fallback_rgb = sample["rgb"]
        full_image_size = sample["full_image_size"] if "full_image_size" in sample else None

    rgb = load_rgb_for_sample(pointcloud_path, fallback_rgb)
    metadata = load_sample_metadata(pointcloud_path)
    points_input = build_points_input(points_map, mask)
    rotation, translation = frame_transform(args.frame_preset, args.camera_height)

    result = pointcloud_to_esdf_pipeline(
        points_input,
        h_min=args.h_min,
        h_max=args.h_max,
        R=rotation,
        t=translation,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        resolution=args.resolution,
        sensor_xy=(args.sensor_x, args.sensor_y),
        robot_radius=args.robot_radius,
        ground_alignment=not args.disable_ground_alignment,
        ground_alignment_prior_normal=ground_normal_prior,
        ground_alignment_smoothing=args.ground_smoothing,
        ground_candidate_min_forward=args.ground_candidate_min_forward,
        ground_candidate_z_min=args.ground_candidate_z_min,
        ground_candidate_z_max=args.ground_candidate_z_max,
        ground_candidate_cell_size=args.ground_candidate_cell_size,
        ground_ransac_iterations=args.ground_ransac_iterations,
        ground_ransac_distance_threshold=args.ground_ransac_distance_threshold,
        ground_max_tilt_deg=args.ground_max_tilt_deg,
        ground_max_correction_deg=args.ground_max_correction_deg,
    )

    output_path = output_path_for_sample(pointcloud_path, args.output_dir)
    save_debug_figure(pointcloud_path, rgb, metadata, full_image_size, result, args, output_path)
    next_ground_normal = None
    ground_alignment = result.get("ground_alignment")
    if isinstance(ground_alignment, dict):
        smoothed_normal = ground_alignment.get("smoothed_ground_normal_xyz")
        if isinstance(smoothed_normal, list) and len(smoothed_normal) == 3:
            next_ground_normal = np.asarray(smoothed_normal, dtype=np.float32)
    return output_path, next_ground_normal


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    files = discover_pointcloud_files(
        input_root=args.input_root,
        sample=args.sample,
        clip_id=args.clip_id,
        limit=args.limit,
        stride=args.stride,
    )
    if not files:
        print("No pointcloud .npz files found for the requested selection.", file=sys.stderr)
        return 1

    print(f"Processing {len(files)} pointcloud file(s)...")
    ground_normal_prior = None
    active_clip_id = None
    for index, pointcloud_path in enumerate(files, start=1):
        clip_id = pointcloud_path.parents[1].name
        if clip_id != active_clip_id:
            active_clip_id = clip_id
            ground_normal_prior = None
        output_path, ground_normal_prior = process_sample(
            pointcloud_path,
            args,
            ground_normal_prior=ground_normal_prior,
        )
        print(f"[{index}/{len(files)}] {pointcloud_path} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
