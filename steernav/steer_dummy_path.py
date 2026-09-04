from __future__ import annotations
import time
import argparse
import warnings
from argparse import Namespace
import cv2
from dataclasses import dataclass, fields
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from PIL import Image

import torch
from torch import Tensor
from transformers import AutoProcessor, AutoModelForCausalLM

import matplotlib
matplotlib.use("TkAgg")
from scipy.ndimage import zoom, gaussian_filter
from scipy.spatial import cKDTree
from moge.model.v2 import MoGeModel
import supervision as sv

from custom_utils.esdf_utils import visualize_static_dynamic_paths
from custom_utils.stream_handler import FrameStatus, InputStreamHandler
from custom_utils.io_utils import save_depth_video_mp4
from custom_utils.io_utils import load_calibration, filter_unwanted_results
from custom_utils.pointcloud_utils import camera_to_base_transform, pointcloud_to_esdf_pipeline, update_points


@dataclass
class BoxRecord:
    x1: float
    y1: float
    x2: float
    y2: float
    centroid: float
    area: float
    median_depth: float

@dataclass
class ESDFVisualMesh:
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    esdf: NDArray[np.float32]
    z_height: NDArray[np.float32]
    z_color: NDArray[np.float32]
    color_scale: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cell_colors: NDArray[np.uint8]
    # trying data validation in dataclass
    def __post_init__(self) -> None:
        # if np.isnan(self.x).any() or np.isnan(self.y).any():
        #     raise ValueError("Arrays 'x' and 'y' must not contain NaN values.")

        if self.x.shape != self.y.shape:
            raise ValueError(f"Shape mismatch: x {self.x.shape} vs y {self.y.shape}")

        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})")


@dataclass
class ValidESDFMesh(ESDFVisualMesh):
    valid: NDArray[np.bool]
    rows: int
    cols: int
    nearest_rows: NDArray[np.int32] | None
    nearest_cols: NDArray[np.int32] | None

    def __post_init__(self) -> None:
        # Run parent checks (x, y, x_min, x_max validation)
        super().__post_init__()

        # Run child specific checks
        if np.isinf(self.valid).any():
            raise ValueError("Array 'valid' must not contain inf values.")

    @classmethod
    def from_base_mesh(
            cls,
            base_mesh: ESDFVisualMesh,
            valid: NDArray[np.bool],
            rows: int,
            cols: int,
            nearest_rows: NDArray[np.int32] | None = None,
            nearest_cols: NDArray[np.int32] | None = None,
    ) -> "ValidESDFMesh":
        # Extract attributes from base_mesh and instantiate
        base_kwargs = {f.name: getattr(base_mesh, f.name) for f in fields(ESDFVisualMesh)}
        return cls(
            **base_kwargs,
            valid=valid,
            rows=rows,
            cols=cols,
            nearest_rows=nearest_rows,
            nearest_cols=nearest_cols,
        )

def build_esdf_mesh(esdf_raw, intrinsics, args: argparse.Namespace, free_space_scaling_factor=1.0) -> ESDFVisualMesh:
    x_min, x_max = args.x_min, args.x_max
    y_min, y_max = args.y_min, args.y_max
    esdf_raw = np.asarray(esdf_raw, dtype=np.float32)
    finite_raw = esdf_raw[np.isfinite(esdf_raw)]
    fallback = 0.0 if finite_raw.size == 0 else float(np.median(finite_raw))
    raw_abs_scale = max(float(args.esdf_height_clip_m), finite_abs_percentile(esdf_raw, 99.7))
    esdf = np.clip(
        np.nan_to_num(esdf_raw, nan=fallback, posinf=raw_abs_scale, neginf=-raw_abs_scale),
        -raw_abs_scale,
        raw_abs_scale,
    )
    smooth_sigma = float(args.esdf_projected_smooth_sigma)
    if smooth_sigma > 0.0:
        try:
            esdf = gaussian_filter(esdf, sigma=smooth_sigma)
        except Exception:
            pass
    color_scale = min(finite_abs_percentile(esdf, float(args.esdf_projected_color_percentile)), raw_abs_scale)
    signed = resize_esdf_for_surface(esdf, max_cols=420)
    rows = np.linspace(0.0, 1.0, signed.shape[0], dtype=np.float32)
    cols = np.linspace(0.0, 1.0, signed.shape[1], dtype=np.float32)
    cc, rr = np.meshgrid(cols, rows)
    base_x = x_min + cc * (x_max - x_min)
    base_y = y_min + rr * (y_max - y_min)
    fov_mask = esdf_camera_fov_mask(intrinsics, base_x, base_y)
    signed = np.nan_to_num(signed, nan=0.0, posinf=color_scale, neginf=-color_scale)
    z_color = np.where(fov_mask, np.clip(signed, -color_scale, color_scale), np.nan)
    # z_height = np.where(fov_mask, np.clip(-z_color, 0.0, color_scale) * float(args.esdf_height_scale), np.nan)
    z_height = np.where(fov_mask, np.clip(-z_color, -color_scale, color_scale) * float(args.esdf_height_scale), np.nan)
    z_height[z_height < 0] = z_height[z_height < 0] * free_space_scaling_factor
    mesh = ESDFVisualMesh(
        x=base_x,
        y=base_y,
        esdf=esdf,
        z_height=z_height,
        z_color=z_color,
        color_scale=color_scale,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        cell_colors=(0, 0, 0),
    )
    return mesh

def finite_abs_percentile(values: np.ndarray, percentile: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(np.abs(finite), percentile)), 1e-3)

def resize_esdf_for_surface(esdf: np.ndarray, max_cols: int) -> np.ndarray:
    # example, resize # (1000, 1250) ->  (416, 520)
    arr = np.asarray(esdf, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    target_cols = min(int(max_cols), arr.shape[1])
    target_rows = max(64, int(round(target_cols * arr.shape[0] / max(1, arr.shape[1]))))
    if target_rows == arr.shape[0] and target_cols == arr.shape[1]:
        return arr
    try:
        zoom_factors = (target_rows / arr.shape[0], target_cols / arr.shape[1])
        return zoom(arr, zoom_factors, order=3).astype(np.float32, copy=False)
    except Exception:
        finite = arr[np.isfinite(arr)]
        fill = 0.0 if finite.size == 0 else float(np.median(finite))
        safe = np.nan_to_num(arr, nan=fill)
        min_v = float(np.min(safe))
        max_v = float(np.max(safe))
        span = max(1e-6, max_v - min_v)
        image = Image.fromarray(np.asarray((safe - min_v) / span * 65535.0, dtype=np.uint16))
        resized = image.resize((target_cols, target_rows), Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32) / 65535.0 * span + min_v


def esdf_camera_fov_mask(intrinsics: np.ndarray, base_x: np.ndarray, base_y: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        return np.isfinite(base_x) & np.isfinite(base_y) & (base_x > 0.15)
    fx = float(intrinsics[0, 0])
    cx = float(intrinsics[0, 2])
    if abs(fx) < 1e-6:
        return np.isfinite(base_x) & np.isfinite(base_y) & (base_x > 0.15)
    cam_x = -base_y
    cam_z = base_x
    u_norm = fx * (cam_x / np.maximum(cam_z, 1e-6)) + cx
    return np.isfinite(u_norm) & (cam_z > 0.15) & (u_norm >= 0.0) & (u_norm <= 1.0)

def esdf_validity_map(mesh: ESDFVisualMesh) -> ValidESDFMesh:
    valid = np.isfinite(mesh.z_height)
    if not np.any(valid):
        valid = np.ones_like(mesh.z_height, dtype=bool)
    rows, cols = mesh.z_height.shape
    try:
        from scipy.ndimage import distance_transform_edt
        _dist, nearest = distance_transform_edt(~valid, return_indices=True)
        nearest_found = True
        nearest_rows = nearest[0].astype(np.int32, copy=False)  # (rows, cols)
        nearest_cols = nearest[1].astype(np.int32, copy=False)  # (rows, cols)
    except Exception:
        warnings.warn("esdf_validity_map failed to find nearest via distance_transform_edt; using KDTree fallback.")
        nearest_found = False
        valid_rows = np.nonzero(valid)[0].astype(np.int32, copy=False)
        valid_cols = np.nonzero(valid)[1].astype(np.int32, copy=False)

        # Fallback: Query all grid points against valid coordinates using KDTree
        grid_r, grid_c = np.indices((rows, cols))
        grid_pts = np.column_stack([grid_r.ravel(), grid_c.ravel()])
        valid_pts = np.column_stack([valid_rows, valid_cols])

        tree = cKDTree(valid_pts)
        _, nearest_indices = tree.query(grid_pts)

        nearest_rows = valid_rows[nearest_indices].reshape(rows, cols)
        nearest_cols = valid_cols[nearest_indices].reshape(rows, cols)

        # Fill mesh.z_height using nearest coordinates for both success and fallback paths
    mesh.z_height = mesh.z_height[nearest_rows, nearest_cols]
    return ValidESDFMesh.from_base_mesh(
        base_mesh=mesh,
        valid=valid,
        rows=rows,
        cols=cols,
        nearest_rows=nearest_rows,
        nearest_cols=nearest_cols,)


def constrain_xy_to_esdf(xy: np.ndarray, valid_mesh: ValidESDFMesh) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float32).copy()
    rows, cols = valid_mesh.rows, valid_mesh.cols
    col_f = (pts[:, 0] - valid_mesh.x_min) / max(1e-6, valid_mesh.x_max - valid_mesh.x_min) * (cols - 1)
    row_f = (pts[:, 1] - valid_mesh.y_min) / max(1e-6, valid_mesh.y_max - valid_mesh.y_min) * (rows - 1)
    outside = (col_f < 0.0) | (col_f > cols - 1) | (row_f < 0.0) | (row_f > rows - 1)
    col = np.clip(np.round(col_f).astype(np.int32), 0, cols - 1)
    row = np.clip(np.round(row_f).astype(np.int32), 0, rows - 1)
    valid = valid_mesh.valid
    bad = outside | ~valid[row, col]
    if not np.any(bad):
        return pts
    bad_rows, bad_cols = row[bad].copy(), col[bad].copy()
    row[bad] = valid_mesh.nearest_rows[bad_rows, bad_cols]
    col[bad] = valid_mesh.nearest_cols[bad_rows, bad_cols]
    pts[bad, 0] = valid_mesh.x[row[bad], col[bad]]
    pts[bad, 1] = valid_mesh.y[row[bad], col[bad]]
    return pts

def sample_mesh_z_and_gradient(path_xy: np.ndarray, valid_mesh: ValidESDFMesh, esdf_height_scale:float) -> np.ndarray:
    x_grid, y_grid, z_grid = valid_mesh.x, valid_mesh.y, valid_mesh.z_height
    rows, cols = z_grid.shape
    x_min, x_max = float(np.nanmin(x_grid)), float(np.nanmax(x_grid))
    y_min, y_max = float(np.nanmin(y_grid)), float(np.nanmax(y_grid))
    # Grid cell spatial dimensions (world units per grid index)
    dx_cell = max(1e-6, x_max - x_min) / max(1, cols - 1)
    dy_cell = max(1e-6, y_max - y_min) / max(1, rows - 1)
    col_f = (path_xy[:, 0] - x_min) / max(1e-6, x_max - x_min) * (cols - 1)
    row_f = (path_xy[:, 1] - y_min) / max(1e-6, y_max - y_min) * (rows - 1)
    col_f = np.clip(col_f, 0.0, cols - 1.001)
    row_f = np.clip(row_f, 0.0, rows - 1.001)
    c0 = np.floor(col_f).astype(np.int32)
    r0 = np.floor(row_f).astype(np.int32)
    c1 = np.clip(c0 + 1, 0, cols - 1)
    r1 = np.clip(r0 + 1, 0, rows - 1)
    wc = col_f - c0
    wr = row_f - r0
    z00 = z_grid[r0, c0]
    z01 = z_grid[r0, c1]
    z10 = z_grid[r1, c0]
    z11 = z_grid[r1, c1]
    # 1. Bilinear Height Interpolation
    z = (1.0 - wr) * ((1.0 - wc) * z00 + wc * z01) + wr * ((1.0 - wc) * z10 + wc * z11)
    z = np.nan_to_num(z, nan=0.0)
    z_scaled = z + 0.045 * valid_mesh.color_scale * esdf_height_scale

    # 2. Analytical Spatial Gradients (dz/dx, dz/dy)
    dz_dwc = (1.0 - wr) * (z01 - z00) + wr * (z11 - z10)
    dz_dwr = (1.0 - wc) * (z10 - z00) + wc * (z11 - z01)

    dz_dx = dz_dwc / dx_cell
    dz_dy = dz_dwr / dy_cell
    grad_xy = np.column_stack((dz_dx, dz_dy))
    # unnormalized slope magnitude in $(x, y) world coordinates
    grad_xy = np.nan_to_num(grad_xy, nan=0.0)

    # 3. Unit Gradient Direction Vector
    norms = np.linalg.norm(grad_xy, axis=1, keepdims=True)
    # normalized 2D direction vectors pointing in the direction of steepest ascent, multiply by negative
    grad_dir = np.divide(grad_xy, norms, out=np.zeros_like(grad_xy), where=norms > 1e-6)
    return z_scaled, grad_xy, grad_dir

def adam_update_numpy(
        path_xy: np.ndarray,
        valid_mesh: ValidESDFMesh,
        esdf_height_scale: float,
        lr: float = 0.1,
        smooth_weight: float = 0.2,
        iterations: int = 5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        fix_beginning: bool = True,
        fix_end: bool = False,
) -> np.ndarray:
    """
    Optimizes path_xy using Adam optimization.
    Drastically accelerates convergence compared to standard gradient descent.
    """
    path = path_xy.copy()

    # Initialize Adam moment vectors (same shape as path)
    m = np.zeros_like(path)
    v = np.zeros_like(path)
    for t in range(1, iterations + 1):
        # 1. Evaluate analytical gradient at current path positions
        z_scaled, grad_xy, _ = sample_mesh_z_and_gradient(path, valid_mesh, esdf_height_scale)
        # 2. Compute smooth force gradient (derivative of Laplacian objective)
        smooth_force = np.zeros_like(path)
        smooth_force[1:-1] = path[:-2] - 2.0 * path[1:-1] + path[2:]

        # fixing the smooth force of the last element to be zero
        smooth_force[-1] = 0
        # 3. Total Loss Gradient: grad_xy pushes AWAY from cost (+), smooth_force pulls towards neighbors (-)
        # We want to minimize cost, so total gradient = grad_xy - smooth_weight * smooth_force
        grad_total = grad_xy - smooth_weight * smooth_force

        if fix_beginning:
            grad_total[0:1] = 0.0
        if fix_end:
            grad_total[-1] = 0.0

        # 4. Adam Moment Updates
        m = beta1 * m + (1.0 - beta1) * grad_total
        v = beta2 * v + (1.0 - beta2) * (grad_total ** 2)

        # Bias correction
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        # 5. Parameter Update
        path -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # path = constrain_xy_to_esdf(path, valid_mesh)

    # filter out waypoints where z-height is positive, indicating obstacle.
    z_scaled, grad_xy, _ = sample_mesh_z_and_gradient(path, valid_mesh, esdf_height_scale)
    z_mask = np.argwhere(z_scaled > 0)
    # print(f"original path: {path}")
    # print(f"z_scaled: {z_scaled}")
    if len(z_mask) > 0:
        last_valid_idx = z_mask[0][0] - 1
        if last_valid_idx == -1:
            path = np.zeros_like(path)
            print("all path below esdf criteria, passing zero paths")
            print(path)
        else:
            path[last_valid_idx + 1:] = path[last_valid_idx]
            # print(f"constraining path to idx {last_valid_idx} pt: {path[last_valid_idx]}")
    return path


def update_trajectories(args: Namespace,
                        points_input: ndarray,
                        intrinsics: ndarray,
                        input_trajectory: ndarray,
                        time_session: bool,
                        n_iter: int = 5,
                        lr: float = 0.1,
                        smooth_weight: float = 0.2,) -> tuple[
    dict[str, Tensor], ndarray, ndarray]:
    t0 = time.perf_counter()
    rotation, translation = camera_to_base_transform(camera_height=args.camera_height)
    args.frame_preset = "camera_to_base"
    esdf_result = pointcloud_to_esdf_pipeline(points_input, h_min=args.h_min, h_max=args.h_max,
                                              R=rotation, t=translation,
                                              x_min=args.x_min, x_max=args.x_max,
                                              y_min=args.y_min, y_max=args.y_max,
                                              )
    if time_session:
        t1 = time.perf_counter()
        print(f"pointcloud_to_esdf_pipeline {(t1 - t0) * 1000:.1f} ms")
    # ================= from VLA Data Generation pipeline =========================
    mesh = build_esdf_mesh(esdf_raw=esdf_result['esdf'], intrinsics=intrinsics,
                           args=args, free_space_scaling_factor=0.25)
    valid_mesh = esdf_validity_map(mesh)
    if time_session:
        t2 = time.perf_counter()
        print(f"build + valid_mesh {(t2 - t1) * 1000:.1f} ms")
    init_path_xy = constrain_xy_to_esdf(input_trajectory[:, :2], valid_mesh)  # (N, 2)
    # Optimize path using ESDF gradients
    # numpy version, n_iter=5, 1.3ms
    opt_path_xy = adam_update_numpy(path_xy=init_path_xy, valid_mesh=valid_mesh,
                                    esdf_height_scale=args.esdf_height_scale,
                                    lr=lr, smooth_weight=smooth_weight,
                                    iterations=n_iter, )
    opt_path_xy = constrain_xy_to_esdf(opt_path_xy, valid_mesh)
    if time_session:
        t3 = time.perf_counter()
        print(f"constrain path, adam_update_numpy {(t3 - t2) * 1000:.1f} ms")
        print(f"total time before rendering {(t3 - t0) * 1000:.1f} ms")
    return esdf_result, init_path_xy, opt_path_xy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pipeline to adjust a dummy path accoring to a depth-map ESDF."
    )
    parser.add_argument("--h-min", type=float, default=0.5, help="Minimum kept height in meters.")
    parser.add_argument("--h-max", type=float, default=1.5, help="Maximum kept height in meters.")
    parser.add_argument("--x-min", type=float, default=0.0, help="Minimum forward extent in meters.")
    parser.add_argument("--x-max", type=float, default=10.0, help="Maximum forward extent in meters.")
    parser.add_argument("--y-min", type=float, default=-5.0, help="Minimum lateral extent in meters.")
    parser.add_argument("--y-max", type=float, default=5.0, help="Maximum lateral extent in meters.")
    parser.add_argument("--resolution", type=float, default=0.10, help="Grid resolution in meters per cell.")
    parser.add_argument("--sensor-x", type=float, default=0.0, help="Sensor x location in map frame.")
    parser.add_argument("--sensor-y", type=float, default=0.0, help="Sensor y location in map frame.")
    parser.add_argument("--camera-height", type=float, default=1.0, help="AGL, in meters")
    parser.add_argument("--img_w", type=int, default=1280, help="resize img width to correctly overlay path")
    parser.add_argument("--img_h", type=int, default=720, help="resize img height to correctly overlay path")
    parser.add_argument("--esdf-height-scale", type=float, default=1.8)
    parser.add_argument("--esdf-height-clip-m", type=float, default=2.0)
    parser.add_argument("--esdf-projected-smooth-sigma", type=float, default=1.5)
    parser.add_argument("--esdf-projected-color-percentile", type=float, default=80.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Initialize predictor (single-GPU streaming)
    save_video_toggle = False
    time_session = False
    stream_type = "video"  # ["yarp", "video", "webcam"]
    # video_path = "/home/jim/Projects/steernav/assets/Cars_and_Gasstation.mp4"
    # video_path = "/home/jim/Projects/steernav/assets/jim_flownav_test.mp4"
    video_path = "/home/jim/Projects/steernav/assets/corridoor_omni_ft_2_left.mp4"
    camera_matrix_dir = "old_cam_matrix.json"
    # camera_matrix_dir = "ghost_fl_cam_matrix.json"
    output_folder = "demo_video"
    webcam_index = 0
    yarp_port = "/sam3/rgbImage:i"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    depth_model_name = "Ruicheng/moge-2-vitl-normal"
    vision_model_name = "microsoft/Florence-2-large"
    print("running", depth_model_name, vision_model_name)
    depth_model = MoGeModel.from_pretrained(depth_model_name).to(device).eval()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    obj_detect_model = AutoModelForCausalLM.from_pretrained(vision_model_name,
                                                            torch_dtype=torch_dtype,
                                                            attn_implementation="eager",
                                                            trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(vision_model_name, trust_remote_code=True)
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_prompt = "people"
    prompt = task_prompt + text_prompt
    tracker = sv.ByteTrack()

    n_pts = 20
    straight_path = np.stack((np.linspace(0, 15, n_pts + 1), np.linspace(0, 0, n_pts + 1))).T

    cam_matrix, dist_coeffs, T_base_from_cam = load_calibration(camera_matrix_dir)
    T_cam_from_base = np.linalg.inv(T_base_from_cam)

    # Initialize input source
    src = InputStreamHandler(
        kind=stream_type,
        video_path=video_path,
        webcam_index=webcam_index,
        skip_n_fr=10,
    )
    print(f"Opening source: {stream_type}")
    src.open()
    stream_buffer = src.read()

    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save
    detection_queue = []
    detection_queue_len = 20
    stop_processing = False
    # shrink image dimensions
    img_w, img_h = 640, 480
    window_name = "esdf_surface"
    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Set fixed width x height (e.g., 640x480 or 1280x720)
    cv2.resizeWindow(window_name, 1200, 900)

    prev_time = time.time()
    try:
        while stop_processing is not True:
            # Read frame (RGB)
            stream_buffer = src.read()
            if stream_buffer.status == FrameStatus.NO_FRAME:
                # YARP: no new frame yet; try again.
                continue
            if stream_buffer.status == FrameStatus.EOS:
                # End of stream for video/webcam or closed YARP port.
                break
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            original_frame = stream_buffer.frame
            # shrink frame for faster inference
            frame_rgb = cv2.resize(original_frame, dsize=(img_w, img_h), interpolation=cv2.INTER_CUBIC)
            pil_image = Image.fromarray(frame_rgb)
            input_image = torch.from_numpy(frame_rgb).to(device).permute(2, 0, 1).float().div_(255.0)
            """
                MOGE model inference:
                `output` contains the final prediction. Pass `return_per_step=True` to also return every refinement step.
                All maps have the same height and width as the input image.
                {
                  "points": (H, W, 3),                  # final metric point map in OpenCV camera coordinates (x right, y down, z forward)
                  "depth": (H, W),                      # final metric depth map
                  "intrinsics": (3, 3),                 # normalized camera intrinsics for the final prediction
                  "mask": (H, W),                       # binary mask for valid pixels
                  "normal": (H, W, 3),                 # normal map in OpenCV camera coordinates (optional)
                }
                With `return_per_step=True`, `points_per_step`, `depth_per_step`, and `intrinsics_per_step`
                contain `refine_steps + 1` entries, including the initial prediction.
            """
            t0 = time.perf_counter()
            depth_model_output = depth_model.infer(input_image)
            moge_points = depth_model_output['points'].cpu().numpy()
            estimated_cam_matrix = depth_model_output['intrinsics'].cpu().numpy()
            points_input = moge_points.astype(np.float32, copy=True)
            points_input[~depth_model_output["mask"].cpu().numpy().astype(bool)] = np.nan
            depth = depth_model_output['depth'].cpu().numpy()

            t1 = time.perf_counter()
            print(" = = = = = = = = = ")
            print(f"{depth_model_name} inference took {(t1 - t0) * 1000:.1f} ms")

            obj_detect_inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, torch_dtype)
            generated_ids = obj_detect_model.generate(
                input_ids=obj_detect_inputs["input_ids"],
                pixel_values=obj_detect_inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            obj_detect_result = processor.post_process_generation(generated_text, task=task_prompt,
                                                              image_size=(pil_image.width, pil_image.height))
            bbox_result = obj_detect_result[task_prompt]
            bbox_result = filter_unwanted_results(bbox_result, img_w, img_h)
            bbox_only = [bbox for bbox, label in zip(bbox_result['bboxes'], bbox_result['labels'])]
            if len(bbox_only) > 0:
                dummy_confidence = np.ones(len(bbox_only)) * 0.7
                sv_detection = sv.Detections(xyxy=np.array(bbox_only), confidence=dummy_confidence)
                detections = tracker.update_with_detections(sv_detection)
                pos_dict = {}
                for box, id in zip(detections.xyxy, detections.tracker_id):
                    x1, y1, x2, y2 = box
                    box_3d_pts = points_input[int(y1):int(y2), int(x1):int(x2)] # (120, 40, 3)
                    if box_3d_pts.size == 0:
                        continue
                    pts_flat = box_3d_pts.reshape(-1, 3) # (N, 3)
                    valid_mask = ~np.isnan(pts_flat).any(axis=1) & (pts_flat != 0).any(axis=1) # (N,)
                    valid_pts = pts_flat[valid_mask] # (N, 3)
                    median_3d = np.median(valid_pts, axis=0)
                    pos_dict[id] = median_3d
                    print(f"detect id {id} median loc: {median_3d}")
                # coopting the data field since it is unused.
                detections.data = pos_dict
                # pred_color = plot_bbox(frame_rgb, bbox_result, detections.tracker_id, show_plot=False, return_img=True)
            else:
                # pred_color = frame_rgb
                detections = sv.Detections(xyxy=np.array([[0, 0, 0, 0]]), confidence=dummy_confidence, tracker_id=np.array([0]), data={})
            # detection queue
            detection_queue.append(detections)
            if len(detection_queue) > detection_queue_len:
                detection_queue.pop(0)
            
            t2 = time.perf_counter()
            print(f"{vision_model_name} inference took {(t2 - t1) * 1000:.1f} ms")
            # 19fps in video, skipping 10 fr, roughly 2fps
            updated_points = update_points(points_input, detection_queue,
                                           robot_velocity_camera=np.array([0, 0, 0.1]),
                                           time_incr=0.5, time_look_ahead=2.0)

            static_esdf_result, init_path_xy, static_path_xy = update_trajectories(
                args, points_input, estimated_cam_matrix, straight_path, time_session)

            dynamic_esdf_result, init_path_xy, dynamic_path_xy = update_trajectories(
                args, updated_points, estimated_cam_matrix, straight_path, time_session)

            # add estimated intrinsics
            # height, width = original_frame.shape[:2]
            # get output from depth model output
            # estimated_cam_matrix[0][0] *= width
            # estimated_cam_matrix[0][2] *= width
            # estimated_cam_matrix[1][1] *= height
            # estimated_cam_matrix[1][2] *= height
            # print(f"cam matrix from file:\n{cam_matrix}")
            # print(f"estimated_cam_matrix:\n{estimated_cam_matrix}")
            # 5. Render baseline and optimized paths together

            t0 = time.perf_counter()
            esdf_surface = visualize_static_dynamic_paths(depth=depth, rgb=frame_rgb,
                                                          dynamic_esdf_result=dynamic_esdf_result,
                                                          static_esdf_result=static_esdf_result,
                                                          bbox_result=bbox_result,
                                                          cam_matrix=cam_matrix,
                                                          T_cam_from_base=T_cam_from_base,
                                                          before_path=init_path_xy,
                                                          static_path=static_path_xy,
                                                          dynamic_path=dynamic_path_xy,
                                                          idx=frame_idx, args=args)
            # esdf_surface = debug_visualize(depth=depth, rgb=original_frame,
            #                                    result=esdf_result, cam_matrix=cam_matrix,
            #                                    T_cam_from_base=T_cam_from_base,
            #                                    before_path=init_path_xy, after_path=opt_path_xy,
            #                                    idx=0, args=args)
            t1 = time.perf_counter()
            print(f"visualize_path {(t1 - t0) * 1000:.1f} ms")
            # Display FPS
            cv2.putText(esdf_surface,f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1,(0, 255, 0),2)
            cv2.imshow(
                window_name, cv2.cvtColor(esdf_surface, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(1) # cv2.waitKey(1) if running in a real-time loop
            # Update running statistics
            frame_idx += 1
            frame_timestamps.append(time.time())
            if save_video_toggle:
                video_frames.append(cv2.cvtColor(esdf_surface, cv2.COLOR_RGB2BGR))
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
            peak_memory = max(peak_memory, current_peak_memory)
            print(
                f"Processed frame {frame_idx}. "
                f"Current peak memory: {current_peak_memory:.2f} GB, "
                f"Overall peak memory: {peak_memory:.2f} GB.",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping processing gracefully...")

    finally:
        # Source cleanup
        src.close()

        if save_video_toggle:
            if len(frame_timestamps) >= 2:
                elapsed = frame_timestamps[-1] - frame_timestamps[0]
                # Use average FPS over the whole run
                effective_fps = (len(frame_timestamps) - 1) / elapsed
            else:
                effective_fps = 30.0

            output_dir = f"{output_folder}/{stream_type}.mp4"
            save_depth_video_mp4(
                video=np.array(video_frames),
                path=output_dir,
                fps=4,
                # fps=effective_fps,
            )
            print(
                f"\nSaved video to {output_dir} at {effective_fps:.2f} FPS."
            )

        # Close any OpenCV windows
        cv2.destroyAllWindows()

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")


if __name__ == "__main__":
    raise SystemExit(main())
