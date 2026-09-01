import cv2
from tqdm import tqdm
import numpy as np
import math
import json
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training
BGR_color_dict = {  # BGR
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "CYAN": (255, 255, 0),
    "YELLOW": (0, 255, 255),
    "CUSTOM": (125, 125, 125),
}

RGB_color_dict = {  # RGB
    "RED": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "BLUE": (0, 0, 255),
    "CYAN": (0, 255, 255),
    "YELLOW": (255, 255, 0),
    "CUSTOM": (125, 125, 125),
}


def fig_to_image(fig: plt.Figure) -> Image.Image:
    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    image = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    plt.close(fig)
    return image

def save_side_by_side(original, pred_color, out_path, fps=20):
    """
    original: [T,H,W,3] float32
    pred_color: [T,H,W,3] uint8
    """
    h, w = original.shape[1], original.shape[2]
    combined_w = w * 2

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (combined_w, h)
    )

    for orig, pred in zip(original, pred_color):
        orig_uint8 = (orig * 255).astype(np.uint8)[..., ::-1]  # RGB → BGR
        combined = np.concatenate([orig_uint8, pred], axis=1)
        writer.write(combined)

    writer.release()


def colorize_pred(pred, vmin=None, vmax=None, add_colorbar=False):
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]

    single_frame = pred.ndim == 2

    if single_frame:
        pred = pred[None, ...]

    vmin = float(pred.min()) if vmin is None else float(vmin)
    vmax = float(pred.max()) if vmax is None else float(vmax)

    cmap = matplotlib.colormaps["Spectral"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    frames = []

    for frame in pred:
        rgb = cmap(norm(frame))[..., :3] * 255
        bgr = rgb.astype(np.uint8)[..., ::-1]
        frames.append(bgr)

    video = np.stack(frames)

    # Add colorbar
    if add_colorbar:
        fig, ax = plt.subplots(figsize=(1.0, 5))
        fig.subplots_adjust(left=0.2, right=0.7)

        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=ax)
        cbar.set_label("Depth")

        fig.canvas.draw()
        colorbar = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)

        colorbar = colorbar[..., ::-1]  # RGB → BGR

        colorbar = cv2.resize(
            colorbar,
            (colorbar.shape[1], video.shape[1])
        )

        video = np.concatenate(
            [video, np.broadcast_to(
                colorbar[None, ...],
                (video.shape[0], *colorbar.shape)
            )],
            axis=2
        )

    if single_frame:
        return video[0]

    return video

def load_video_as_numpy(path):
    """
    Loads Video
    -----------------
    Loads a video as a numpy array of type float 32 and in range 0 to 1

    Parameters
    ----------
    path : str
        Path to the video you want to load

    Returns
    -------
    result : np.ndarray
        Numpy array of shape [Time, Height, Width, Channels]
    """
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()
    return np.stack(frames)

def save_video_mp4(video: np.ndarray, path: str, fps=20):
    vmin = float(video.min())
    vmax = float(video.max())

    h, w = video.shape[1], video.shape[2]

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    # colormap = cm.get_cmap('Spectral')
    colormap = matplotlib.colormaps["Spectral"]

    for frame in video:
        # Normalize globally
        norm = (np.clip(frame, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)

        # Apply matplotlib colormap → RGBA in [0..1]
        colored = colormap(norm)[..., :3]   # drop alpha channel

        # Convert to uint8 + BGR for OpenCV
        colored_bgr = (colored * 255).astype(np.uint8)[..., ::-1]

        writer.write(colored_bgr)

    writer.release()

def save_depth_video_mp4(video: np.ndarray, path: str, fps=20):
    h, w = video.shape[1], video.shape[2]

    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for frame in tqdm(video):
        writer.write(frame)

    writer.release()

def overlay_path(trajectories: np.ndarray,
                 img: np.ndarray,
                 cam_matrix: np.ndarray,
                 T_cam_from_base: np.ndarray,
                 path_color=RGB_color_dict['GREEN'],
                 policy_color=RGB_color_dict['RED'],
                 ):
    if len(trajectories.shape) == 2:
        n_trajectories = 1
        trajectories = np.expand_dims(trajectories, 0)
    elif len(trajectories.shape) == 3:
        n_trajectories = trajectories.shape[0]
    else:
        print(f"error, unable to process trajectories dimension: {trajectories.shape}")
        return None

    # Points in base frame -> camera frame -> pixels
    R_cb = T_cam_from_base[:3, :3]
    t_cb = T_cam_from_base[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cb)
    overlay = img.copy()
    for i in range(n_trajectories):
        pts_3d = np.hstack([trajectories[i], np.zeros((trajectories[i].shape[0], 1))])  # z=0 in base frame
        img_pts, _ = cv2.projectPoints(pts_3d, rvec, t_cb, cam_matrix, None)
        img_pts = img_pts.reshape(-1, 2)

        # Keep points in front of camera and inside image
        pts_cam = (R_cb @ pts_3d.T + t_cb.reshape(3, 1)).T
        valid_z = pts_cam[:, 2] > 0
        h, w = img.shape[:2]
        valid_xy = (
                (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
                (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)
        )
        keep = valid_z & valid_xy
        if not keep.any():
            print(f"out of {pts_cam.shape} points, no points kept in front of camera...")
            continue

        pts_pix = img_pts[keep].astype(int)
        my_color = path_color
        if i == 0:
            my_color = policy_color
        if len(pts_pix) >= 2:
            cv2.polylines(overlay, [pts_pix], isClosed=False, color=my_color, thickness=2)
        else:
            for pt in pts_pix:
                cv2.circle(overlay, tuple(pt), radius=3, color=my_color, thickness=-1)
    return overlay


def load_calibration(json_path: str):
    """
    Builds:
      K (3x3), dist=None, T_cam_from_base (4x4)
    from tf.json with H_cam_bl: pitch(deg), x,y,z.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if data is None or "H_cam_bl" not in data:
        raise ValueError(f"Missing H_cam_bl in {json_path}")

    h = data["H_cam_bl"]
    roll = math.radians(float(h["roll"]))
    xt, yt, zt = float(h["x"]), float(h["y"]), float(h["z"])

    # Rotation about +y (camera pitched down is positive pitch if y up/right-handed)
    Ry = np.array([
        [0.0, math.sin(roll), math.cos(roll)],
        [-1.0, 0.0, 0.0],
        [0.0, -math.cos(roll), math.sin(roll)]
    ], dtype=np.float64)

    T_base_from_cam = np.eye(4, dtype=np.float64)
    T_base_from_cam[:3, :3] = Ry
    T_base_from_cam[:3, 3] = np.array([xt, yt, zt], dtype=np.float64)

    fx = data["Intrinsics"]["fx"]
    fy = data["Intrinsics"]["fy"]
    cx = data["Intrinsics"]["cx"]
    cy = data["Intrinsics"]["cy"]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    dist = None  # explicitly no distortion
    return K, dist, T_base_from_cam