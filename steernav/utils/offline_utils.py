import cv2
import math
import json
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import Image
from typing import List, Tuple, Dict, Optional

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


def msg_to_pil(msg: Image) -> PILImage.Image:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = PILImage.fromarray(img)
    return pil_image


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes()
    ros_image.step = ros_image.width
    return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))  # crop to the right ratio
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


# clip angle between -pi and pi
def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


def plot_trajs_and_points(
        ax: plt.Axes,
        list_trajs: list,
        list_points: list,
        traj_colors: list = [CYAN, MAGENTA],
        point_colors: list = [RED, GREEN],
        traj_labels: Optional[list] = ["prediction", "ground truth"],
        point_labels: Optional[list] = ["robot", "goal"],
        traj_alphas: Optional[list] = None,
        point_alphas: Optional[list] = None,
        quiver_freq: int = 1,
        default_coloring: bool = True,
):
    assert len(list_trajs) <= len(traj_colors) or default_coloring, (
        "Not enough colors for trajectories"
    )
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
            traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), (
        "Not enough labels for points"
    )

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")


# my custom utils
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