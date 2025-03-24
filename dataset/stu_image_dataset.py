from pathlib import Path
import torch

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import R
from torch.utils.data import Dataset


# Normalize intensities
def normalize_intensities(intensities):
    return np.clip(intensities / 255, 0, 1)


# Filter points that are inside a given polygon
def filter_points_in_polygon(image_points, polygon, corresponding_3d_points):
    path = Path(polygon)
    inside = path.contains_points(image_points)
    return image_points[inside], corresponding_3d_points[inside], inside


# Save colored point cloud as a PCD file
def save_colored_pcd(points, colors, output_pcd_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_pcd_file, pcd)
    print(f"Saved colored PCD at: {output_pcd_file}")


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape(
        (-1, 4)
    )  # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
    points = scan[:, :3]  # Extracting the (x, y, z) coordinates
    intensities = scan[:, 3:]  # Extracting the (x, y, z) coordinates
    return (points, intensities)


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).astype(np.int32)
    semantic_label = labels & 0xFFFF
    instance_label = labels >> 16

    return semantic_label, instance_label


def project_points_pinhole(points, camera_matrix, dist_coeffs):
    if points.size == 0:
        return (np.array([]), np.array([]))
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    image_points, _ = cv2.projectPoints(
        points.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    image_points = image_points.reshape(-1, 2)
    return image_points


def transform_points(points, T):
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = (T @ points_hom.T).T[:, :3]
    return points_transformed


def get_image_labels(
    base_path,
    idx,
    image_width,
    image_height,
    camera_matrix,
    dist_coeffs,
    translation,
    yaw,
    pitch,
    roll,
):
    label_file = base_path / f"labels/{idx:06d}.label"
    point_file = base_path / f"velodyne/{idx:06d}.bin"

    label_file = Path(label_file)
    labels, _ = load_labels(label_file)
    points, _ = load_point_cloud(point_file)

    labels[(labels != 2) & (labels != 0)] = 1

    # Construct the transformation matrix
    r = R.from_euler("ZYX", [yaw, pitch, roll])
    rotation_matrix = r.as_matrix()
    transformation_inv = np.eye(4)
    transformation_inv[:3, :3] = rotation_matrix.T
    transformation_inv[:3, 3] = -np.dot(rotation_matrix.T, translation)

    # Transform points into the camera frame
    points_transformed = transform_points(points, transformation_inv)

    # Use only points in front of the camera (z > 0)
    valid_indices = points_transformed[:, 2] > 0
    points_camera_valid = points_transformed[valid_indices]
    valid_labels = labels[valid_indices]

    # Project to image plane
    image_points = project_points_pinhole(
        points_camera_valid, camera_matrix, dist_coeffs
    )

    # Convert projected points to integer coordinates
    pts_int = image_points.astype(int)

    # Filter points that lie within image bounds
    inside_mask = (
        (pts_int[:, 0] >= 0)
        & (pts_int[:, 0] < image_width)
        & (pts_int[:, 1] >= 0)
        & (pts_int[:, 1] < image_height)
    )
    pts_in = pts_int[inside_mask]
    valid_labels = valid_labels[inside_mask]

    # return pts_in, valid_labels, image
    return np.hstack((pts_in, valid_labels[:, None]))


class STUDataset(Dataset):
    def __init__(self, base_path, offset=0, transform=None):
        self.base_path = Path(base_path)
        self.transform = transform
        self.offset = offset
        self.data = sorted(list(self.base_path.glob("*/velodyne/*.bin")))

        self.image_width = 1920
        self.image_height = 1208
        self.camera_matrix = np.array(
            [
                [1827.48989, 0.0, 925.91346],
                [0.0, 1835.88358, 642.07154],
                [0.0, 0.0, 1.0],
            ]
        )
        self.dist_coeffs = np.array([-0.260735, 0.046071, 0.001173, -0.000154, 0.0])
        self.translation = np.array([0.7658, 0.0124, -0.3925])
        self.yaw = -1.5599
        self.pitch = 0.0188
        self.roll = -1.5563

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 2 - anomaly
        # 0 - ignore
        # 1 - inlier
        base_path = self.data[idx].parent.parent
        idx = int(self.data[idx].stem)
        image_file = str(base_path / "port_a_cam_0" / (f"{idx:06d}" + ".png"))
        image = Image.open(image_file)  # BGR format by default

        # Apply optional transformation
        if self.transform:
            image = self.transform(image)

        label = get_image_labels(
            base_path,
            idx,
            image.shape[2],
            image.shape[1],
            self.camera_matrix,
            self.dist_coeffs,
            self.translation,
            self.yaw,
            self.pitch,
            self.roll,
        )

        return image, label, image_file

    def get_predictions_targets(self, uncertainty, target):
        coords = target.squeeze(0)

        # Separate indices
        x_indices = coords[:, 0]  # Width (columns)
        y_indices = coords[:, 1]  # Height (rows)
        labels_1 = coords[
            :, 2
        ]  # Label (0 = negative, 1 = positive, 2 = ignored)

        x_indices = torch.clamp(x_indices, 0, self.image_width - 1)
        y_indices = torch.clamp(y_indices, 0, self.image_height - 1)

        # Sample pixel values
        sampled_values = uncertainty[y_indices, x_indices]

        # Separate into categories
        uncertainty = sampled_values[labels_1 != 0]  # Pixels with label 0
        labels = labels_1[labels_1 != 0] - 1
        return uncertainty, labels
