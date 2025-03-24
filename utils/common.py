import numpy as np


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape(
        (-1, 4)
    )  # The point cloud data is stored in a Nx4 format (x, y, z, intensity)
    points = scan[:, :3]  # Extracting the (x, y, z) coordinates
    intensities = scan[:, 3:]  # Extracting the (x, y, z) coordinates
    return points, intensities


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).astype(np.int32)
    semantic_label = labels & 0xFFFF
    instance_label = labels >> 16

    return semantic_label, instance_label

def save_labels(labels, save_path):
    # Ensure input arrays have compatible shapes
    instance_label, semantic_label = labels
    assert instance_label.shape == semantic_label.shape, "Instance and semantic labels must have the same shape"

    # Convert to 32-bit unsigned integers for bitwise operations
    instance_upper = (instance_label.astype(np.uint32) & 0xFFFF) << 16  # Upper 16 bits
    semantic_lower = semantic_label.astype(np.uint32) & 0xFFFF           # Lower 16 bits

    # Combine into final 32-bit labels
    combined_labels = instance_upper | semantic_lower

    # Save to binary file
    combined_labels.tofile(save_path)

def convert_to_builtin_types(obj):
    if isinstance(obj, (np.integer, np.int64)):  # Handle NumPy integer types
        return int(obj)
    elif isinstance(
        obj, (np.floating, np.float32, np.float64)
    ):  # Handle NumPy float types
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):  # Handle NumPy arrays
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

