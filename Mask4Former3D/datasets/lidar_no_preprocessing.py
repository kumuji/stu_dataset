from pathlib import Path
from random import choice, random, uniform
from typing import List, Optional, Union

import numpy as np
import volumentations as V
import yaml
from file_packer import FilePackReader
from torch.utils.data import Dataset


class LidarDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        config_path: str,
        instance_data_fpack: Optional[str] = None,
        mode: Optional[str] = "train",
        add_distance: Optional[bool] = False,
        ignore_label: Optional[Union[int, List[int]]] = 255,
        volume_augmentations_path: Optional[str] = None,
        instance_population: Optional[int] = 0,
        sweep: Optional[int] = 1,
    ):
        self.mode = mode
        self.data_dir = Path(data_dir)
        self.ignore_label = ignore_label
        self.add_distance = add_distance
        self.instance_population = instance_population
        self.sweep = sweep
        self.config = self._load_yaml(config_path)
        self.label_info = self._select_correct_labels(self.config["learning_ignore"])

        # Load scenes directly from directory
        self.scenes = self._load_scenes()
        # And for instance population, I used Ali's filepacker, but it is not necessery
        if self.instance_population > 0:
            self.instance_data = FilePackReader(instance_data_fpack)

        # Preload all poses for scenes
        self.pose_cache = self._preload_poses()

        # Augmentations
        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path is not None:
            self.volume_augmentations = V.load(
                volume_augmentations_path, data_format="yaml"
            )

    def _preload_poses(self):
        """Load and cache all pose transformations for each frame in each scene."""
        poses = list()
        for scene in self.scenes:
            # print(scene[0])
            scene_path = Path(scene[0]).parent.parent
            calib_file = scene_path / "calib.txt"
            pose_file = scene_path / "poses.txt"
            calibration = self.parse_calibration(calib_file)
            # Load poses for each frame in the scene
            poses.append(self.parse_poses(pose_file, calibration))
        return poses

    def _load_scenes(self):
        """Load all scenes and their respective frames based on mode."""
        scene_data = []
        scene_ids = self.config["split"].get(self.mode, [])
        if self.mode != "train":
            print(f"{self.mode}: {scene_ids}, {len(scene_ids)}")
        for scene_id in scene_ids:
            scene_path = self.data_dir / f"{int(scene_id):02}"
            print(Path(scene_path / "velodyne"))
            frames = list(Path(scene_path / "velodyne").glob("*.bin"))
            frames = sorted(frames)
            scene_data.append([str(frame) for frame in frames])
        return scene_data

    def __len__(self):
        return sum(len(scene) for scene in self.scenes)

    def __getitem__(self, idx: int):
        # Locate scene and frame
        scene_idx, frame_idx = self._find_scene_frame(idx)
        frame_path = Path(self.scenes[scene_idx][frame_idx])

        # Load point cloud and apply transformation
        points = np.fromfile(frame_path, dtype=np.float32).reshape(-1, 4)
        coordinates = points[:, :3]
        features = points[:, 3:4]
        time_array = np.zeros((features.shape[0], 1))
        features = np.hstack((time_array, features))

        # Apply pose transformation
        pose = self.pose_cache[scene_idx][frame_idx]
        coordinates = coordinates @ pose[:3, :3] + pose[3, :3]
        acc_num_points = [0, len(coordinates)]

        # Get labels
        labels = self._load_labels(frame_path)

        # Add instance population if required
        if "train" in self.mode and self.instance_population > 0:
            max_instance_id = np.amax(labels[:, 1])
            pc_center = coordinates.mean(axis=0)
            instance_coords, instance_feats, instance_labels = self.populate_instances(
                max_instance_id, pc_center, num_instances=self.instance_population
            )
            coordinates = np.vstack((coordinates, instance_coords))
            features = np.vstack((features, instance_feats))
            labels = np.vstack((labels, instance_labels))

        # Add distance if required
        if self.add_distance:
            center_coordinate = coordinates.mean(0)
            features = np.hstack(
                (
                    features,
                    np.linalg.norm(coordinates - center_coordinate, axis=1)[
                        :, np.newaxis
                    ],
                )
            )

        # Apply augmentations
        if "train" in self.mode:
            coordinates -= coordinates.mean(0)
            if 0.5 > random():
                coordinates += (
                    np.random.uniform(coordinates.min(0), coordinates.max(0)) / 2
                )
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]

        labels[:, 0] = np.vectorize(self.label_info.__getitem__)(labels[:, 0])

        return {
            "num_points": acc_num_points,
            "coordinates": coordinates,
            "features": np.hstack((coordinates, features)),
            "labels": labels,
            "sequence": (str(frame_path.parent.parent.name), str(frame_path.stem)),
        }

    def _find_scene_frame(self, idx):
        """Determine which scene and frame corresponds to a dataset index."""
        cumulative = 0
        for scene_idx, frames in enumerate(self.scenes):
            if idx < cumulative + len(frames):
                return scene_idx, idx - cumulative
            cumulative += len(frames)
        raise IndexError("Index out of range")

    def _load_labels(self, frame_path):
        """Load and process the labels for a given frame."""
        label_path = (
            str(frame_path).replace("velodyne", "labels").replace(".bin", ".label")
        )
        panoptic_label = np.fromfile(label_path, dtype=np.uint32)
        semantic_label = panoptic_label & 0xFFFF
        semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(
            semantic_label
        )
        labels = np.hstack((semantic_label[:, None], panoptic_label[:, None]))
        return labels

    def populate_instances(self, max_instance_id, pc_center, num_instances):
        coordinates_list = []
        features_list = []
        labels_list = []

        # Get all instance directories (assuming the root directory in the pack is named "instances")
        instance_dirs = self.instance_data.listdir(self.instance_data.base_path)
        for _ in range(num_instances):
            # Randomly select an instance directory
            instance_dir = choice(instance_dirs)
            semantic_label = int(instance_dir.split("_")[1])  # Extract semantic label
            # List all bin files within the chosen instance directory
            bin_files = sorted(
                self.instance_data.listdir(
                    f"{self.instance_data.base_path}/{instance_dir}"
                )
            )
            # Choose a random starting index for the sequence of sweeps
            idx = np.random.randint(len(bin_files))
            instance_list = []
            for time in range(self.sweep):
                if idx < len(bin_files):
                    instance_filepath = f"{self.instance_data.base_path}/{instance_dir}/{bin_files[idx]}"
                    # Read the binary file directly from the file pack
                    with self.instance_data.open(instance_filepath, mode="rb") as file:
                        instance = np.frombuffer(file.read(), dtype=np.float32).reshape(
                            -1, 4
                        )
                    # Add a time dimension to the instance points
                    time_array = np.ones((instance.shape[0], 1)) * time
                    instance = np.hstack(
                        (instance[:, :3], time_array, instance[:, 3:4])
                    )
                    instance_list.append(instance)
                    # Increment index for the next timestep in the sweep
                    idx = idx + 1

            # Aggregate instances from the list into a single array
            instances = np.vstack(instance_list)
            # Center the coordinates and apply translation
            coordinates = instances[:, :3] - instances[:, :3].mean(0)
            coordinates += pc_center + np.array(
                [uniform(-10, 10), uniform(-10, 10), uniform(-1, 1)]
            )
            # Extract features
            features = instances[:, 3:]
            # Create labels with semantic and instance IDs
            labels = np.zeros_like(features, dtype=np.int64)
            labels[:, 0] = semantic_label
            max_instance_id += 1
            labels[:, 1] = max_instance_id
            # Apply augmentations if defined
            aug = self.volume_augmentations(points=coordinates)
            coordinates = aug["points"]

            # Append to output lists
            coordinates_list.append(coordinates)
            features_list.append(features)
            labels_list.append(labels)
        return (
            np.vstack(coordinates_list),
            np.vstack(features_list),
            np.vstack(labels_list),
        )

    @staticmethod
    def parse_calibration(filename):
        calib = {}
        with open(filename) as calib_file:
            for line in calib_file:
                key, content = line.strip().split(":")
                values = [float(v) for v in content.strip().split()]
                pose = np.eye(4)
                pose[:3, :4] = np.array(values).reshape(3, 4)
                calib[key] = pose
        return calib

    @staticmethod
    def parse_poses(filename, calibration):
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        poses = list()
        with open(filename) as file:
            for line in file:
                values = [float(v) for v in line.strip().split()]
                pose = np.eye(4)
                pose[:3, :4] = np.array(values).reshape(3, 4)
                pose = Tr_inv @ pose @ Tr
                poses.append(pose)
        return poses

    def _select_correct_labels(self, learning_ignore):
        count = 0
        label_info = dict()
        for k, v in learning_ignore.items():
            if v:
                label_info[k] = self.ignore_label
            else:
                label_info[k] = count
                count += 1
        return label_info

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            return yaml.safe_load(f)

    def _remap_model_output(self, output):
        inv_map = {v: k for k, v in self.label_info.items()}
        output = np.vectorize(inv_map.__getitem__)(output)
        return output
