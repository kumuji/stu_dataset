import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from utils.common import load_point_cloud, save_labels


def process_frame(
    points_path: Path, pred_mask: Path, output_dir: Path, eps: float, min_samples: int
):
    points = load_point_cloud(points_path)
    instances = np.zeros(len(points), dtype=np.int32)
    semantics = np.zeros(len(points), dtype=np.int32)
    if np.sum(pred_mask) > 0:
        clusters = (
            DBSCAN(eps=eps, min_samples=min_samples).fit(points[pred_mask]).labels_
        )
        instances[pred_mask] = clusters + 1  # Cluster IDs start at 1
        semantics[pred_mask] = 1  # single class, that we are evaluating

    output_path = output_dir / f"{points_path.stem}.label"
    save_labels((instances, semantics), output_path)


def main():
    """
    Since not all of the methods directly predict instances,
    this script generates instances using a threshold and a DBSCAN clustering algorithm.
    """
    parser = argparse.ArgumentParser(description="Generate DBSCAN instance predictions")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing sequences",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory with prediction .npy files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for instance files",
    )
    parser.add_argument(
        "--eps", type=float, default=1.0, help="DBSCAN epsilon parameter"
    )
    parser.add_argument(
        "--min-samples", type=int, default=-1, help="DBSCAN minimum samples parameter"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threhsold for semantic prediction"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process all sequences
    for seq_path in args.data_dir.glob("1[0-9][0-9]"):
        if seq_path.is_dir():
            seq_output = args.output_dir / seq_path.name
            seq_output.mkdir(exist_ok=True)

            pred_files = sorted((args.pred_dir / seq_path.name).glob("*.txt"))
            for pred_file in pred_files:
                points_file = seq_path / "velodyne" / f"{pred_file.stem}.bin"
                pred_mask = np.loadtxt(pred_file).astype(np.float32)

                # since we are predicting a single class, we can use a simple threshold
                pred_mask = (pred_mask > args.threshold).astype(np.int)

                process_frame(points_file, pred_mask, seq_output, args.eps, args.min_samples)


if __name__ == "__main__":
    main()
