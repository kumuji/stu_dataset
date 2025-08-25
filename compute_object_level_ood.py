import argparse
import json
from pathlib import Path

import numpy as np

from utils.common import load_labels, load_point_cloud, convert_to_builtin_types


class UQEvaluator:
    def __init__(
        self, min_points=5, num_classes=2, ignore_labels=[-1], instance_labels=[1]
    ):
        """
        We only evaluate a single instance class, that is 1 (anomaly), against 0 (inlier).
        Other classes are expected to be ignored, or could be ignored with ignore_labels argument
        """
        super(UQEvaluator, self).__init__()
        self.min_points = min_points
        self.num_classes = num_classes
        self.ignore_labels = ignore_labels
        self.instance_labels = instance_labels
        self.reset()

    def reset(self):
        self.pan_tp = np.zeros(self.num_classes, dtype=np.int64)
        self.pan_iou = np.zeros(self.num_classes, dtype=np.double)
        self.pan_fn = np.zeros(self.num_classes, dtype=np.int64)
        self.pan_fp = np.zeros(self.num_classes, dtype=np.int64)

    def addBatchUnknown(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
        # make sure instances are not zeros (it messes with my approach)
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        # only interested in points that are outside the void area (not in excluded classes)
        for cl in self.ignore_labels:
            # make a mask for this class
            gt_not_in_excl_mask = y_sem_row != cl
            # remove all other points
            x_sem_row = x_sem_row[gt_not_in_excl_mask]
            y_sem_row = y_sem_row[gt_not_in_excl_mask]
            x_inst_row = x_inst_row[gt_not_in_excl_mask]
            y_inst_row = y_inst_row[gt_not_in_excl_mask]

        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        # we only go over the inlier and outlier labels
        for cl in self.instance_labels:
            # print("CLASS", cl)
            # get a class mask
            x_inst_in_cl_mask = x_sem_row == cl
            y_inst_in_cl_mask = y_sem_row == cl

            # get instance points in class (makes outside stuff 0)
            x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            # print("x_inst_in_cl: ",np.sum(x_inst_in_cl_mask))
            # print("y_inst_in_cl: ",np.sum(y_inst_in_cl_mask))

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(
                x_inst_in_cl[x_inst_in_cl > 0], return_counts=True
            )
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])
            # print("Unique predictions:", unique_pred)

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(
                y_inst_in_cl[y_inst_in_cl > 0], return_counts=True
            )
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])
            # print("Unique ground truth:", unique_gt)
            # print("matched_gt: ",matched_gt)

            # generate intersection using offset
            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = (
                x_inst_in_cl[valid_combos] + 2**32 * y_inst_in_cl[valid_combos]
            )
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // 2**32
            pred_labels = unique_combo % 2**32
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float64) / unions.astype(np.float64)

            tp_indexes = ious > 0.5
            # print(ious[tp_indexes])
            self.pan_tp[cl] += np.sum(tp_indexes)
            self.pan_iou[cl] += np.sum(ious[tp_indexes])

            # print("pan_iou: ", self.pan_iou_uq)
            # print("TP: ",self.pan_tp[cl])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn[cl] += np.sum(
                np.logical_and(counts_gt >= self.min_points, matched_gt == False)
            )
            self.pan_fp[cl] += np.sum(
                np.logical_and(counts_pred >= self.min_points, matched_pred == False)
            )
            # print("FN: ",self.pan_fn[cl])

    def get_stats(self):
        # we are only interested in anomaly class
        return (
            self.pan_iou[1],
            self.pan_tp[1],
            self.pan_fp[1],
            self.pan_fn[1],
        )

    def getUQ(self):
        sq = self.pan_iou.astype(np.float64) / np.maximum(
            self.pan_tp.astype(np.float64), 1e-15
        )

        recallq = self.pan_tp.astype(np.float64) / np.maximum(
            self.pan_tp.astype(np.float64) + self.pan_fn.astype(np.float64), 1e-15
        )

        uq = sq * recallq
        return sq, recallq, uq

    def getPQ(self):
        sq = self.pan_iou.astype(np.float64) / np.maximum(
            self.pan_tp.astype(np.float64), 1e-15
        )

        rq = self.pan_tp.astype(np.float64) / np.maximum(
            self.pan_tp.astype(np.float64)
            + 0.5 * self.pan_fp.astype(np.float64)
            + 0.5 * self.pan_fn.astype(np.float64),
            1e-15,
        )

        pq = sq * rq
        return sq, rq, pq


class ObjectOODMetricsCalculator:
    min_eval_distance = 2.5
    max_eval_distance = 50
    min_num_points_to_eval = 5

    def __init__(self):
        self.evaluator = UQEvaluator(min_points=self.min_num_points_to_eval)

    def update(
        self,
        points,
        semantic_prediction,
        instance_prediction,
        semantic_target,
        instance_target,
    ):
        """Update the stored scores and labels with new data.

        Args:
            points (np.ndarray): Point cloud coordinates
            semantic_prediction (np.ndarray): Semantic prediction
            instance_prediction (np.ndarray): Instance prediction
            semantic_target (np.ndarray): Ground truth labels
            instance_target (np.ndarray): Ground truth labels
        """

        if len(semantic_prediction) != len(instance_prediction):
            raise ValueError("Semantic and Instance prediiction count mismatch")
        if len(semantic_target) != len(instance_target):
            raise ValueError("Semantic and Instance prediiction count mismatch")

        distances = np.linalg.norm(points, axis=1)
        # Process labels and apply distance mask
        inlier_labels = np.where(semantic_target != 0, 0, -1)
        processed_labels = np.where(semantic_target == 2, 1, inlier_labels)
        processed_labels = np.where(
            (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
            -1,
            processed_labels,
        )
        ignore_mask = processed_labels != -1
        semantic_labels = processed_labels[ignore_mask]
        # Only evaluate if sufficient anomaly points
        if np.sum(semantic_labels) >= self.min_num_points_to_eval:
            if len(semantic_prediction) != len(semantic_target):
                raise ValueError("Prediction and label count mismatch")

            self.evaluator.addBatchUnknown(
                semantic_prediction[ignore_mask],
                instance_prediction[ignore_mask],
                semantic_labels,
                instance_target[ignore_mask],
            )

    def compute_metrics(self):
        """Compute OOD detection metrics on accumulated data."""

        metrics = dict()
        sq, rq, uq = self.evaluator.getUQ()
        metrics.update({"SQ": sq[1] * 100, "RecallQ": rq[1] * 100, "UQ": uq[1] * 100})

        sq, rq, pq = self.evaluator.getPQ()
        metrics.update({"RQ": rq[1] * 100, "PQ": pq[1] * 100})
        _, tp, fp, fn = self.evaluator.get_stats()
        metrics["TP"] = tp
        metrics["FP"] = fp
        metrics["FN"] = fn

        return metrics


def main(args):
    metrics_calculator = ObjectOODMetricsCalculator()

    for seq_path in args.data_dir.glob("1[0-9][0-9]"):
        if seq_path.is_dir():
            pred_files = sorted((args.data_dir / seq_path.name).glob("*.label"))

            for pred_file in pred_files:
                prediction_semantic, prediction_instance = load_labels(pred_file)

                label_file = seq_path / "labels" / f"{pred_file.stem}.label"
                gt_semantic, gt_instance = load_labels(label_file)

                pcd_file = seq_path / "velodyne" / f"{pred_file.stem}.bin"
                points, _ = load_point_cloud(pcd_file)

                metrics_calculator.update(
                    points,
                    prediction_semantic,
                    prediction_instance,
                    gt_semantic,
                    gt_instance,
                )

    metrics = metrics_calculator.compute_metrics()
    print(metrics)
    # Convert all NumPy types to native Python types for JSON serialization
    metrics = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=4, default=convert_to_builtin_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Panoptic Quality metrics")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="panoptic_metrics.json")
    parser.add_argument("--min-points", type=int, default=5)

    args = parser.parse_args()
    main(args)
