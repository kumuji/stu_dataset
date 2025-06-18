import argparse
import json
from pathlib import Path

import numpy as np
import torch
from .common import load_labels, load_point_cloud
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics import auc, average_precision_score, roc_curve
from tqdm import tqdm


class UQEvaluator:
    def __init__(
        self, min_points=5, num_classes=2, ignore_labels=[255], instance_labels=[1]
    ):
        super(UQEvaluator, self).__init__()
        self.min_points = min_points
        self.num_classes = num_classes
        self.ignore_labels = ignore_labels
        self.instance_labels = instance_labels
        self.reset()

    def addBatchforUQ(
        self,
        points,
        prediction,
        semantic_label,
        instance_label,
    ):
        prediction = prediction.numpy()
        instance_prediction = np.zeros(prediction.shape, dtype=np.int64)
        # only if any predictions available
        if len(points[prediction]) > 0:
            clusters = (
                DBSCAN(
                    eps=1.0,
                    min_samples=1,
                    n_jobs=-1,
                )
                .fit(points[prediction])
                .labels_
            )
            instance_prediction[prediction] = clusters + 1

        self.addBatchUnknown(
            prediction,
            instance_prediction,
            semantic_label,
            instance_label,
        )

    def reset(self):
        self.pan_tp_uq = np.zeros(self.num_classes, dtype=np.int64)
        self.pan_iou_uq = np.zeros(self.num_classes, dtype=np.double)
        self.pan_fn_uq = np.zeros(self.num_classes, dtype=np.int64)
        self.pan_fp_uq = np.zeros(self.num_classes, dtype=np.int64)

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
            # print("*"*80)
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
            self.pan_tp_uq[cl] += np.sum(tp_indexes)
            self.pan_iou_uq[cl] += np.sum(ious[tp_indexes])

            # print("pan_iou_uq: ", self.pan_iou_uq)
            # print("TP: ",self.pan_tp_uq[cl])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            # count the FN
            self.pan_fn_uq[cl] += np.sum(
                np.logical_and(counts_gt >= self.min_points, matched_gt == False)
            )
            self.pan_fp_uq[cl] += np.sum(
                np.logical_and(counts_pred >= self.min_points, matched_pred == False)
            )
            # print("FN: ",self.pan_fn_uq[cl])

    def get_stats(self):
        return (
            self.pan_iou_uq[1],
            self.pan_tp_uq[1],
            self.pan_fp_uq[1],
            self.pan_fn_uq[1],
        )

    def getUQ(self):
        sq_uq_all = self.pan_iou_uq.astype(np.float64) / np.maximum(
            self.pan_tp_uq.astype(np.float64), 1e-15
        )

        recallq_all = self.pan_tp_uq.astype(np.float64) / np.maximum(
            self.pan_tp_uq.astype(np.float64) + self.pan_fn_uq.astype(np.float64), 1e-15
        )

        uq_all = sq_uq_all * recallq_all

        return sq_uq_all, recallq_all, uq_all


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


def calculate_auroc(predictions, targets):
    fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
    roc_auc = auc(fpr, tpr)
    fpr_best = 0
    threshold = 0
    for i, j, threshold in zip(tpr, fpr, thresholds):
        if i > 0.95:
            fpr_best = j
            break
    return roc_auc, fpr_best, threshold


def calculate_ood_metrics(targets, predictions):
    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    AP = average_precision_score(y_true=targets, y_score=predictions)
    roc_auc, fpr, threshold = calculate_auroc(predictions=predictions, targets=targets)
    return {
        "AP": AP * 100,
        "FPR95": fpr * 100,
        "AUROC": roc_auc * 100,
        "threshold": threshold,
    }


def get_rba(logit, mask, inv_map):
    confid = mask.float().sigmoid().matmul(logit)
    rba = -confid.tanh().sum(dim=1)[inv_map]
    rba[rba < -1] = -1
    rba = rba + 1
    return rba


def get_void(logit, mask, inv_map):
    confid = mask.float().sigmoid().matmul(logit)
    prediction = confid.argmax(dim=1)[inv_map] == 0
    return prediction.float()


def get_maxlogit(logit, mask, inv_map):
    confid = mask.float().sigmoid().matmul(logit)
    max_softmax = torch.max(confid, dim=1).values[inv_map]
    max_softmax = (max_softmax * -1) + 1
    return max_softmax


def process_sequence(sequence, data_path, prediction_path, prediction_type):
    pc_indices = sorted(
        [int(f.stem) for f in (data_path / f"{sequence}" / "velodyne").glob("*.bin")]
    )

    sequence_data = {
        "sequence_name": sequence,
        "anomaly_sizes": [],
        "unique_labels": [],
        "no_labels_frames": [],
        "uq": [],
        "metrics": None,
    }

    predictions_list = []
    labels_list = []
    unique_list = []
    no_labels = []
    anomaly_size = []
    distances_list = []

    uq_per_sequence = UQEvaluator()

    for index in pc_indices:
        points, _ = load_point_cloud(
            data_path / f"{sequence}" / "velodyne" / f"{index:06}.bin"
        )
        semantic_labels, instance_label = load_labels(
            data_path / f"{sequence}" / "labels" / f"{index:06}.label"
        )

        prediction = np.load(
            prediction_path / f"{sequence}" / f"{prediction_type}_{index:06}.npy"
        )
        prediction = torch.from_numpy(prediction)

        # prediction = get_rba(logit, mask, inv_map)
        # prediction = get_maxlogit(logit, mask, inv_map)
        # prediction = get_void(logit, mask, inv_map)

        distances = np.linalg.norm(points, axis=1)

        unique_labels = np.unique(semantic_labels)
        inlier_labels = np.where(semantic_labels != 0, 0, -1)
        processed_labels = np.where(semantic_labels == 2, 1, inlier_labels)
        processed_labels = np.where(
            (distances > 50) | (distances < 2.5), -1, processed_labels
        )
        labels = processed_labels[processed_labels != -1]

        if sum(labels) < 5:
            no_labels.append(index)
            continue

        uq_per_sequence.addBatchforUQ(
            points[processed_labels != -1],
            prediction[processed_labels != -1] > 0.5,
            labels,
            instance_label[processed_labels != -1],
        )

        prediction = prediction[processed_labels != -1]
        distances = distances[processed_labels != -1]

        # Append per-scan metrics to sequence data
        distances_list.append(distances)
        labels_list.append(labels)
        predictions_list.append(prediction)
        unique_list.append(unique_labels)
        anomaly_size.append(np.sum(labels == 1))

    sq, rq, uq = uq_per_sequence.getUQ()
    print(f"seq: {sequence} sq: {sq[1] * 100} rq: {rq[1] * 100} uq: {uq[1] * 100}")

    sequence_data.update(
        {
            "metrics": calculate_ood_metrics(labels_list, predictions_list),
            "anomaly_sizes": anomaly_size,
            "unique_labels": list(np.unique(np.concatenate(unique_list))),
            "no_labels_frames": no_labels,
            "uq": uq_per_sequence.get_stats(),
        }
    )

    print(sequence_data["metrics"])
    # Return collected data for overall metrics calculation
    return (
        labels_list,
        predictions_list,
        unique_list,
        anomaly_size,
        distances_list,
        sequence_data,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_type", type=str)
    parser.add_argument("--data_path", type=str, default="data/test_scenes")
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--num_jobs", type=int, default=1)

    args = parser.parse_args()
    print(f"{args}")

    overall_metrics = {"sequence_metrics": []}

    overall_predictions = []
    overall_labels = []
    overall_distances = []

    def process_and_collect(sequence_path):
        sequence_name = sequence_path.name
        (
            labels_list,
            predictions_list,
            unique_list,
            anomaly_size,
            distances_list,
            sequence_data,
        ) = process_sequence(
            sequence_name, Path(args.data_path), Path(args.prediction_path), args.prediction_type
        )
        return labels_list, predictions_list, distances_list, sequence_data

    sequence_paths = sorted(list(Path(args.data_path).glob("*")))
    print(f"sequences: {len(sequence_paths)}")
    print(f"sequences: {[p.stem for p in sequence_paths]}")
    results = Parallel(n_jobs=args.num_jobs)(
        delayed(process_and_collect)(sequence_path)
        for sequence_path in tqdm(sequence_paths, leave=False)
    )

    # Collect results
    for labels_list, predictions_list, distances_list, sequence_data in results:
        overall_metrics["sequence_metrics"].append(sequence_data)
        overall_labels.extend(labels_list)
        overall_predictions.extend(predictions_list)
        overall_distances.extend(distances_list)

    pan_iou_uq = list()
    pan_tp_uq = list()
    pan_fp_uq = list()
    pan_fn_uq = list()
    for stats in overall_metrics["sequence_metrics"]:
        pan_iou_uq.append(stats["uq"][0])
        pan_tp_uq.append(stats["uq"][1])
        pan_fp_uq.append(stats["uq"][2])
        pan_fn_uq.append(stats["uq"][3])

    pan_iou_uq = sum(pan_iou_uq)
    pan_tp_uq = sum(pan_tp_uq)
    pan_fp_uq = sum(pan_fp_uq)
    pan_fn_uq = sum(pan_fn_uq)
    sq_uq_all = pan_iou_uq.astype(np.float64) / np.maximum(
        pan_tp_uq.astype(np.float64), 1e-15
    )
    recallq_all = pan_tp_uq.astype(np.float64) / np.maximum(
        pan_tp_uq.astype(np.float64) + pan_fn_uq.astype(np.float64), 1e-15
    )
    uq_all = sq_uq_all * recallq_all

    rq_all = pan_tp_uq.astype(np.float64) / np.maximum(
        pan_tp_uq.astype(np.float64)
        + 0.5 * pan_fp_uq.astype(np.float64)
        + 0.5 * pan_fn_uq.astype(np.float64),
        1e-15,
    )
    pq_all = sq_uq_all * rq_all

    # Calculate and store overall metrics
    overall_metrics["overall_metrics"] = {
        "semantic": calculate_ood_metrics(overall_labels, overall_predictions),
        "object": {
            "SQ": sq_uq_all * 100,
            "RecallQ": recallq_all * 100,
            "UQ": uq_all * 100,
            "RQ": rq_all * 100,
            "PQ": pq_all * 100,
        },
    }
    print(overall_metrics["overall_metrics"])

    # Calculate metrics for different distance thresholds
    distance_thresholds = [0, 10, 20, 30, 40, 50]
    overall_metrics["distance_threshold_metrics"] = {}

    for threshold_idx in range(1, len(distance_thresholds)):
        threshold_labels = []
        threshold_predictions = []

        for labels, predictions, distances in zip(
            overall_labels,
            overall_predictions,
            overall_distances,
        ):
            mask_within_threshold = (
                distances > distance_thresholds[threshold_idx - 1]
            ) & (distances <= distance_thresholds[threshold_idx])
            threshold_labels.append(labels[mask_within_threshold])
            threshold_predictions.append(predictions[mask_within_threshold])

        overall_metrics["distance_threshold_metrics"][
            f"{distance_thresholds[threshold_idx]}m"
        ] = {
            "metric": calculate_ood_metrics(threshold_labels, threshold_predictions),
        }

    print(overall_metrics["overall_metrics"])
    print(overall_metrics["distance_threshold_metrics"])

    # Save metrics to JSON file
    with open(args.output_path, "w") as f:
        json.dump(overall_metrics, f, indent=4, default=convert_to_builtin_types)

    print(f"Metrics logged to {args.output_path}")
