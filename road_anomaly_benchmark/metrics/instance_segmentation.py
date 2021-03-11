
import dataclasses

import numpy as np

from easydict import EasyDict
from scipy.ndimage.measurements import label
from pathlib import Path

from .base import EvaluationMetric, MetricRegistry
from ..evaluation import DIR_OUTPUTS
from ..datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file
from ..jupyter_show_image import adapt_img_data, imwrite


def default_instancer(anomaly_p: np.ndarray, label_pixel_gt: np.ndarray, thresh_p: float,
                      thresh_segsize: int, thresh_instsize: int = 0):

    """segmentation from pixel-wise anoamly scores"""
    segmentation = np.copy(anomaly_p)
    segmentation[anomaly_p > thresh_p] = 1
    segmentation[anomaly_p <= thresh_p] = 0

    anomaly_gt = np.zeros(label_pixel_gt.shape)
    anomaly_gt[label_pixel_gt == 1] = 1
    anomaly_pred = np.zeros(label_pixel_gt.shape)
    anomaly_pred[segmentation == 1] = 1
    anomaly_pred[label_pixel_gt == 255] = 0

    """connected components"""
    structure = np.ones((3, 3), dtype=np.int)
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """remove connected compontents below size threshold"""
    if thresh_segsize is not None:
        minimum_cc_sum  = thresh_segsize
        labeled_mask = np.copy(anomaly_seg_pred)
        for comp in range(n_seg_pred+1):
            if len(anomaly_seg_pred[labeled_mask == comp]) < minimum_cc_sum:
                anomaly_seg_pred[labeled_mask == comp] = 0
    labeled_mask = np.copy(anomaly_instances)
    for comp in range(n_anomaly + 1):
        if len(anomaly_instances[labeled_mask == comp]) < thresh_instsize:
            label_pixel_gt[labeled_mask == comp] = 255

    mask_roi = label_pixel_gt < 255
    return anomaly_instances[mask_roi], anomaly_seg_pred[mask_roi]


def segment_metrics(anomaly_instances, anomaly_seg_pred, iou_thresholds):
    """
    function that computes the segments metrics based on the adjusted IoU
    anomaly_instances: (numpy array) anomaly instance annoation
    anomaly_seg_pred: (numpy array) anomaly instance prediction
    iou_threshold: (float) threshold for true positive
    """

    """Loop over ground truth instances"""
    sIoU_gt = []
    size_gt = []

    for i in np.unique(anomaly_instances[anomaly_instances>0]):
        tp_loc = anomaly_seg_pred[anomaly_instances == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])

        """calc area of intersection"""
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_seg_pred[np.logical_and(~np.isin(anomaly_instances, [0, i]), np.isin(anomaly_seg_pred, seg_ind))])

        adjusted_union = np.sum(np.isin(anomaly_seg_pred, seg_ind)) + np.sum(
            anomaly_instances == i) - intersection - adjustment
        sIoU_gt.append(intersection / adjusted_union)
        size_gt.append(np.sum(anomaly_instances == i))

    """Loop over prediction instances"""
    sIoU_pred = []
    size_pred = []
    for i in np.unique(anomaly_seg_pred[anomaly_seg_pred>0]):
        tp_loc = anomaly_instances[anomaly_seg_pred == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])
        intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
        adjustment = len(
            anomaly_instances[np.logical_and(~np.isin(anomaly_seg_pred, [0, i]), np.isin(anomaly_instances, seg_ind))])
        adjusted_union = np.sum(np.isin(anomaly_instances, seg_ind)) + np.sum(
            anomaly_seg_pred == i) - intersection - adjustment
        sIoU_pred.append(intersection / adjusted_union)
        size_pred.append(np.sum(anomaly_seg_pred == i))

    sIoU_gt = np.array(sIoU_gt)
    sIoU_pred = np.array(sIoU_pred)
    size_gt = np.array((size_gt))
    size_pred = np.array(size_pred)

    """create results dictionary"""
    results = EasyDict(sIoU_gt=sIoU_gt, sIoU_pred=sIoU_pred, size_gt=size_gt, size_pred=size_pred)
    for t in iou_thresholds:
        results["tp_" + str(int(t*100))] = np.count_nonzero(sIoU_gt >= t)
        results["fn_" + str(int(t*100))] = np.count_nonzero(sIoU_gt < t)
        results["fp_" + str(int(t*100))] = np.count_nonzero(sIoU_pred < t)

    return results


@dataclasses.dataclass
class ResultsInfo:
    method_name : str
    dataset_name : str

    tp_25 : int
    tp_50 : int
    tp_75 : int
    fn_25 : int
    fn_50 : int
    fn_75 : int
    fp_25 : int
    fp_50 : int
    fp_75 : int
    f1_25 : float
    f1_50 : float
    f1_75 : float

    sIoU_gt : float
    sIoU_pred : float

    def __iter__(self):
        return dataclasses.asdict(self).items()

    def save(self, path):
        hdf5_write_hierarchy_to_file(path, dataclasses.asdict(self))

    @classmethod
    def from_file(cls, path):
        return cls(**hdf5_read_hierarchy_from_file(path))


@MetricRegistry.register_class()
class MetricSegment(EvaluationMetric):
    configs = [
        EasyDict(
            name='SegEval',
            thresh_p=None,
            thresh_sIoU=[0.25, 0.5, 0.75],
            thresh_segsize=500,
            thresh_instsize=100,
        ),
        EasyDict(
            name='SegEval-AnomalyTrack',
            thresh_p=None,
            thresh_sIoU=[0.25, 0.5, 0.75],
            thresh_segsize=500,
            thresh_instsize=100,
        ),
        EasyDict(
            name='SegEval-ObstacleTrack',
            thresh_p=None,
            thresh_sIoU=[0.25, 0.5, 0.75],
            thresh_segsize=50,
            thresh_instsize=10,
        )
    ]

    @property
    def name(self):
        return self.cfg.name

    def vis_frame(self, fid, dset_name, method_name, mask_roi, anomaly_p, image=None, **_):
        segmentation = np.copy(anomaly_p)
        segmentation[anomaly_p > self.cfg.thresh_p] = 1
        segmentation[anomaly_p <= self.cfg.thresh_p] = 0
        h, w = mask_roi.shape[:2]
        canvas = image.copy() if image is not None else np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_color = adapt_img_data(segmentation)
        canvas[mask_roi] = canvas[mask_roi] // 2 + heatmap_color[mask_roi] // 2
        imwrite(DIR_OUTPUTS / f'vis_SegPred' / method_name / dset_name / f'{fid}.webp', canvas)

    def process_frame(self, label_pixel_gt: np.ndarray, anomaly_p: np.ndarray, fid : str=None, dset_name : str=None,
                      method_name : str=None, visualize : bool = True, **_):
        """
        @param label_pixel_gt: HxW uint8
            0 = in-distribution / road
            1 = anomaly / obstacle
            255 = void / ignore
        @param anomaly_p: HxW float16
            heatmap of per-pixel anomaly detection, value from 0 to 1
        @param visualize: bool
            saves an image with segment predictions
        """

        mask_roi = label_pixel_gt < 255
        anomaly_gt, anomaly_pred = default_instancer(anomaly_p, label_pixel_gt, self.cfg.thresh_p,
                                                     self.cfg.thresh_segsize, self.cfg.thresh_instsize)

        results = segment_metrics(anomaly_gt, anomaly_pred, self.cfg.thresh_sIoU)

        if visualize and fid is not None and dset_name is not None and method_name is not None:
            self.vis_frame(fid=fid, dset_name=dset_name, method_name=method_name, mask_roi=mask_roi,
                           anomaly_p=anomaly_p, **_)

        return results

    def aggregate(self, frame_results: list, method_name: str, dataset_name: str):

        sIoU_gt_mean = sum(np.sum(r.sIoU_gt) for r in frame_results) / sum(len(r.sIoU_gt) for r in frame_results)
        sIoU_pred_mean = sum(np.sum(r.sIoU_pred) for r in frame_results) / sum(len(r.sIoU_pred) for r in frame_results)
        ag_results = {"sIoU_gt" : sIoU_gt_mean, "sIoU_pred" : sIoU_pred_mean}
        print("Mean sIoU GT   :", sIoU_gt_mean)
        print("Mean sIoU PRED :", sIoU_pred_mean)
        for t in self.cfg.thresh_sIoU:
            tp_total = sum(r["tp_" + str(int(t*100))] for r in frame_results)
            fn_total = sum(r["fn_" + str(int(t*100))] for r in frame_results)
            fp_total = sum(r["fp_" + str(int(t*100))] for r in frame_results)
            f1 = (2 * tp_total) / (2 * tp_total + fn_total + fp_total)
            ag_results["tp_" + str(int(t * 100))] = tp_total
            ag_results["fn_" + str(int(t * 100))] = fn_total
            ag_results["fp_" + str(int(t * 100))] = fp_total
            ag_results["f1_" + str(int(t * 100))] = f1
            print("---sIoU thresh =", t)
            print("Number of TPs  :", tp_total)
            print("Number of FNs  :", fn_total)
            print("Number of FPs  :", fp_total)
            print("F1 score       :", f1)

        seg_info = ResultsInfo(
            method_name=method_name,
            dataset_name=dataset_name,
            **EasyDict(ag_results),
        )

        return seg_info

    def persistence_path_data(self, method_name, dataset_name):
        return DIR_OUTPUTS / self.name / 'data' / f'{self.name}Results_{method_name}_{dataset_name}.hdf5'

    def save(self, aggregated_result, method_name: str, dataset_name: str, path_override: Path = None):
        out_path = path_override or self.persistence_path_data(method_name, dataset_name)
        aggregated_result.save(out_path)

    def load(self, method_name: str, dataset_name: str, path_override: Path = None):
        out_path = path_override or self.persistence_path_data(method_name, dataset_name)
        return ResultsInfo.from_file(out_path)

    def fields_for_table(self):
        return ['sIoU_gt', 'sIoU_pred', 'fn_25', 'fp_25', 'f1_25', 'fn_50', 'fp_50', 'f1_50', 'fn_75', 'fp_75', 'f1_75']

    def get_thresh_p_from_curve(self, method_name, dataset_name):
        out_path = DIR_OUTPUTS / "PixBinaryClass" / 'data' / f'PixClassCurve_{method_name}_{dataset_name}.hdf5'
        pixel_results = hdf5_read_hierarchy_from_file(out_path)
        if "best_f1_threshold" in pixel_results.keys():
            self.cfg.thresh_p = pixel_results.best_f1_threshold
        else:
            prc = pixel_results.curve_precision
            rec = pixel_results.curve_recall
            f1_scores = (2 * prc * rec) / (prc + rec)
            ix = np.argmax(f1_scores)
            self.cfg.thresh_p  = float(pixel_results.thresholds[ix])

