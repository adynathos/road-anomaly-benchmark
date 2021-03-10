
import dataclasses

import numpy as np

from easydict import EasyDict
from scipy.ndimage.measurements import label
from pathlib import Path

from .base import EvaluationMetric, MetricRegistry
from ..evaluation import DIR_OUTPUTS
from ..datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file


def segment_metrics(anomaly_gt, anomaly_pred, iou_threshold):
    """
    function that computes the segments metrics based on the adjusted IoU
    anomaly_gt: (numpy array) anomaly annoation
    anomaly_pred: (numpy array) anomaly anomaly_pred
    iou_threshold: (float) threshold for true positive
    """

    structure = np.ones((3, 3), dtype=np.int)

    """connected components"""
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """remove connected compontents below a size threshold"""
    # minimum_cc_sum  = 250
    # labeled_mask = np.copy(anomaly_seg_pred)
    # for comp in range(n_seg_pred):
    #     if np.sum(anomaly_seg_pred[labeled_mask == comp]) < minimum_cc_sum:
    #         anomaly_seg_pred[labeled_mask == comp] = 0

    """Loop over ground truth instances"""
    sIoU_gt = []
    size_gt = []
    for i in range(1, n_anomaly + 1):
        tp_loc = anomaly_seg_pred[anomaly_instances == i]
        seg_ind = np.unique(tp_loc[tp_loc != 0])

        # calc area of intersection
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
    for i in range(1, n_seg_pred + 1):
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

    tp = np.count_nonzero(sIoU_gt >= iou_threshold)
    fn = np.count_nonzero(sIoU_gt < iou_threshold)
    fp = np.count_nonzero(sIoU_pred < iou_threshold)

    return EasyDict(tp=tp, fn=fn, fp=fp, sIoU_gt=sIoU_gt, sIoU_pred=sIoU_pred, size_gt=size_gt, size_pred=size_pred)


@dataclasses.dataclass
class ResultsInfo:
    method_name : str
    dataset_name : str

    tp : int
    fn : int
    fp : int
    f1 : float
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
            thresh_p=0.8,
            thresh_sIoU=0.5
        )
    ]

    @property
    def name(self):
        return self.cfg.name

    def process_frame(self, label_pixel_gt: np.ndarray, anomaly_p: np.ndarray, **_):
        """
        @param label_pixel_gt: HxW uint8
            0 = in-distribution / road
            1 = anomaly / obstacle
            255 = void / ignore
        @param anomaly_p: HxW float16
            heatmap of per-pixel anomaly detection, value from 0 to 1
        """

        segmentation = np.copy(anomaly_p)
        segmentation[segmentation > self.cfg.thresh_p] = 1
        segmentation[segmentation <= self.cfg.thresh_p] = 0

        anomaly_gt = np.zeros(label_pixel_gt.shape)
        anomaly_gt[label_pixel_gt == 1] = 1
        anomaly_pred = np.zeros(label_pixel_gt.shape)
        anomaly_pred[segmentation == 1] = 1
        anomaly_pred[label_pixel_gt == 255] = 0

        results = segment_metrics(anomaly_gt, anomaly_pred, self.cfg.thresh_sIoU)

        # from PIL import Image
        # Image.fromarray((anomaly_pred*255).astype("uint8")).save("pred.png")
        # Image.fromarray((anomaly_gt * 255).astype("uint8")).save("gt.png")
        # Image.fromarray((label_pixel_gt).astype("uint8")).save("void.png")
        # exit()

        return results

    def aggregate(self, frame_results: list, method_name: str, dataset_name: str):

        tp_total = sum(r.tp for r in frame_results)
        fn_total = sum(r.fn for r in frame_results)
        fp_total = sum(r.fp for r in frame_results)
        f1 = 2 * tp_total / (2 * tp_total + fn_total + fp_total)
        sIoU_gt_mean = sum(np.sum(r.sIoU_gt) for r in frame_results) / sum(len(r.sIoU_gt) for r in frame_results)
        sIoU_pred_mean = sum(np.sum(r.sIoU_pred) for r in frame_results) / sum(len(r.sIoU_pred) for r in frame_results)

        print("Number of TPs  :", tp_total)
        print("Number of FNs  :", fn_total)
        print("Number of FPs  :", fp_total)
        print("F1 score       :", f1)
        print("Mean sIoU GT   :", sIoU_gt_mean)
        print("Mean sIoU PRED :", sIoU_pred_mean)

        seg_info = ResultsInfo(
            method_name=method_name,
            dataset_name=dataset_name,
            **EasyDict(tp=tp_total, fn=fn_total, fp=fp_total, f1=f1, sIoU_gt=sIoU_gt_mean, sIoU_pred=sIoU_pred_mean),
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
        return ['sIoU_gt', 'fn', 'fp', 'f1']
