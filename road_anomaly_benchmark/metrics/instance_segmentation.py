import numpy as np
from easydict import EasyDict

from .base import EvaluationMetric, MetricRegistry, save_figure, save_table
from scipy.ndimage.measurements import label


def segment_metrics(anomaly_gt, anomaly_pred, iou_threshold):
    """
    function that computes the segments metrics based on the adjusted IoU
    anomaly_gt: (numpy array) anomaly annoation
    anomaly_pred: (numpy array) anomaly anomaly_pred
    iou_threshold: (float) threshold for true positive
    """

    structure = np.ones((3, 3), dtype=np.int)
    # connected components
    anomaly_instances, n_anomaly = label(anomaly_gt, structure)
    anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)

    """Loop over ground truth instances"""
    sIoU_gt = []
    size_gt = []
    for i in range(1, n_anomaly + 1):
        # TODO bbox transform
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

    tp = len([i for i in range(len(sIoU_gt)) if sIoU_gt[i] >= iou_threshold])
    fn = len([i for i in range(len(sIoU_gt)) if sIoU_gt[i] < iou_threshold])
    fp = len([i for i in range(len(sIoU_pred)) if sIoU_pred[i] < iou_threshold])

    # return tp, fn, fp, sIoU_gt, sIoU_pred

    return EasyDict(tp=tp, fn=fn, fp=fp, sIoU_gt=sIoU_gt, sIoU_pred=sIoU_pred, size_gt=size_gt, size_pred=size_pred)



@MetricRegistry.register_class()
class MetricSegment(EvaluationMetric):
    configs = [
        EasyDict(
            name='SegIoU',
        )
    ]

    @property
    def name(self):
        return self.cfg.name


    def process_frame(self, label_pixel_gt: np.ndarray, anomaly_p: np.ndarray,
                      thresh_p: float = 0.8, thresh_sIoU: float = 0.5, **_):
        """
        @param label_pixel_gt: HxW uint8
            0 = in-distribution / road
            1 = anomaly / obstacle
            255 = void / ignore
        @param anomaly_p: HxW float16
            heatmap of per-pixel anomaly detection, value from 0 to 1
        """

        segmentation = np.copy(anomaly_p)
        segmentation[segmentation > thresh_p] = 1
        segmentation[segmentation <= thresh_p] = 0

        anomaly_gt = np.zeros(label_pixel_gt.shape)
        anomaly_gt[label_pixel_gt == 1] = 1
        anomaly_pred = np.zeros(label_pixel_gt.shape)
        anomaly_pred[segmentation == 1] = 1
        anomaly_pred[label_pixel_gt == 255] = 0

        results = segment_metrics(anomaly_gt, anomaly_pred, thresh_sIoU)

        # from PIL import Image
        # Image.fromarray((anomaly_pred*255).astype("uint8")).save("pred.png")
        # Image.fromarray((anomaly_gt * 255).astype("uint8")).save("gt.png")
        # Image.fromarray((label_pixel_gt).astype("uint8")).save("void.png")
        # exit()

        return results


    def aggregate(self, frame_results: list, method_name: str, dataset_name: str):

        l = len(frame_results)
        tp_total = sum([frame_results[i]["tp"] for i in range(l)])
        fn_total = sum([frame_results[i]["fn"] for i in range(l)])
        fp_total = sum([frame_results[i]["fp"] for i in range(l)])

        sIoU_gt_all = []
        sIoU_pred_all = []
        for i in range(l):
            sIoU_gt_all = sIoU_gt_all + frame_results[i]["sIoU_gt"]
            sIoU_pred_all = sIoU_pred_all + frame_results[i]["sIoU_pred"]
        sIoU_gt_mean = np.mean(sIoU_gt_all)
        sIoU_pred_mean = np.mean(sIoU_pred_all)

        print("Number of TPs  :", tp_total)
        print("Number of FNs  :", fn_total)
        print("Number of FPs  :", fp_total)
        print("Mean sIoU GT   :", sIoU_gt_mean)
        print("Mean sIoU PRED :", sIoU_pred_mean)

        exit()
        # TODO: returning the results
