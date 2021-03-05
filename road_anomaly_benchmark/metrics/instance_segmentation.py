import numpy as np

import cv2 as cv
from scipy.ndimage.measurements import label
from collections import namedtuple


class RoadObstacle:
	"""Masks Labels"""
	Label = namedtuple('Label', ['name', 'id', 'ignoreInEval', 'color'])
	labels = [
		Label('not obstacle',   0,  False, (255, 255, 255)),
		Label('obstacle'    ,   1,  False, (255, 102,   0)),
		Label('void'        , 255,   True, (  0,   0,   0)),
	]
	name2id = {label.name: label for label in labels}


def segment_metrics(segmentation, ground_truth, iou_threshold):
	"""
	function that computes the segments metrics based on the adjusted IoU
	segmentation: (numpy array) anomaly prediction
	ground_truth: (numpy array) anomaly annotation
	iou_threshold: (float) threshold for true positive
	"""
	anomaly_label = RoadObstacle.name2id["obstacle"].id
	anomaly_gt = np.zeros(ground_truth.shape)
	anomaly_gt[ground_truth==anomaly_label] = 1
	anomaly_pred = np.zeros(ground_truth.shape)
	anomaly_pred[segmentation==anomaly_label] = 1

	structure = np.ones((3, 3), dtype=np.int)

	# connected components
	anomaly_instances, n_anomaly = label(anomaly_gt, structure)
	anomaly_seg_pred, n_seg_pred = label(anomaly_pred, structure)


	"""Loop over ground truth instances"""
	gt_metric = {"sIoU": []}
	for i in range(1, n_anomaly+1):
		# TODO bbox transform
		tp_loc = anomaly_seg_pred[anomaly_instances == i]
		seg_ind = np.unique(tp_loc[tp_loc != 0])

		# calc area of intersection
		intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])

		adjustment = len(anomaly_seg_pred[np.logical_and(~np.isin(anomaly_instances, [0, i]), np.isin(anomaly_seg_pred, seg_ind))])
		
		adjusted_union = np.sum(np.isin(anomaly_seg_pred, seg_ind)) + np.sum(anomaly_instances == i) - intersection - adjustment
		gt_metric["sIoU"].append(intersection / adjusted_union)

	"""Loop over prediction instances"""
	pred_metric = {"sIoU": []}
	for i in range(1, n_seg_pred+1):
		tp_loc = anomaly_instances[anomaly_seg_pred == i]
		seg_ind = np.unique(tp_loc[tp_loc != 0])
		intersection = len(tp_loc[np.isin(tp_loc, seg_ind)])
		adjustment = len(anomaly_instances[np.logical_and(~np.isin(anomaly_seg_pred, [0, i]), np.isin(anomaly_instances, seg_ind))])
		adjusted_union = np.sum(np.isin(anomaly_instances, seg_ind)) + np.sum(anomaly_seg_pred == i) - intersection - adjustment
		pred_metric["sIoU"].append(intersection / adjusted_union)

	tp = len([ i for i in range(len(gt_metric["sIoU"])) if gt_metric["sIoU"][i] >= iou_threshold ])
	fn = len([ i for i in range(len(gt_metric["sIoU"])) if gt_metric["sIoU"][i] < iou_threshold ])
	fp = len([ i for i in range(len(pred_metric["sIoU"])) if pred_metric["sIoU"][i] < iou_threshold ])
	
	return tp, fn, fp



if __name__ == '__main__':

	ground_truth = cv2.imread('data/gt_labels_semantic.png', 0)
	segmentation = cv2.imread('data/pred_labels_semantic.png', 0)
	
	iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
	for iou_threshold in iou_thresholds:
		tp, fn, fp = segment_metrics(segmentation, ground_truth, iou_threshold)
		print("\nSegment Evaluation Metrics --- Threshold t = %.02f" % iou_threshold)
		print("Number of TPs:", tp)
		print("Number of FNs:", fn)
		print("Number of FPs:", fp)
