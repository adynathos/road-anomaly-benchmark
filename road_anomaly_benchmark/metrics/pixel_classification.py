
import dataclasses
from math import ceil
from typing import List, Dict

import numpy as np
from matplotlib import pyplot
from easydict import EasyDict

from .base import EvaluationMetric
from ..evaluation import DIR_OUTPUTS
from ..jupyter_show_image import adapt_img_data, imwrite

from pandas import DataFrame, Series
from operator import attrgetter

@dataclasses.dataclass
class BinaryClassificationCurve:
	method_name : str
	dataset_name : str
#	display_name : str = ''

	area_PRC : float
	curve_recall : np.ndarray
	curve_precision : np.ndarray

	area_ROC : float
	curve_tpr : np.ndarray
	curve_fpr : np.ndarray

	# TODO thresholds?

	fpr_at_95_tpr : float = -1
	threshold_at_95_tpr : float = -1

	IOU_at_05: float = float('nan')
	PDR_at_05: float = float('nan')

	def __iter__(self):
		return dataclasses.asdict(self).items()

	def save(self, path):
		hdf5_write_hierarchy_to_file(path, dataclasses.asdict(self))

	@classmethod
	def from_file(cls, path):
		return cls(**hdf5_read_hierarchy_from_file(path))

def binary_confusion_matrix(prob, gt_label_bool, num_bins=1024, normalize=False, dtype=np.float64):
	area = gt_label_bool.__len__()

	gt_area_true = np.count_nonzero(gt_label_bool)
	gt_area_false = area - gt_area_true

	prob_at_true = prob[gt_label_bool]
	prob_at_false = prob[~gt_label_bool]

	bins = np.linspace(0, 1, num_bins+1)

	# TODO dynamic bins
	# - bins spread uniforms in 0 .. 1
	#	np.linspace(0, 1, ceil(levels*0.5))
	# - bins aligned to percentiles
	# 	np.quantile(prob, np.linspae(0, 1, levels//2))


	tp_rel, _ = np.histogram(prob_at_true, bins=bins, range=[0, 1])

	tp = np.cumsum(tp_rel[::-1])
	fn = gt_area_true - tp

	fp_rel, _ = np.histogram(prob_at_false, bins=bins, range=[0, 1])

	fp = np.cumsum(fp_rel[::-1])
	tn = gt_area_false - fp

	cmat_sum = np.array([
		[tp, fp],
		[fn, tn],
	]).transpose(2, 0, 1).astype(dtype)

	cmat_rel = np.array([
		[tp_rel, -fp_rel],
		[-tp_rel, fp_rel],
	]).transpose(2, 0, 1).astype(dtype)
	
	if normalize:
		cmat_sum *= (1./area)
		cmat_rel *= (1./area)

	return EasyDict(
		bins = bins,
		cmat_sum = cmat_sum,
		cmat_rel = cmat_rel,
		num_pos = gt_area_true,
		num_neg = gt_area_false,
	)


def test_binary_confusion_matrix():
	pp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	gt = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=np.bool)
	cmat = binary_confusion_matrix(pp, gt, levels=20).cmat_sum
	cmat_all_p = np.sum(cmat[:, :, 0], axis=1)
	cmat_all_n = np.sum(cmat[:, :, 1], axis=1)

	print(cmat_all_p, cmat_all_n)

	pyplot.plot(cmat[:, 0, 1] / cmat_all_n, cmat[:, 0, 0] / cmat_all_p)

def cmats_to_rocinfo(cmats):
	# roi_area = np.count_nonzero(roi) if roi is not None else 1

	num_levels = cmats.shape[0]
	tp = cmats[:, 0, 0]
	fp = cmats[:, 0, 1]
	fn = cmats[:, 1, 0]
	tn = cmats[:, 1, 1]

	tp_rates = tp / (tp+fn)
	fp_rates = fp / (fp+tn)

	pos_pred = tp+fp
	precisions = tp / pos_pred
	precisions[pos_pred < 10] = 1

	pos_gt = tp+fn
	recalls = tp / (tp+fn)
	recalls[pos_gt < 10] = 1

	return EasyDict(
		# num_levels = num_levels,
		curve_tpr = tp_rates,
		curve_fpr = fp_rates,
		curve_precision = precisions,
		curve_recall = recalls,

		#cmats = cmats,
		area_ROC = np.trapz(tp_rates, fp_rates),
		area_PRC = np.trapz(precisions, recalls),

		# TODO fpr95
	)



def plot_classification_curves_draw_entry(plot_roc : pyplot.Axes, plot_prc : pyplot.Axes, curve_info : BinaryClassificationCurve, display_name : str):

	if plot_prc is not None:
		plot_prc.plot(curve_info.curve_recall, curve_info.curve_precision,
			# label='{lab:<24}{a:.02f}'.format(lab=label, a=area_under),
			label=f'{display_name}  {curve_info.area_PRC:.02f}',
			marker = '.',
		)

	if plot_roc is not None:
		plot_roc.plot(curve_info.curve_fpr, curve_info.curve_tpr,
			# label='{lab:<24}{a:.02f}'.format(lab=label, a=area_under),
			label=f'{display_name}  {curve_info.area_ROC:.03f}',
			marker = '.',
		)


def plot_classification_curves(curve_infos : List[BinaryClassificationCurve], method_names : Dict[str, str] = {}):
	
	table_scores = DataFrame(data = [
		Series({
			'AveragePrecision': crv.area_PRC, 
			'AUROC': crv.area_ROC, 
			'FPR-at-95-TPR:': crv.fpr_at_95_tpr,
			'IOU': crv.IOU_at_05,
			'PDR': crv.PDR_at_05,
			},
			name = method_names.get(crv.method_name, crv.method_name),
		)
		for crv in curve_infos
	])
	table_scores = table_scores.sort_values('AveragePrecision', ascending=False)


	fig = pyplot.figure(figsize=(18, 8))
	plot_roc, plot_prc = fig.subplots(1, 2)
	
	plot_prc.set_xlabel('recall')
	plot_prc.set_ylabel('precision')
	plot_prc.set_xlim([0, 1])
	plot_prc.set_ylim([0, 1])

	
	plot_roc.set_xlabel('false positive rate')
	plot_roc.set_ylabel('true positive rate')
	plot_roc.set_xlim([0, 0.2])

	# sort descending by AP
	curve_infos_sorted = list(curve_infos)
	curve_infos_sorted.sort(key=attrgetter('area_PRC'), reverse=True)

	for crv in curve_infos_sorted:
		plot_classification_curves_draw_entry(
			plot_prc = plot_prc, 
			plot_roc = plot_roc, 
			curve_info = crv,
			display_name = method_names.get(crv.method_name, crv.method_name),
		)

	def make_legend(plot_obj, position='lower right', title=None):
		handles, labels = plot_obj.get_legend_handles_labels()
		plot_obj.legend(handles, labels, loc=position, title=title)

		plot_obj.grid(True)

	make_legend(plot_prc, 'lower left', title='Average Precision')
	make_legend(plot_roc, 'lower right', title='AUROC')

	fig.tight_layout()

	return EasyDict(
		plot_figure = fig,
		score_table = table_scores,
	)









class MetricPixelClassification(EvaluationMetric):

	

	# pixel scores in a given image are quantized into bins, 
	# so that big datasets can be stored in memory and processed in parallel
	num_bins: int = 1024

	# something about visualization

	def vis_frame(self, fid, dset_name, method_name, mask_roi, anomaly_p, image = None, **_):
		h, w = mask_roi.shape[:2]

		canvas = image.copy() if image is not None else np.zeros((h, w, 3), dtype=np.uint8)

		heatmap_color = adapt_img_data(anomaly_p)

		canvas[mask_roi] = canvas[mask_roi]//2 + heatmap_color[mask_roi]//2

		imwrite(
			DIR_OUTPUTS / f'vis_PixelClassification' / method_name / dset_name / f'{fid}_demo_anomalyP.webp',
			canvas,
		)


	def process_frame(self, label_pixel_gt : np.ndarray, anomaly_p : np.ndarray, fid : str=None, dset_name : str=None, method_name : str=None, **_):
		"""
		@param label_pixel_gt: HxW uint8
			0 = road
			1 = obstacle
			255 = ignore
		@param anomaly_p: HxW float16
			heatmap of per-pixel anomaly detection, value from 0 to 1
		@param fid: frame identifier, for saving extra outputs
		@param dset_name: dataset identifier, for saving extra outputs
		"""

		mask_roi = label_pixel_gt < 255

		labels_in_roi = label_pixel_gt[mask_roi]
		predictions_in_roi = anomaly_p[mask_roi]

		bc = binary_confusion_matrix(
			prob = predictions_in_roi,
			gt_label_bool = labels_in_roi.astype(bool),
			num_bins = self.num_bins,
		)
		# bc.cmats, bc.bins

		# visualization
		if fid is not None and dset_name is not None and method_name is not None:
			self.vis_frame(fid=fid, dset_name=dset_name, method_name=method_name, mask_roi=mask_roi, anomaly_p=anomaly_p, **_)

		#TODO dataclass
		return bc

	def aggregate(self, frame_results : list, method_name : str, dataset_name : str):
		# fuse cmats FIXED BINS

		print(frame_results[0].cmat_sum.shape)

		cmat_sum = np.sum([result.cmat_sum for result in frame_results], axis=0)

		print(cmat_sum.shape)

		rocinfo = cmats_to_rocinfo(cmat_sum)

		bc_info = BinaryClassificationCurve(
			method_name = method_name,
			dataset_name = dataset_name,
			**rocinfo,
		)

		return bc_info
