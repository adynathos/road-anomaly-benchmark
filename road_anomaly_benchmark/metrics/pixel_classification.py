
import numpy as np
from easydict import EasyDict

from .base import EvaluationMetric
from ..evaluation import DIR_OUTPUTS
from ..jupyter_show_image import adapt_img_data, imwrite

def binary_confusion_matrix(prob, gt_label_bool, levels=1024, normalize=False):
	area = gt_label_bool.__len__()

	gt_area_true = np.count_nonzero(gt_label_bool)
	gt_area_false = area - gt_area_true

	prob_at_true = prob[gt_label_bool]
	prob_at_false = prob[~gt_label_bool]

	# TODO dynamic bins
	tp, bins = np.histogram(prob_at_true, bins=levels, range=[0, 1])
	tp = np.cumsum(tp[::-1])

	fn = gt_area_true - tp

	# TODO dynamic bins
	fp, _ = np.histogram(prob_at_false, bins=levels, range=[0, 1])
	fp = np.cumsum(fp[::-1])

	tn = gt_area_false - fp

	cmat = np.array([
		[tp, fp],
		[fn, tn],
	]).transpose(2, 0, 1)

	cmat = cmat.astype(np.float64)
	
	if normalize:
		cmat /= area

	return EasyDict(
		bins = bins,
		cmats = cmat,
	)


def test_binary_confusion_matrix():
	from matplotlib import pyplot
	pp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	gt = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=np.bool)
	cmat = binary_confusion_matrix(pp, gt, levels=20)
	cmat_all_p = np.sum(cmat[:, :, 0], axis=1)
	cmat_all_n = np.sum(cmat[:, :, 1], axis=1)

	print(cmat_all_p, cmat_all_n)

	pyplot.plot(cmat[:, 0, 1] / cmat_all_n, cmat[:, 0, 0] / cmat_all_p)



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
			levels = self.num_bins,
		)
		# bc.cmats, bc.bins

		# visualization
		if fid is not None and dset_name is not None and method_name is not None:
			self.vis_frame(fid=fid, dset_name=dset_name, method_name=method_name, mask_roi=mask_roi, anomaly_p=anomaly_p, **_)

		return EasyDict( #TODO dataclass
			cmats = bc.cmats,
			bins = bc.bins,
		)


	def aggregate(self, frame_results : list):
		...
		

	#TODO put in main eval

