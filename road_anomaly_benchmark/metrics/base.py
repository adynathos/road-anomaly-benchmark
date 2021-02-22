 
from easydict import EasyDict


class EvaluationMetric:

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)

	def process_frame(self, frame : EasyDict):
		"""
		@param frame: dict containing frame fields, such as `image`, `semantic_class_gt`, `pred_anomaly_p`
		@return: 
		"""

		...

	def aggregate(self, frame_results : list):
		"""
		@param frame_results: sequence of outputs of `process_frame` for the whole dataset
		@return: 
		"""

		...
	


