 
from .base import EvaluationMetric

class MetricPixelClassification(EvaluationMetric):

	def process_frame(self, frame : EasyDict):
		...

	def aggregate(self, frame_results : list):
		...

