
from pathlib import Path
from os import environ
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from . import DIR_SRC
from .datasets import DatasetRegistry
from .datasets.dataset_io import ChannelLoaderHDF5

DIR_OUTPUTS = Path(environ.get('DIR_OUTPUTS', DIR_SRC / 'outputs'))

class Evaluation:

	channels = {
		'anomaly_p': ChannelLoaderHDF5(
			str(DIR_OUTPUTS / "anomaly_p/{method_name}/{dset_name}/{fid}.hdf5"),
			compression = 9,
		),
	}

	threads = None

	def __init__(self, method_name, dataset_name, threaded_saver=True, num_workers=8):
		self.method_name = method_name
		self.dataset_name = dataset_name

		if threaded_saver:
			self.threads = ThreadPoolExecutor(num_workers)

	def get_dataset(self):
		return DatasetRegistry.get(self.dataset_name)

	def save_output(self, frame, anomaly_p):
		value = anomaly_p.astype(np.float16)

		f = self.channels['anomaly_p'].write
		a = (value,)
		kw = dict(method_name = self.method_name, **frame)

		if self.threads is not None:
			# TODO log errors from thread
			self.threads.submit(f, *a, **kw)
		else:
			f(*a, **kw)

		# self.channels['anomaly_p'].write(value, method_name = self.method_name, **frame)
	
	def calculate_metric_from_saved_outputs(self, metric, sample=None):
		# TODO sample is part of evaluation

		fr_results = []

		if sample is not None:
			dset_name, frame_indices = sample

			ds = DatasetRegistry.get(dset_name)
			fr_iterable = (ds[i] for i in frame_indices)
			fr_iterable_len = frame_indices.__len__()
		else:
			fr_iterable = self.get_dataset()
			fr_iterable_len = fr_iterable.__len__()

		for fr in tqdm(fr_iterable, total=fr_iterable_len):
			fr_result = metric.process_frame(
				anomaly_p = self.channels['anomaly_p'].read(
					method_name = self.method_name, 
					dset_name = fr.dset_name,
					fid = fr.fid,
				),
				method_name = self.method_name, 
				**fr,
			)
			fr_results.append(fr_result)

		return metric.aggregate(
			fr_results,
			method_name = self.method_name,
			dataset_name = self.dataset_name,
		)


	def wait_to_finish_saving(self):
		if self.threads is not None:
			self.threads.shutdown(True)
			self.threads = None




