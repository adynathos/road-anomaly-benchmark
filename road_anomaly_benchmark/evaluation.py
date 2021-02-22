
from pathlib import Path
from os import environ
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import DIR_SRC
from .datasets import DatasetRegistry
from .datasets.dataset_io import ChannelLoaderHDF5

DIR_OUTPUTS = Path(environ.get('DIR_OUTPUTS', DIR_SRC / 'outputs'))

class Evaluation:

	channels = {
		'anomaly_p': ChannelLoaderHDF5(str(DIR_OUTPUTS / "anomaly_p/{method_name}/{dset_name}/{fid}.hdf5")),
	}

	threads = None

	def __init__(self, method_name, dataset_name, threaded_saver=True, num_workers=8):
		self.method_name = method_name
		self.dataset_name = dataset_name

		if threaded_saver:
			self.threads = ThreadPoolExecutor(num_workers)

	def save_output(self, frame, anomaly_p):
		value = anomaly_p.astype(np.float16)

		f = self.channels['anomaly_p'].write
		a = (value,)
		kw = dict(method_name = self.method_name, **frame)

		if self.threads is not None:
			self.threads.submit(f, *a, **kw)
		else:
			f(*a, **kw)

		# self.channels['anomaly_p'].write(value, method_name = self.method_name, **frame)
	
	# def save_frame_list_threaded(self, frame_list, anomaly_key = 'anomaly_p', num_workers=8):
	# 	self.threads = ThreadPoolExecutor(num_workers)
	
	def wait_to_finish_saving(self):
		if self.threads is not None:
			self.threads.shutdown(True)
			self.threads = None




