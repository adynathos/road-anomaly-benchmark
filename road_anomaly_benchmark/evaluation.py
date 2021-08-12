
from pathlib import Path
from os import environ
from functools import partial
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
from tqdm import tqdm

from .paths import DIR_SRC, DIR_OUTPUTS
from .datasets import DatasetRegistry
from .datasets.dataset_io import ChannelLoaderHDF5
from .metrics import MetricRegistry


try:
	import cv2 as cv
except:
	...

log = logging.getLogger(__name__)



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

	def get_frames(self):
		return self.get_dataset().iter('image')

	def __len__(self):
		return self.get_dataset().__len__()

	@staticmethod
	def write_task(channel, value, extra):
		try:
			channel.write(value, **extra)
		except Exception as e:
			log.exception('In writing result')
			raise e


	def save_output(self, frame, anomaly_p):
		value = anomaly_p.astype(np.float16)

		write_func = partial(
			self.write_task, 
			self.channels['anomaly_p'], 
			value, 
			dict(method_name = self.method_name, **frame),
		)

		if self.threads is not None:
			self.threads.submit(write_func)
		else:
			write_func()

		# self.channels['anomaly_p'].write(value, method_name = self.method_name, **frame)
	

	def run_metric_single(self, metric_name, sample=None, frame_vis=False):
		# TODO sample is part of evaluation

		metric = MetricRegistry.get(metric_name)
		try:
			if "Seg" in metric_name:
				if metric.cfg.thresh_p is None:
					metric.get_thresh_p_from_curve(self.method_name, self.dataset_name)
		except AttributeError:
			print("Perform 'PixBinaryClass' first")
			exit()

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
				visualize = frame_vis,
				**fr,
			)
			fr_results.append(fr_result)

		return metric.aggregate(
			fr_results,
			method_name = self.method_name,
			dataset_name = self.dataset_name,
		)

	@classmethod
	def metric_worker(cls, method_name, metric_name, frame_vis, dataset_name_and_frame_idx):
		try:
			dataset_name, frame_idx = dataset_name_and_frame_idx

			dset = DatasetRegistry.get(dataset_name)
			metric = MetricRegistry.get(metric_name)
			
			metric.init(method_name, dataset_name)

			fr = dset[frame_idx]

			heatmap = cls.channels['anomaly_p'].read(
				method_name = method_name, 
				dset_name = fr.dset_name,
				fid = fr.fid,
			)

			if heatmap.shape[1] < fr.image.shape[1]:
				heatmap = cv.resize(heatmap.astype(np.float32), fr.image.shape[:2][::-1], interpolation=cv.INTER_LINEAR)

			result = metric.process_frame(
				anomaly_p = heatmap,
				method_name = method_name, 
				visualize = frame_vis,
				**fr,
			)

			return result
		except Exception as e:
			log.exception(f'Metric worker {e}')
			raise e


	def run_metric_parallel(self, metric_name, sample=None, frame_vis=False):

		metric = MetricRegistry.get(metric_name)
		try:
			if "Seg" in metric_name:
				if metric.cfg.thresh_p is None:
					metric.get_thresh_p_from_curve(self.method_name, self.dataset_name)
		except AttributeError:
			print("Perform 'PixBinaryClass' first")
			exit()
		
		if sample is not None:
			dset_name, frame_indices = sample
		else:
			dset_name = self.dataset_name
			frame_indices = range(self.get_dataset().__len__())

		tasks = [
			(dset_name, idx)
			for idx in frame_indices
		]

		with multiprocessing.Pool() as pool:
			it = pool.imap_unordered(
				partial(self.metric_worker, self.method_name, metric_name, frame_vis),
				tasks,
				chunksize = 4,
			)
			
			processed_frames = list(tqdm(it, total=tasks.__len__()))

		ag = metric.aggregate(	
			processed_frames,
			method_name = self.method_name,
			dataset_name = dset_name,
		)

		return ag
	

	def calculate_metric_from_saved_outputs(self, metric_name, sample=None, parallel=True, show_plot=False, frame_vis=False):

		metric = MetricRegistry.get(metric_name)

		if parallel:
			ag = self.run_metric_parallel(metric_name, sample=sample, frame_vis=frame_vis)
		else:
			ag = self.run_metric_single(metric_name, sample=sample, frame_vis=frame_vis)

		dset_name = sample[0] if sample is not None else self.dataset_name

		metric.save(
			ag, 
			method_name = self.method_name,
			dataset_name = dset_name,
		)

		metric.plot_single(ag, close = not show_plot)

		return ag


	def wait_to_finish_saving(self):
		if self.threads is not None:
			self.threads.shutdown(True)
			self.threads = None




