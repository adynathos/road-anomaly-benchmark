
from pathlib import Path
from os import environ

from easydict import EasyDict
import numpy as np

from .dataset_registry import DatasetRegistry
from .dataset_io import DatasetBase, ChannelLoaderImage

from .. import DIR_SRC


class DatasetRA(DatasetBase):

	def __init__(self, cfg):
		super().__init__(cfg)

		fids = [p.stem for p in (self.cfg.dir_root / 'images').glob('*.jpg')]
		self.set_frames([EasyDict(fid=fid) for fid in fids])

	def __getitem__(self, key):
		"""

		"""
		fr = super().__getitem__(key)

		sem_gt = fr.get('semantic_class_gt')
		if sem_gt is not None:
			h, w = sem_gt.shape[:2]
			label = np.full((h, w), 255, dtype=np.uint8)
			label[sem_gt == self.cfg.classes.usual] = 0
			label[sem_gt == self.cfg.classes.anomaly] = 1

			fr['label_pixel_gt'] = label

		return fr


@DatasetRegistry.register_class()
class DatasetAnomalyTrack(DatasetRA):

	configs = [
		dict(
			name = 'RoadAnomalyTrack-test',
			dir_root = Path(environ.get('DIR_DATASETS', DIR_SRC / 'datasets')) / 'dataset_RoadAnomalyTrack',
			classes = dict(
				usual = 0,
				anomaly = 1,
				ignore = 2,
			),
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.jpg"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_final/{fid}_labels_semantic.png"),
	}




@DatasetRegistry.register_class()
class DatasetObstacleTrack(DatasetRA):

	configs = [
		dict(
			name = 'RoadObstacleTrack-test',
			dir_root = Path(environ.get('DIR_DATASETS', DIR_SRC / 'datasets')) / 'dataset_RoadObstacleTrack',
			classes = dict(
				road = 0,
				obstacle = 254,
				ignore = 253,

				usual = 0,
				anomaly = 254,
			),
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.jpg"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}


@DatasetRegistry.register_class()
class DatasetWeather(DatasetRA):

	configs = [
		dict(
			name = 'RoadObstacleWeather-v1',
			dir_root = Path(environ.get('DIR_DATASETS', DIR_SRC / 'datasets')) / 'dataset_RoadObstacleWeather_v1',
			# classes = dict(
			# 	road = 253,
			# 	obstacle = 254,
			# 	ignore = 0,
			# )
		),
		dict(
			name = 'RoadObstacleExtra-v1',
			dir_root = Path(environ.get('DIR_DATASETS', DIR_SRC / 'datasets')) / 'dataset_RoadObstacleExtra',
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.jpg"),
		#'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}


