
from pathlib import Path
from os import environ
from operator import itemgetter
import logging, re

from easydict import EasyDict
import numpy as np

from ..paths import DIR_DATASETS
from .dataset_registry import DatasetRegistry
from .dataset_io import DatasetBase, ChannelLoaderImage


log = logging.getLogger(__name__)

class DatasetRA(DatasetBase):

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()

	def discover(self):
		""" Discover frames in file system """
		path_template = Path(self.channels['image'].resolve_template(
			dset = self,
			fid = '*',
		))
		print(path_template, path_template.parent, path_template.name)
		fids = [p.stem for p in path_template.parent.glob(path_template.name)]
		fids.sort()
		self.set_frames([EasyDict(fid=fid) for fid in fids])
		self.check_size()


	def check_size(self):
		desired_len = self.cfg.get('expected_length')
		actual_len = self.__len__()

		if desired_len is not None and actual_len != desired_len:
			raise ValueError(f'The dataset should have {desired_len} frames but found {actual_len}')

	def __getitem__(self, key):
		"""

		"""
		fr = super().__getitem__(key)

		sem_gt = fr.get('semantic_class_gt')
		if sem_gt is not None:
			h, w = sem_gt.shape[:2]
			label = np.full((h, w), 255, dtype=np.uint8)
			label[sem_gt == self.cfg.classes.usual] = 0

			anomaly = self.cfg.classes.anomaly
			if isinstance(anomaly, (tuple, list)) and anomaly.__len__() == 2:
				range_low, range_high = anomaly
				anomaly_mask = (range_low <= sem_gt) & (sem_gt <= range_high)
			else:
				anomaly_mask = sem_gt == anomaly

			label[anomaly_mask] = 1

			fr['label_pixel_gt'] = label

		return fr


@DatasetRegistry.register_class()
class DatasetAnomalyTrack(DatasetRA):

	configs = [
		dict(
			name = 'AnomalyTrack-test',
			dir_root = DIR_DATASETS / 'dataset_AnomalyTrack',
			img_fmt = 'jpg',
			classes = dict(
				usual = 0,
				anomaly = 1,
				ignore = 255,
			),
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}


@DatasetRegistry.register_class()
class DatasetObstacleTrack(DatasetRA):

	configs = [
		dict(
			name = 'ObstacleTrack-test',
			dir_root = DIR_DATASETS / 'dataset_ObstacleTrack',
			img_fmt = 'webp',
			classes = dict(
				road = 0,
				obstacle = 1,
				ignore = 255,

				usual = 0,
				anomaly = 1,
			),
		),
		dict(
			name = 'RoadObstacleTrack-test',
			dir_root = DIR_DATASETS / 'dataset_RoadObstacleTrack',
			img_fmt = 'webp',
			classes = dict(
				road = 0,
				obstacle = 1,
				ignore = 255,

				usual = 0,
				anomaly = 1,
			),
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}


@DatasetRegistry.register_class()
class DatasetWeather(DatasetRA):

	configs = [
		dict(
			name = 'RoadObstacleWeather-v1',
			dir_root = DIR_DATASETS / 'dataset_RoadObstacleWeather_v1',
			# classes = dict(
			# 	road = 253,
			# 	obstacle = 254,
			# 	ignore = 0,
			# )
		),
		dict(
			name = 'RoadObstacleExtra-v1',
			dir_root = DIR_DATASETS / 'dataset_RoadObstacleExtra',
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.jpg"),
		#'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}

@DatasetRegistry.register_class()
class DatasetLostAndFound(DatasetRA):
	"""
	https://github.com/mcordts/cityscapesScripts#dataset-structure
	"""

	DIR_LAF = Path(environ.get('DIR_LAF', DIR_DATASETS / 'dataset_LostAndFound'))

	LAF_CLASSES = dict(
		ignore = 0,
		usual = 1, # road
		anomaly = [2, 200], # range
	)

	configs = [
		dict(
			name = 'LostAndFound-train',
			split = 'train',
			dir_root = DIR_LAF,

			# invalid frames are those where np.count_nonzero(labels_source) is 0
			invalid_labeled_frames = [44,  67,  88, 109, 131, 614],
			expected_length = 1030,

			classes = LAF_CLASSES,
		),
		dict(
			name = 'LostAndFound-test',
			split = 'test',
			dir_root = DIR_LAF,

			# invalid frames are those where np.count_nonzero(labels_source) is 0
			invalid_labeled_frames = [17,  37,  55,  72,  91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793],
			expected_length = 1186,

			classes = LAF_CLASSES,
		),
	]

	channels = {
		'image': ChannelLoaderImage(
			'{dset.cfg.dir_root}/leftImg8bit/{dset.cfg.split}/{scene_id:02d}_{scene_name}/{fid}_leftImg8bit.{dset.img_fmt}',
		),
		'semantic_class_gt': ChannelLoaderImage(
			'{dset.cfg.dir_root}/gtCoarse/{dset.cfg.split}/{scene_id:02d}_{scene_name}/{fid}_gtCoarse_labelIds.png',
		),
		'instances': ChannelLoaderImage(
			'{dset.cfg.dir_root}/gtCoarse/{dset.cfg.split}/{scene_id:02d}_{scene_name}/{fid}_gtCoarse_instanceIds.png',
		),
	}

	RE_LAF_NAME = re.compile(r'([0-9]{2})_(.*)_([0-9]{6})_([0-9]{6})')
	LAF_SUFFIX_LEN = '_leftImg8bit'.__len__()

	@classmethod
	def laf_id_from_image_path(cls, path, **_):
		fid = path.stem[:-cls.LAF_SUFFIX_LEN]

		m = cls.RE_LAF_NAME.match(fid)

		return EasyDict(
			fid = fid,
			scene_id = int(m.group(1)),
			scene_name = m.group(2),
			scene_seq = int(m.group(3)),
			scene_time = int(m.group(4))
		)


	def discover(self):
		img_dir = Path(self.cfg.dir_root) / 'leftImg8bit' / self.cfg.split

		for img_ext in ['png', 'webp', 'jpg']:
			img_files = list(img_dir.glob(f'*/*_leftImg8bit.{img_ext}'))
			if img_files:
				break

		if not img_files:
			raise FileNotFoundError(f'Did not find images at {img_dir}')


		log.info(f'LAF: found images in {img_ext} format')
		self.img_fmt = img_ext

		# LAF's PNG images contain a gamma value which makes them washed out, ignore it
		# if img_ext == '.png':
			# self.channels['image'].opts['ignoregamma'] = True

		frames = [
			self.laf_id_from_image_path(p)
			for p in img_files
		]
		frames.sort(key = itemgetter('fid'))

		# remove invalid labeled frames
		invalid_indices = self.cfg.invalid_labeled_frames
		valid_indices = np.delete(np.arange(frames.__len__()), invalid_indices)
		#print('\n '.join([frames[i].fid for i in invalid_indices]))
		frames = [frames[i] for i in valid_indices]

		self.set_frames(frames)
		self.check_size()


@DatasetRegistry.register_class()
class DatasetSmallObstacle(DatasetRA):
	configs = [
		dict(
			name='SmallObstacleDataset-train',
			split='train',
			dir_root='/home/datasets/Small_Obstacle_Dataset',
			classes=dict(
				road=1,
				obstacle=2,
				ignore=0,
			),
		),
		dict(
			name='SmallObstacleDataset-test',
			split='test',
			dir_root='/home/datasets/Small_Obstacle_Dataset',
			classes=dict(
				road=1,
				obstacle=2,
				ignore=0,
			),
		),
		dict(
			name='SmallObstacleDataset-val',
			split='val',
			dir_root='/home/datasets/Small_Obstacle_Dataset',
			classes=dict(
				road=1,
				obstacle=2,
				ignore=0,
			),
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/{dset.cfg.split}/{direc}/image/{fid}.png"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/{dset.cfg.split}/{direc}/labels/{fid}.png"),
	}

	def sod_id_from_image_path(cls, path, **_):
		fid = path.stem
		direc = str(fid).split('_')[0] + '_' + str(fid).split('_')[1]
		return EasyDict(
			fid=fid,
			direc=direc
		)

	def discover(self):
		img_dir = Path(self.cfg.dir_root) / self.cfg.split

		for img_ext in ['png', 'webp', 'jpg']:
			img_files = list(img_dir.glob(f'*/labels/*.{img_ext}'))
			if img_files:
				break

		if not img_files:
			raise FileNotFoundError(f'Did not find images at {img_dir}')

		log.info(f'SOD: found images in {img_ext} format')
		self.img_fmt = img_ext

		frames = [
			self.sod_id_from_image_path(p)
			for p in img_files
		]
		frames.sort(key=itemgetter('fid'))

		self.set_frames(frames)
		self.check_size()