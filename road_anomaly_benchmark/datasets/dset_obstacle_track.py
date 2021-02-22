
from pathlib import Path
from os import environ
from easydict import EasyDict
from .dataset_registry import DatasetRegistry
from .dataset_io import DatasetBase, ChannelLoaderImage

from .. import DIR_SRC

@DatasetRegistry.register_class()
class DatasetObstacleTrack(DatasetBase):

	configs = [
		dict(
			name = 'RoadObstacleTrack-test',
			dir_root = Path(environ.get('DIR_DATASETS', DIR_SRC / 'datasets')) / 'dataset_RoadObstacleTrack',
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.jpg"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}

	def __init__(self, cfg):
		super().__init__(cfg)

		fids = [p.stem for p in (self.cfg.dir_root / 'images').glob('*.jpg')]
		self.set_frames([EasyDict(fid=fid) for fid in fids])

