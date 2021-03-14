

from .dataset_io import DatasetBase, imread
from .dataset_registry import DatasetRegistry

from pathlib import Path
import json, os

import numpy as np

def read_json(path, key=None, allow_failure=False):
	try:
		file_content = json.loads(Path(path).read_text())
		return file_content[key] if key is not None else file_content

	except Exception as e:
		print(f'Failed to load JSON {path}: ', e)

		if allow_failure:
			return []
		else:
			raise e


@DatasetRegistry.register_class()
class FishyscapesLAFSubset(DatasetBase):
	configs = [
		dict(
			name = 'FishyLAF-val',
			split = 'val',
		),
		dict(
			name = 'FishyLAF-test',
			split = 'test',
		),
	]

	FRAME_IDS = {
		'val': [
			"01_Hanns_Klemm_Str_45_000000_000080",
			"01_Hanns_Klemm_Str_45_000000_000140",
			"01_Hanns_Klemm_Str_45_000000_000200",
			"01_Hanns_Klemm_Str_45_000000_000230",
			"01_Hanns_Klemm_Str_45_000000_000260",
			"01_Hanns_Klemm_Str_45_000001_000060",
			"01_Hanns_Klemm_Str_45_000001_000120",
			"01_Hanns_Klemm_Str_45_000001_000180",
			"01_Hanns_Klemm_Str_45_000001_000210",
			#"01_Hanns_Klemm_Str_45_000001_000240", # the LAF labels are invalid
			"01_Hanns_Klemm_Str_45_000002_000070",
			"01_Hanns_Klemm_Str_45_000002_000130",
			"01_Hanns_Klemm_Str_45_000002_000190",
			"01_Hanns_Klemm_Str_45_000002_000220",
			"01_Hanns_Klemm_Str_45_000003_000030",
			"01_Hanns_Klemm_Str_45_000003_000090",
			"01_Hanns_Klemm_Str_45_000003_000150",
			"01_Hanns_Klemm_Str_45_000003_000180",
			"01_Hanns_Klemm_Str_45_000003_000210",
			"01_Hanns_Klemm_Str_45_000004_000120",
			"01_Hanns_Klemm_Str_45_000004_000180",
			"01_Hanns_Klemm_Str_45_000004_000210",
			"01_Hanns_Klemm_Str_45_000005_000080",
			"01_Hanns_Klemm_Str_45_000005_000140",
			"01_Hanns_Klemm_Str_45_000005_000200",
			#"01_Hanns_Klemm_Str_45_000005_000230", # the LAF labels are invalid
			"01_Hanns_Klemm_Str_45_000008_000060",
			"01_Hanns_Klemm_Str_45_000008_000120",
			"01_Hanns_Klemm_Str_45_000008_000180",
			"01_Hanns_Klemm_Str_45_000008_000240",
			"01_Hanns_Klemm_Str_45_000009_000090",
			"01_Hanns_Klemm_Str_45_000009_000150",
			"01_Hanns_Klemm_Str_45_000009_000210",
			"01_Hanns_Klemm_Str_45_000010_000050",
			"01_Hanns_Klemm_Str_45_000010_000110",
			"01_Hanns_Klemm_Str_45_000010_000170",
			"01_Hanns_Klemm_Str_45_000010_000200",
			"01_Hanns_Klemm_Str_45_000011_000050",
			"01_Hanns_Klemm_Str_45_000011_000110",
			"01_Hanns_Klemm_Str_45_000011_000170",
			"01_Hanns_Klemm_Str_45_000011_000200",
			"01_Hanns_Klemm_Str_45_000011_000230",
			"01_Hanns_Klemm_Str_45_000011_000260",
			"01_Hanns_Klemm_Str_45_000012_000160",
			"01_Hanns_Klemm_Str_45_000012_000220",
			"01_Hanns_Klemm_Str_45_000012_000250",
			"01_Hanns_Klemm_Str_45_000012_000280",
			"04_Maurener_Weg_8_000000_000030",
			"04_Maurener_Weg_8_000000_000090",
			"04_Maurener_Weg_8_000000_000150",
			"04_Maurener_Weg_8_000000_000180",
			"04_Maurener_Weg_8_000001_000030",
			"04_Maurener_Weg_8_000001_000090",
			"04_Maurener_Weg_8_000001_000150",
			"04_Maurener_Weg_8_000001_000180",
			"04_Maurener_Weg_8_000001_000210",
			"04_Maurener_Weg_8_000002_000060",
			"04_Maurener_Weg_8_000002_000120",
			"04_Maurener_Weg_8_000002_000150",
			"04_Maurener_Weg_8_000003_000030",
			"04_Maurener_Weg_8_000003_000090",
			"04_Maurener_Weg_8_000003_000120",
			"04_Maurener_Weg_8_000004_000020",
			"04_Maurener_Weg_8_000004_000080",
			"04_Maurener_Weg_8_000004_000140",
			"04_Maurener_Weg_8_000004_000170",
			"04_Maurener_Weg_8_000004_000200",
			"04_Maurener_Weg_8_000005_000050",
			"04_Maurener_Weg_8_000005_000110",
			"04_Maurener_Weg_8_000005_000170",
			"04_Maurener_Weg_8_000005_000200",
			"04_Maurener_Weg_8_000006_000040",
			"04_Maurener_Weg_8_000006_000100",
			"04_Maurener_Weg_8_000006_000130",
			"04_Maurener_Weg_8_000007_000020",
			"04_Maurener_Weg_8_000007_000080",
			"04_Maurener_Weg_8_000007_000140",
			"04_Maurener_Weg_8_000007_000170",
			"04_Maurener_Weg_8_000008_000040",
			"04_Maurener_Weg_8_000008_000100",
			"04_Maurener_Weg_8_000008_000130",
			"04_Maurener_Weg_8_000008_000160",
			"04_Maurener_Weg_8_000008_000190",
			"13_Elly_Beinhorn_Str_000000_000050",
			"13_Elly_Beinhorn_Str_000000_000110",
			"13_Elly_Beinhorn_Str_000000_000170",
			"13_Elly_Beinhorn_Str_000000_000230",
			"13_Elly_Beinhorn_Str_000000_000260",
			"13_Elly_Beinhorn_Str_000000_000290",
			"13_Elly_Beinhorn_Str_000001_000080",
			"13_Elly_Beinhorn_Str_000001_000140",
			"13_Elly_Beinhorn_Str_000001_000200",
			"13_Elly_Beinhorn_Str_000001_000230",
			"13_Elly_Beinhorn_Str_000002_000030",
			"13_Elly_Beinhorn_Str_000002_000090",
			"13_Elly_Beinhorn_Str_000002_000120",
			"13_Elly_Beinhorn_Str_000003_000080",
			"13_Elly_Beinhorn_Str_000003_000140",
			"13_Elly_Beinhorn_Str_000003_000200",
			"13_Elly_Beinhorn_Str_000003_000260",
		],
		'test': read_json(
			os.environ.get('FISHY_LAF_ID_FILE', Path('~/FishyLAF-ids.json').expanduser()), 
			'test', 
			allow_failure=True,
		),
	}

	for fidlist in FRAME_IDS.values():
		fidlist.sort()

	FRAME_IDS_SETS = {
		# split: {
		# 	fid: i for i, fid in enumerate(fids)
		# }	
		split: set(fids)
		for split, fids in FRAME_IDS.items()
	}

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()
		
	@property
	def fids(self):
		return self.FRAME_IDS[self.cfg.split]

	def discover(self):
		self.laf_dsets = {
			'train': DatasetRegistry.get('LostAndFound-train'),
			'test': DatasetRegistry.get('LostAndFound-test'),
		}

	def get_frame(self, idx_or_fid, *channels):
		if isinstance(idx_or_fid, str): # by fid
			if idx_or_fid not in FRAME_IDS_SETS:
				raise KeyError('Id {idx_or_fid} is not in this dataset')
			else:
				fid = idx_or_fid
		else:
			fid = self.FRAME_IDS[self.cfg.split][idx_or_fid]

		if fid in self.laf_dsets['train'].frames_by_fid:
			base_ds = self.laf_dsets['train']
		else:
			base_ds = self.laf_dsets['test']

		fr = base_ds.get_frame(fid, *channels)

		return fr

	def __len__(self):
		return self.fids.__len__()


@DatasetRegistry.register_class()
class LostAndFoundAnomaly(DatasetBase):
	"""
	LAF obstacles, but the ROI is the whole image 
	(except ego vehicle and sensor artifacts)
	not just the drivable space.
	"""
	configs = [
		dict(
			name = 'LostAndFound-anomalyTrain',
			base_ds = 'LostAndFound-train',
			roi_img = Path(__file__).with_name('LAF_roi_2048.png'),
		),
		dict(
			name = 'LostAndFound-anomalyTest',
			base_ds = 'LostAndFound-test',
			roi_img = Path(__file__).with_name('LAF_roi_2048.png'),
		),
	]

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()
		
	def discover(self):
		self.base_ds = DatasetRegistry.get(self.cfg.base_ds)
		self.roi_mask = imread(self.cfg.roi_img).astype(bool)

	def get_frame(self, idx_or_fid, *channels):
		fr = self.base_ds.get_frame(idx_or_fid, *channels)

		pix_gt = fr.get('label_pixel_gt')
		if pix_gt is not None:
			h, w = pix_gt.shape[:2]
			label = np.full((h, w), 255, dtype=np.uint8)
			label[self.roi_mask] = 0
			label[pix_gt == 1] = 1
			fr['label_pixel_gt'] = label

		return fr

	def __len__(self):
		return self.base_ds.__len__()



@DatasetRegistry.register_class()
class ROSubset(DatasetBase):
	configs = [
		dict(
			name = 'RO-20',
			split = 'RO20',
		),
		dict(
			name = 'RO-20nodog',
			split = 'RO20-no-dog',
		),

		dict(
			name = 'RO-21',
			split = 'RO21',
		),
	]

	FRAME_IDS = {
		'RO20-no-dog': [
			"darkasphalt_axestump_2",
			"darkasphalt_axestump_3",
			"darkasphalt_bag_3",
			"darkasphalt_basket_2",
			"darkasphalt_basket_3",
			"darkasphalt_boot_2",
			"darkasphalt_boot_3",
			"darkasphalt_bottles_1",
			"darkasphalt_bottles_2",
			"darkasphalt_bucket_3",
			"darkasphalt_bucket_4",
			"darkasphalt_canister_2",
			"darkasphalt_canister_3",
			"darkasphalt_cans_1",
			"darkasphalt_cans_2",
			"darkasphalt_helmetO_1",
			"darkasphalt_helmetO_2",
			"darkasphalt_stump_1",
			"darkasphalt_stump_2",
			"darkasphalt_watercanS_2",
			"gravel_axestump_2",
			"gravel_axestump_3",
			"gravel_bag_3",
			"gravel_basket_2",
			"gravel_basket_3",
			"gravel_boot_1",
			"gravel_boot_2",
			"gravel_bottle_2",
			"gravel_bottle_3",
			"gravel_bottle_4",
			"gravel_bucket_3",
			"gravel_canister_3",
			"gravel_canister_4",
			"gravel_cans2_2",
			"gravel_helmetG_2",
			"gravel_log_2",
			"gravel_log_3",
			"gravel_stump_1",
			"gravel_stump_2",
			"gravel_watercanS_2",
			"greyasphalt_axestump_2",
			"greyasphalt_axestump_3",
			"greyasphalt_bag_3",
			"greyasphalt_bag_4",
			"greyasphalt_basket_3",
			"greyasphalt_basket_4",
			"greyasphalt_boot_2",
			"greyasphalt_boot_3",
			"greyasphalt_bottle_1",
			"greyasphalt_bottle_2",
			"greyasphalt_canister_2",
			"greyasphalt_canister_3",
			"greyasphalt_cansA_1",
			"greyasphalt_cansB_2",
			"greyasphalt_helmetB_1",
			"greyasphalt_helmetB_2",
			"greyasphalt_stump_2",
			"greyasphalt_stump_3",
			"greyasphalt_watercanB_4",
			"greyasphalt_watercanS_2",
			"motorway_axestump_2",
			"motorway_basket_1",
			"motorway_basket_4",
			"motorway_boot_1",
			"motorway_boot_3",
			"motorway_boot_6_and_bird",
			"motorway_bottles_1",
			"motorway_bottles_k",
			"motorway_bucket_3",
			"motorway_canister_3",
			"motorway_cans_1",
			"motorway_helmetB_1",
			"motorway_helmetB_3",
			"motorway_helmetO_1",
			"motorway_stone_1",
			"motorway_stone_2",
			"motorway_stump_2",
			"motorway_stump_3",
			"motorway_triangle_2",
			"motorway_watercanS_1",
			"paving_axe_2",
			"paving_axestump_2",
			"paving_bag_3",
			"paving_bag_6",
			"paving_basket_3",
			"paving_basket_5",
			"paving_boot_1",
			"paving_boot_2",
			"paving_bottles_1",
			"paving_bottles_2",
			"paving_bucket_3",
			"paving_bucket_6",
			"paving_canister_4",
			"paving_cans_2",
			"paving_helmetB_1",
			"paving_helmetB_2",
			"paving_stump_2",
			"paving_watercanS_2",
			"paving_wood_2",
			"paving_wood_3",
		],
		'RO20-dog': [
			"darkasphalt2_dog_1",
			"darkasphalt2_dog_2",
			"darkasphalt2_dog_3",
			"darkasphalt2_dog_4",
			"darkasphalt2_dog_5",
		],
		'RO21-snowstorm': [
			'snowstorm1_00_16_02.578',
			'snowstorm1_00_14_05.595',
			'snowstorm1_00_14_37.727',
			'snowstorm1_00_03_05.185',
			'snowstorm2_00_02_32.753',
			'snowstorm2_00_06_00.260',
			'snowstorm1_00_11_28.972',
			'snowstorm1_00_15_56.322',
			'snowstorm1_00_00_56.106',
			'snowstorm1_00_10_55.939',
			'snowstorm2_00_04_07.881',
			'snowstorm1_00_00_51.868',
			'snowstorm2_00_02_30.667',
			'snowstorm1_00_15_20.970',
			'snowstorm2_00_01_39.950',
			'snowstorm1_00_13_15.261',
			'snowstorm1_00_07_41.845',
			'snowstorm1_00_03_24.738',
			'snowstorm2_00_04_46.303',
			'snowstorm2_00_05_55.972',
			'snowstorm1_00_08_26.106',
			'snowstorm1_00_14_02.892',
			'snowstorm2_00_00_49.600',
			'snowstorm1_00_16_00.827',
			'snowstorm2_00_05_47.380',
			'snowstorm1_00_13_18.748',
			'snowstorm2_00_06_46.423',
			'snowstorm1_00_07_43.930',
			'snowstorm2_00_06_49.910',
			'snowstorm1_00_10_52.886',
			'snowstorm1_00_15_24.040',
			'snowstorm1_00_14_40.363',
			'snowstorm1_00_10_48.898',
			'snowstorm1_00_03_08.939',
			'snowstorm1_00_08_21.968',
			'snowstorm2_00_01_35.045',
			'snowstorm2_00_03_09.556',
			'snowstorm2_00_02_27.047',
			'snowstorm2_00_04_52.192',
			'snowstorm1_00_08_18.898',
			'snowstorm2_00_04_14.087',
			'snowstorm2_00_07_27.397',
			'snowstorm1_00_10_06.489',
			'snowstorm2_00_03_13.443',
			'snowstorm1_00_11_33.009',
			'snowstorm2_00_04_12.152',
			'snowstorm2_00_00_52.502',
			'snowstorm2_00_06_54.030',
			'snowstorm2_00_04_49.172',
			'snowstorm2_00_01_38.181',
			'snowstorm1_00_09_22.178',
			'snowstorm1_00_15_26.042',
			'snowstorm2_00_06_52.545',
			'snowstorm2_00_05_53.503',
			'snowstorm1_00_00_59.226',
		],
	}

	FRAME_IDS['RO20'] = FRAME_IDS['RO20-no-dog'] + FRAME_IDS['RO20-dog']
	FRAME_IDS['RO21'] = FRAME_IDS['RO20'] + FRAME_IDS['RO21-snowstorm']

	for fidlist in FRAME_IDS.values():
		fidlist.sort()

	FRAME_IDS_SETS = {
		# split: {
		# 	fid: i for i, fid in enumerate(fids)
		# }	
		split: set(fids)
		for split, fids in FRAME_IDS.items()
	}

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()
		
	@property
	def fids(self):
		return self.FRAME_IDS[self.cfg.split]

	def discover(self):
		self.base_ds = DatasetRegistry.get('ObstacleTrack-test')

	def get_frame(self, idx_or_fid, *channels):
		if isinstance(idx_or_fid, str): # by fid
			if idx_or_fid not in FRAME_IDS_SETS:
				raise KeyError('Id {idx_or_fid} is not in this dataset')
			else:
				fid = idx_or_fid
		else:
			fid = self.FRAME_IDS[self.cfg.split][idx_or_fid]

		fr = self.base_ds.get_frame(fid, *channels)
		return fr

	def __len__(self):
		return self.fids.__len__()
