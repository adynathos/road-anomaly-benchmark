
from pathlib import Path
from os import environ

environ["DIR_DATASETS"] = "/home/chan/PycharmProjects/SegmentMeIfYouCan/datasets"
environ["DIR_LAF"] = "/home/datasets/lost_and_found/"
environ["DIR_FISHY_LAF"] = "/home/uhlemeyer/dataset_FishyLAF/"

DIR_SRC = Path(__file__).absolute().parents[1]
DIR_OUTPUTS = Path(environ.get('DIR_OUTPUTS', DIR_SRC / 'outputs'))
DIR_DATASETS = Path(environ.get('DIR_DATASETS', DIR_SRC / 'datasets'))
