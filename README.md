
## Datasets

* Fishyscapes - Lost & Found - anomalies
* Fishyscapes - Lost & Found - obstacles
* Fishyscapes - Web - anomalies

* [Road Anomaly](doc/RoadAnomaly.md) - big anomalies, mostly animals, from ICCV2019 article
* [Road Obstacles V1](https://arxiv.org/abs/2012.13633) - road scenes with obstacles, on diverse road surfaces.
* [Road Obstacles V2](doc/RoadObstaclesV2.md) - additional recordings of obstacles, in difficult weather and at night.


## Evaluation procedure

### Inference

* Place the datasets in `./datasets` (or override with env var `DIR_DATASETS`)
	* `dataset_ObstacleTrack`
	* `dataset_RoadAnomalyTrack`
	* `dataset_LostAndFound` (or provide location in env `DSET_LAF`)
	* `dataset_FishyLAF` (or provide location in env `DSET_FISHY_LAF`)

* Run inference and store results in files. Run inference for the following splits, as the other splits are subsets of those:
	* `RoadAnomalyTrack-test`
	* `ObstacleTrack-all`
	* `LostAndFound-test`
	* `LostAndFound-train`

```python
import numpy as np
from tqdm import tqdm
import cv2 as cv
from road_anomaly_benchmark.evaluation import Evaluation

def method_dummy(image, **_):
	""" Very naive method: return color saturation """
	image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)
	anomaly_p = image_hsv[:, :, 1].astype(np.float32) * (1./255.)
	return anomaly_p


def main():

	ev = Evaluation(
		method_name = 'Dummy', 
		dataset_name = 'ObstacleTrack-all',
		# dataset_name = 'AnomalyTrack-test',
	)

	for frame in tqdm(ev.get_frames()):
		# run method here
		result = method_dummy(frame.image)
		# provide the output for saving
		ev.save_output(frame, result)

	# wait for the background threads which are saving
	ev.wait_to_finish_saving()
```

The files will be stored in `./outputs/anomaly_p/...`. The storage directory can be overriden with env var `DIR_OUTPUTS`.

### Metrics

This step will also create plots in `./outputs/{metric}`

* Metrics for anomaly track, splits *RoadAnomalyTrack-test, FishyLAFAnomaly-val*

```bash
methods=Method1,Method2

python -m road_anomaly_benchmark metric PixBinaryClass $methods RoadAnomalyTrack-test,FishyLAFAnomaly-val
python -m road_anomaly_benchmark metric SegEval-AnomalyTrack $methods RoadAnomalyTrack-test,FishyLAFAnomaly-val
```

* Metrics for obstacle track, splits: *ObstacleTrack-test, LostAndFound-testNoKnown*

```bash
methods=Method1,Method2

python -m road_anomaly_benchmark metric PixBinaryClass $methods ObstacleTrack-test,LostAndFound-testNoKnown
python -m road_anomaly_benchmark metric SegEval-ObstacleTrack $methods ObstacleTrack-test,LostAndFound-testNoKnown
```

* Upload the metric files from `./outputs/{metric_name}` to `outputs/{metric_name}` in the network disk.

### Plots and tables

```bash
python -m road_anomaly_benchmark comparison MyComparison metric1,metric2 method1,method2 dset1,dset2
```

* Anomaly splits: *RoadAnomalyTrack-test, FishyLAFAnomaly-val*

```bash
# Anomaly tables
python -m road_anomaly_benchmark comparison TableAnomaly1 PixBinaryClass,SegEval-AnomalyTrack $methods_ano RoadAnomalyTrack-test --names names.json
python -m road_anomaly_benchmark comparison TableAnomaly2 PixBinaryClass,SegEval-AnomalyTrack $methods_ano FishyLAFAnomaly-val --names names.json
```

* Obstacle splits: *ObstacleTrack-test, LostAndFound-testNoKnown*

```bash
# Obstacle tables
python -m road_anomaly_benchmark comparison TableObstacle1 PixBinaryClass,SegEval-ObstacleTrack $methods_obs ObstacleTrack-test  --names names.json
python -m road_anomaly_benchmark comparison TableObstacle2 PixBinaryClass,SegEval-ObstacleTrack $methods_obs LostAndFound-testNoKnown  --names names.json
```

## Splits

### Obstacle scene splits

Scene loaders:

* ObstacleScene-curvy - cracked road, surrounded by snow
* ObstacleScene-gravel - gravel road, no snow
* ObstacleScene-greyasphalt - grey roads, in village and in forest
* ObstacleScene-motorway - motorway with side railings
* ObstacleScene-shiny - sun reflects off wet road
* ObstacleScene-paving - road made of bricks
* ObstacleScene-darkasphaltAll - asphalt after rain, with some autumn leaves, combines 2 locations (below)
* ObstacleScene-darkasphalt - asphalt after rain, with some autumn leaves, dog sequence excluded
* ObstacleScene-darkasphaltDog - asphalt after rain, with some autumn leaves, dog sequence only
* ObstacleScene-night - evening or night illuminated with car lamps
* ObstacleScene-snowstorm - snow falling, camera lens may be dirty

These splits re-use outputs inferred for the `ObstacleTrack-all` loader. Calculating metrics:

```bash
ds_scenes=ObstacleScene-curvy,ObstacleScene-darkasphalt,ObstacleScene-darkasphaltDog,ObstacleScene-darkasphaltAll,ObstacleScene-gravel,ObstacleScene-greyasphalt,ObstacleScene-motorway,ObstacleScene-shiny,ObstacleScene-paving,ObstacleScene-night,ObstacleScene-snowstorm

python -m road_anomaly_benchmark metric PixBinaryClass $methods $ds_scenes
python -m road_anomaly_benchmark metric SegEval-ObstacleTrack $methods $ds_scenes

python -m road_anomaly_benchmark comparison TableObstacleScenes PixBinaryClass,SegEval-ObstacleTrack $ds_scenes   --names names.json
```

## Implementing a metric

A metric should implement the `EvaluationMetric` interface from ([road_anomaly_benchmark/metrics/base.py](road_anomaly_benchmark/metrics/base.py).
See `MetricPixelClassification` from [road_anomaly_benchmark/metrics/pixel_classification.py](road_anomaly_benchmark/metrics/pixel_classification.py).




### Calculate segment metrics

* Anomaly Track
```bash
python -m road_anomaly_benchmark metric SegEval-AnomalyTrack <method> RoadAnomalyTrack-test
```
* Obstacle Track
```bash
python -m road_anomaly_benchmark metric SegEval-ObstacleTrack <method> RoadObstacleTrack-test
```