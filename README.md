
## Datasets

* [RoadAnomaly21](https://uni-wuppertal.sciebo.de/s/TVR7VxukVrV7fUH/download)
* [RoadObstacle21](https://uni-wuppertal.sciebo.de/s/wQQq2saipS339QA/download)
* [Fishyscapes LostAndFound](https://fishyscapes.com/) - Anomalies
* [LostAndFound](http://www.6d-vision.com/lostandfounddataset) - Obstacles

## Evaluation procedure

### Inference

* Place the datasets in `./datasets` (or override with env var `DIR_DATASETS`)
	* `dataset_ObstacleTrack`
	* `dataset_AnomalyTrack`
	* `dataset_LostAndFound` (or provide location in env `DSET_LAF`)
	* `dataset_FishyLAF` (or provide location in env `DSET_FISHY_LAF`)

* Run inference and store results in files. Run inference for the following splits, as the other splits are subsets of those:
	* `AnomalyTrack-all`
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
There are also some methods already implemented and available in `some_methods_inference_public.py`. 

### Metrics

This step will also create plots in `./outputs/{metric}`

* Metrics for anomaly track, splits *AnomalyTrack-validation, FishyLAFAnomaly-val*

```bash
methods=Method1,Method2

python -m road_anomaly_benchmark metric PixBinaryClass $methods AnomalyTrack-validation,FishyLAFAnomaly-val
python -m road_anomaly_benchmark metric SegEval-AnomalyTrack $methods AnomalyTrack-validation,FishyLAFAnomaly-val
```

* Metrics for obstacle track, splits *ObstacleTrack-validation, LostAndFound-testNoKnown*

```bash
methods=Method1,Method2

python -m road_anomaly_benchmark metric PixBinaryClass $methods ObstacleTrack-validation,LostAndFound-testNoKnown
python -m road_anomaly_benchmark metric SegEval-ObstacleTrack $methods ObstacleTrack-validation,LostAndFound-testNoKnown
```

Use `--frame-vis` option to visualize anomaly scores (requires ground truth).

### Plots and Tables

```bash
python -m road_anomaly_benchmark comparison MyComparison metric1,metric2 method1,method2 dset1,dset2
```

* Anomaly splits: *AnomalyTrack-validation, FishyLAFAnomaly-val*

```bash
# Anomaly tables
python -m road_anomaly_benchmark comparison TableAnomaly1 PixBinaryClass,SegEval-AnomalyTrack $methods_ano AnomalyTrack-validation --names names.json
python -m road_anomaly_benchmark comparison TableAnomaly2 PixBinaryClass,SegEval-AnomalyTrack $methods_ano FishyLAFAnomaly-val --names names.json
```

* Obstacle splits: *ObstacleTrack-validation, LostAndFound-testNoKnown*

```bash
# Obstacle tables
python -m road_anomaly_benchmark comparison TableObstacle1 PixBinaryClass,SegEval-ObstacleTrack $methods_obs ObstacleTrack-validation --names names.json
python -m road_anomaly_benchmark comparison TableObstacle2 PixBinaryClass,SegEval-ObstacleTrack $methods_obs LostAndFound-testNoKnown --names names.json
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

