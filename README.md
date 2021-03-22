
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
from road_anomaly_benchmark.evaluation import Evaluation

ev = Evaluation(
    method_name = 'Resynthesis',
    dataset_name = 'ObstacleTrack-all',
)

for fr in ev.get_frames():
    anomaly_p = my_method(fr.image)
    ev.save_result(fr, anomaly_p)

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