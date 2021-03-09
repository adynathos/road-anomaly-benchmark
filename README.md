
### Datasets

* Fishyscapes - Lost & Found - anomalies
* Fishyscapes - Lost & Found - obstacles
* Fishyscapes - Web - anomalies

* [Road Anomaly](doc/RoadAnomaly.md) - big anomalies, mostly animals, from ICCV2019 article
* [Road Obstacles V1](https://arxiv.org/abs/2012.13633) - road scenes with obstacles, on diverse road surfaces.
* [Road Obstacles V2](doc/RoadObstaclesV2.md) - additional recordings of obstacles, in difficult weather and at night.


### Evaluation procedure

* Run inference and store results in files

```python
from road_anomaly_benchmark.evaluation import Evaluation

ev = Evaluation(
	method_name = 'Resynthesis', 
	datasets = ['LostAndFound-test'], 
)

for fr in ev.get_frames():
	anomaly_p = my_method(fr.image)
	ev.save_result(fr, anomaly_p)

ev.wait_to_finish_saving()
```

The files will be stored in `./outputs/anomaly_p/...`

* Calculate metrics  
This step will also create plots in `./outputs/{metric}/plot`

```bash
python -m road_anomaly_benchmark metric PixBinaryClass Resynth2048Orig LostAndFound-test
```

* Plots and tables

```bash
python -m road_anomaly_benchmark comparison MyComparison metric1,metric2 method1,method2 dset1,dset2
```
For example
```
python -m road_anomaly_benchmark comparison LAF1 PixBinaryClass Resynth2048Orig,Min_softmax LostAndFound-test,LostAndFound-train
```

for example

```bash
python -m road_anomaly_benchmark comparison MyComparison PixBinaryClass Resynth2048Orig,Min_softmax LostAndFound-test,LostAndFound-test
```



