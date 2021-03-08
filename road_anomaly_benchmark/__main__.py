
import logging
log = logging.getLogger('road_anomaly_benchmark.__main__')

import click
from .evaluation import Evaluation
from .metrics import MetricRegistry

@click.group()
def main():
	...

@main.command()
@click.argument('metric_name', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
def metric(method_names, metric_name, dataset_names):

	ev = Evaluation(
		method_name = method_names, 
		dataset_name = dataset_names,
		# metrics
		# metrics = ['PixelClassification'],
	)

	metric = MetricRegistry.get(metric_name)

	met_result = ev.calculate_metric_from_saved_outputs(
		metric, 
		sample=(dataset_names, range(20)),
	)
	metric.plot_single(met_result, close=False)


main()	
