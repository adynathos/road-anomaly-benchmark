
import logging
log = logging.getLogger('road_anomaly_benchmark.__main__')

import click
from .evaluation import Evaluation
from .metrics import MetricRegistry

def name_list(name_list):
	return [name for name in name_list.split(',') if name]

@click.group()
def main():
	...

@main.command()
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
@click.option('--limit-length', type=int, default=0)
@click.option('--parallel/--no-parallel', default=True)
@click.option('--frame-vis/--no-frame-vis', default=False)
def metric(method_names, metric_names, dataset_names, limit_length, parallel, frame_vis):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = dataset_names(metric_names)

	for dset in dataset_names:
		for method in method_names:
			for metric in metric_names:

				log.info(f'Metric: {metric} | Method : {method} | Dataset : {dset}')

				ev = Evaluation(
					method_name = method_names, 
					dataset_name = dataset_names,
					# metrics
					# metrics = ['PixelClassification'],
				)

				ag = ev.calculate_metric_from_saved_outputs(
					metric_name,
					sample = (dataset_names, range(limit_length)) if limit_length != 0 else None,
					parallel = parallel,
					show_plot = False,
					frame_vis = frame_vis,
				)


@main.command()
@click.argument('comparison_name', type=str)
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
def comparison(comparison_name, method_names, metric_names, dataset_names):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = dataset_names(metric_names)



main()	
