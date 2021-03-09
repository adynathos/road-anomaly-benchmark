
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
@click.option('--limit-length', type=int, default=0)
@click.option('--parallel/--no-parallel', default=True)
@click.option('--frame-vis/--no-frame-vis', default=False)
def metric(method_names, metric_name, dataset_names, limit_length, parallel, frame_vis):

	ev = Evaluation(
		method_name = method_names, 
		dataset_name = dataset_names,
		# metrics
		# metrics = ['PixelClassification'],
	)

	ag = ev.calculate_metric_from_saved_outputs(
		metric_name,
		sample = (dataset_names, limit_length) if limit_length != 0 else None,
		parallel = parallel,
		show_plot = False,
		frame_vis = frame_vis,
	)


main()	
