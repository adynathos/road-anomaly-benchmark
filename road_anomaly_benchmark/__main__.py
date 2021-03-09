
import logging
log = logging.getLogger('road_anomaly_benchmark.__main__')

import click
from pandas import DataFrame, Series

from .paths import DIR_OUTPUTS
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
@click.option('order-by', type=str, default='LostAndFound-test.PixBinaryClass.area_PRC')
def comparison(comparison_name, method_names, metric_names, dataset_names, order_by):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = dataset_names(metric_names)

	metrics = [MetricRegistry.get(m) for m in metric_names]

	columns = {}

	def get_col(name):
		c = columns.get(name)
		if c:
			return c
		else:
			return columns.setdefault(name, Series(dtype=np.float64))

	for metric in metrics:
		for dset in dataset_names:
			ags = [
				metric.load(method_name = method, dataset_name = dset)
				for method in method_names
			]
			# TODO provide method name map
			metric.plot_many(ags, comparison_name)

			for ag, method in zip(ags, method_names):
				for f, v in metric.extracts_fields_for_table(ag).items():
					get_col(f'{dset}.{metric}.{f}')[method] = v

	table = DataFrame(data = columns)

	if order_by in table:
		table = table.sort_values(order_by, ascending=False)

	table_t = table.transpose()
	float_format = lambda f: '-' if np.isnan(f) else f'{100*f:.01f}'

	table_tex = table_t.to_latex(
		float_format = float_format,
	)

	table_html = table_t.to_html(
		float_format = float_format,
	)

	out_f = DIR_OUTPUTS / 'tables' / comparison_name

	out_f.with_suffix('.html').write_text(table_html)
	out_f.with_suffix('.tex').write_text(table_texl)




main()	
