
import logging
log = logging.getLogger('road_anomaly_benchmark.__main__')
import json
from pathlib import Path

import click
import numpy as np
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
@click.option('--default-instancer/--own-instancer', default=True)
def metric(method_names, metric_names, dataset_names, limit_length, parallel, frame_vis, default_instancer):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	for dset in dataset_names:
		for method in method_names:
			for metric in metric_names:

				log.info(f'Metric: {metric} | Method : {method} | Dataset : {dset}')

				ev = Evaluation(
					method_name = method, 
					dataset_name = dset,
					# metrics
					# metrics = ['PixelClassification'],
				)

				ag = ev.calculate_metric_from_saved_outputs(
					metric,
					sample = (dset, range(limit_length)) if limit_length != 0 else None,
					parallel = parallel,
					show_plot = False,
					frame_vis = frame_vis,
					default_instancer = default_instancer,
				)


@main.command()
@click.argument('comparison_name', type=str)
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
@click.option('--order-by', type=str, default=None)
@click.option('--names', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def comparison(comparison_name, method_names, metric_names, dataset_names, order_by=None, names=None, plot_formats=None):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	order_by = order_by or f'{dataset_names[0]}.{metric_names[0]}.area_PRC'

	if names is not None:
		rename_map = json.loads(Path(names).read_text())
		rename_methods = rename_map.get('methods', {})
		rename_dsets = rename_map.get('datasets', {})
		rename_metrics = rename_map.get('metrics', {})

		plot_formats = rename_map.get('plots', {})

		order_by = rename_metrics.get(order_by, order_by)

	else:
		rename_methods = rename_dsets = rename_metrics = plot_formats = {}

	columns = {}

	def get_col(name):
		c = columns.get(name)
		if c is not None:
			return c
		else:
			return columns.setdefault(name, Series(dtype=np.float64))



	for dset in dataset_names:
		for metric_name in metric_names:
			metric = MetricRegistry.get(metric_name)
	
			ags = [
				metric.load(method_name = method, dataset_name = dset)
				for method in method_names
			]

			if "PixBinaryClass" in metric_name:
				metric.plot_many(
					ags,
					f'{comparison_name}_{dset}',
					method_names = rename_methods,
					plot_formats = plot_formats,
				)

			for ag, method in zip(ags, method_names):
				for f, v in metric.extracts_fields_for_table(ag).items():

					ds = rename_dsets.get(dset, dset)
					met = f'{metric_name}.{f}'
					met = rename_metrics.get(met, met)
					
					if met:
						# colname = f'{dset}.{metric_name}.{f}'
						colname = f'{ds}.{met}'
						mn = rename_methods.get(method, method)

						get_col(colname)[mn] = v


	table = DataFrame(data = columns)

	if order_by in table:
		table = table.sort_values(order_by, ascending=False)
	else:
		log.warn(f'Order by: no column {order_by}')

	print(table)

	float_format = lambda f: '-' if np.isnan(f) else f'{100*f:.01f}'

	table_tex = table.to_latex(
		float_format = float_format,
	)

	table_html = table.to_html(
		float_format = float_format,
	)

	# json dump for website
	table['method'] = table.index
	table_json = table.to_json(orient='records')
	table_data = json.loads(table_json)
	table_data = [{k.replace('.', '-'): v for k, v in r.items()} for r in table_data]
	table_json = json.dumps(table_data)

	out_f = DIR_OUTPUTS / 'tables' / comparison_name
	out_f.parent.mkdir(parents=True, exist_ok=True)
	out_f.with_suffix('.html').write_text(table_html)
	out_f.with_suffix('.tex').write_text(table_tex)
	out_f.with_suffix('.json').write_text(table_json)


main()	
