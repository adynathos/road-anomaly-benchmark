
from typing import Callable

from functools import partial

class DatasetRegistry:
	INITIALIZERS = {}

	MODULES = {}

	@classmethod
	def register(cls, name : str, init_func : Callable, *opts_s, **opts_kw):
		if opts_s or opts_kw:
			init_func = partial(init_func, *opts_s, **opts_kw)

		if name in cls.INITIALIZERS or name in cls.MODULES:
			# log.warn
			print(f'Dataset {name} is already registered')
		else:
			cls.INITIALIZERS[name] = init_func

	@classmethod
	def register_concrete(cls, name : str, dset_object):

		if name in cls.MODULES:
			# log.warn
			print(f'Dataset {name} is already registered')
		else:
			cls.MODULES[name] = dset_object

		return dset_object

	@classmethod
	def list_available_dsets(cls):
		names = set(cls.INITIALIZERS.keys()).union(cls.MODULES.keys())
		names = list(names)
		names.sort()
		return names

	@classmethod
	def get(cls, name : str):
		obj = cls.MODULES.get(name)
		
		if obj is None:
			init_func = cls.INITIALIZERS.get(name)

			if init_func is None:
				dslist = '\n '.join(cls.list_available_dsets())
				raise KeyError(f'No dataset called {name} in registry, avaiable datasets:\n {dslist}')

			else:
				obj = init_func()
				cls.register_concrete(name, obj)

		return obj

	@classmethod
	def register_class(cls, *args, **kwargs):
		def decorator(class_to_register):
			configs = getattr(class_to_register, 'configs')

			# config generator function
			if isinstance(configs, Callable):
				configs = configs()

			for cfg in configs:
				cls.register(cfg['name'], partial(class_to_register, cfg))	

		return decorator
