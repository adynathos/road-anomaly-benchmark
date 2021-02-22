 
from pathlib import Path
from operator import itemgetter
from easydict import EasyDict
from ..jupyter_show_image import imread, imwrite

class ChannelLoader:
	def read(self, **frame):
		raise NotImplementedError()

	def write(self, value, **frame):
		raise NotImplementedError()

class ChannelLoaderFileCollection:
	"""
	The channel's value for each frame is in a separate file, for example an image.
	@param file_path_tmpl: a template string or a function(channel, **frame)
	"""
	def __init__(self, file_path_tmpl):
		# convert Path to str so that we can .format it
		self.file_path_tmpl = str(file_path_tmpl) if isinstance(file_path_tmpl, Path) else file_path_tmpl

	def resolve_template(self, **frame):
		template = self.file_path_tmpl
		if isinstance(template, str):
			# string template
			return template.format(
				**frame,
			)
		else:
			# function template
			return template(channel=self, **frame)

	def read(self, **frame):
		path = Path(self.resolve_template(**frame))
		return self.read_file(path)

	def write(self, value, **frame):
		path = Path(self.resolve_template(**frame))
		path.parent.mkdir(exist_ok=True, parents=True)
		self.write_file(path, value)

	def read_file(self, path):
		raise NotImplementedError('read_file for {c}'.format(c=self.__class__.__name__))

	def write_file(self, path, data):
		raise NotImplementedError('write_file for {c}'.format(c=self.__class__.__name__))

	def __repr__(self):
		return '{cls}({tp})'.format(cls=self.__class__.__name__, tp=self.file_path_tmpl)


# class ImageBackgroundService:
# 	IMWRITE_BACKGROUND_THREAD = ThreadPoolExecutor(max_workers=3)

# 	@classmethod
# 	def imwrite(cls, path, data):
# 		cls.IMWRITE_BACKGROUND_THREAD.submit(imwrite, path, data)


class ChannelLoaderImage(ChannelLoaderFileCollection):
	def __init__(self, file_path_tmpl=None, save_opts={}):
		super().__init__(file_path_tmpl)
		self.save_opts = save_opts

	def read_file(self, path):
		return imread(path, **self.save_opts)

	def write_file(self, path, data):
		imwrite(path, data)


class DatasetBase:

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)
		self.frames = []
		self.frames_by_fid = {}

	def set_frames(self, frame_list):
		self.frames = list(frame_list)
		self.frames.sort(key = itemgetter('fid'))
		self.frames_by_fid = {fr['fid']: fr for fr in self.frames}

	def __getitem__(self, idx_or_fid):
		if isinstance(idx_or_fid, int):
			fr = self.frames[idx_or_fid]
		else:
			fr = self.frames_by_fid[idx_or_fid]
		
		out_fr = EasyDict(fr)

		for ch_name, ch_obj in self.channels.items():
			out_fr[ch_name] = ch_obj.read(dset=self, **fr)

		return out_fr

	def __len__(self):
		return self.frames.__len__()

	def __iter__(self):
		for idx in range(self.__len__()):
			yield self[idx]



