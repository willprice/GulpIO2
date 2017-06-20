import os
import cv2
import struct
import pickle
import numbers 
import random
import ctypes
import numpy as np 
from PIL import Image
from collections import namedtuple, defaultdict


MetaInfo = namedtuple('MetaInfo', ['loc', 'label', 'idx', 'pad', 'length'])

class GulpVideoIO(object):


	def __init__(self, path, flag, meta_path):
		self.meta_path = meta_path
		self.path = path
		self.flag = flag
		self.is_open = False
		self.is_writable = False
		self.f = None
		self.meta_dict = None


	def open(self):
		if os.path.exists(self.meta_path):
			self.meta_dict = pickle.load(open(self.meta_path, 'rb'))
		else:
			self.meta_dict = defaultdict()

		if self.flag == 'wb':
			self.f = open(self.path, self.flag)
			self.is_writable = True
		elif self.flag == 'rb':
			self.f = open(self.path, self.flag)
			self.is_writable = False
		self.is_open = True


	def close(self):
		if self.is_open:
			pickle.dump(self.meta_dict, open(self.meta_path, 'wb'))
			self.f.close()
			self.is_open = False
		else:
			return


	def write(self, label, vid_idx, image):
		assert self.is_writable
		loc = self.f.tell()
		img_str = cv2.imencode('.jpg', image)[1].tostring()
		pad = 4 - (len(img_str) % 4)
		record = img_str.ljust(len(img_str)+pad, b'\0')
		meta_info = MetaInfo(loc=loc, length=len(record), label=label, idx=vid_idx, pad=pad)
		try:
			self.meta_dict[vid_idx].append(meta_info)
		except KeyError:
			self.meta_dict[vid_idx] = [meta_info]
		self.f.write(record)


	def read(self, meta_info):
		assert not self.is_writable
		self.f.seek(meta_info.loc)
		record = self.f.read(meta_info.length)
		img_str = record[:-meta_info.pad]
		nparr = np.fromstring(img_str, np.uint8)
		img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)
		return Image.fromarray(img)


	def reset(self):
		self.close()
		self.open()


	def seek(self, loc):
		self.f.seek(loc)
