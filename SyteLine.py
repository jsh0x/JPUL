__author__ = 'jsh0x'
__version__ = '1.0.0'


import numpy as np
from Convert import hex2rgb
from Constants import FORM_LIST


#Initial Variables
#

class Form:
	def __init__(self, name):
		if name in FORM_LIST: raise NameError(f"Form '{name}' already exists")
		self.name = name
		self.checkboxes = None
		self.radiobuttons = None
		self.grids = None
		self.tabs = None
		self.comboboxes = None
		self.textboxes = None
		self.numberboxes = None
		self.buttons = None
		self.labels = None

	def add_checkbox(self, pos):
		if type(pos) is np.ndarray:
			if len(pos.shape) == 1:
				if pos.shape[0] != 4: raise IndexError(f"Expected an array size of 4, instead got {pos.shape[0]}")
				pass
			elif len(pos.shape) == 2:
				pass
			elif len(pos.shape) == 3:
				pass
			else: raise IndexError(f"Expected an array shape of 1 or 2 or 3, instead got {len(pos.shape)}")
		elif type(pos) is tuple or type(pos) is list:
			pass
		else: raise TypeError(f"Expected type: {np.ndarray} or {tuple} or {list}, instead got type: {type(pos)}")
		if self.checkboxes is None:
			self.checkboxes = np.empty((1,2), dtype=np.uint8)
