__author__ = 'jsh0x'
__version__ = '3.0.0'


import numpy as np
from Constants import FORM_LIST


#Initial Variables
#

class form_object:
	def __init__(self, name, coordinates):
			if type(name) is not str: raise TypeError(f"Expected type: {str}, instead got type: {type(name)}")
			if type(coordinates) is np.ndarray:
				if len(coordinates.shape) == 1:
					if coordinates.shape[0] != 4: raise IndexError(f"Expected an array size of 4, instead got {coordinates.shape[0]}")
					coordinates = np.asarray(coordinates, dtype=np.uint8)
				else: raise IndexError(f"Expected an array shape of 1, instead got {len(coordinates.shape)}")
			elif type(coordinates) is tuple or type(coordinates) is list:
				if len(coordinates) != 4: raise IndexError(f"Expected a list/tuple length of 4, instead got {len(coordinates)}")
				coordinates = np.array(coordinates, dtype=np.uint8)
			else: raise TypeError(f"Expected type: {np.ndarray} or {tuple} or {list}, instead got type: {type(coordinates)}")
			self.name = name
			self.coordinates = coordinates

# TODO: form_object_group class

class checkbox(form_object):
	def __init__(self, name, coordinates):
		super().__init__(name, coordinates)
		self.checked_status = False

	def check(self):
		if self.checked_status:
			self.checked_status = False
		elif not self.checked_status:
			self.checked_status = True

	def set_checked_state(self, state):
		if str(state) != "True" and str(state) != "False": raise TypeError(f"Expected type: {bool}, instead got type: {type(state)}")
		else: self.checked_status = state

class radiobutton(form_object):
	def __init__(self, name, coordinates):
		super().__init__(name, coordinates)
		self.checked_status = False

	def check(self):
		if self.checked_status: self.checked_status = False
		elif not self.checked_status: self.checked_status = True

	def set_checked_state(self, state):
		if str(state) != "True" and str(state) != "False": raise TypeError(f"Expected type: {bool}, instead got type: {type(state)}")
		else: self.checked_status = state

class radiobuttonGroup:
	def __init__(self, name):
		if type(name) is not str: raise TypeError(f"Expected type: {str}, instead got type: {type(name)}")
		self.name = name
		self.radiobuttons = None

	def add_radiobutton(self, name, coordinates):
		if self.radiobuttons is None: self.radiobuttons = np.empty((1,), dtype=object)
		else:
			for val in self.radiobuttons:
				if name.lower() == val.name.lower(): raise NameError(f"Radiobutton with name '{name}' already exists")
			n = self.radiobuttons.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.radiobuttons
			self.radiobuttons = _temp
		self.radiobuttons[self.radiobuttons.shape[0]-1] = radiobutton(name, coordinates)

	def remove_radiobutton(self, name):
		if self.radiobuttons is None: raise IndexError("Radiobuttons is empty")
		n = self.radiobuttons.shape[0]
		if n == 1 and self.radiobuttons[0].name == name:
			self.radiobuttons = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.radiobuttons:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Radiobutton with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.radiobuttons = _temp

	def get_radiobutton(self, name):
		if self.radiobuttons is None: raise IndexError("Radiobuttons is empty")
		for val in self.radiobuttons:
			if val.name == name: return val
		else: raise NameError(f"Radiobutton with name '{name}' does not exist")

	def check(self, name):
		_temp = self.get_radiobutton(name)
		if not _temp.checked_status:
			for val in self.radiobuttons:
				val.set_checked_state(False)
			_temp.checked_status = True

class grid(form_object):
	def __int__(self, name, coordinates, size):
		super().__init__(name, coordinates)
		if type(size) is np.ndarray:
			if len(size.shape) == 1:
				if size.shape[0] != 2: raise IndexError(f"Expected an array size of 2, instead got {size.shape[0]}")
				size = np.asarray(size, dtype=np.uint8)
			else: raise IndexError(f"Expected an array shape of 1, instead got {len(size.shape)}")
		elif type(size) is tuple or type(size) is list:
			if len(size) != 2: raise IndexError(f"Expected a list/tuple length of 2, instead got {len(size)}")
			size = np.array(size, dtype=np.uint8)
		else: raise TypeError(f"Expected type: {np.ndarray} or {tuple} or {list}, instead got type: {type(size)}")
		self.grid = np.empty(size, dtype=object)

	def get(self, xy):
		if type(xy) is np.ndarray:
			if len(xy.shape) == 1:
				if xy.shape[0] != 2: raise IndexError(f"Expected an array size of 2, instead got {xy.shape[0]}")
				xy = np.asarray(xy, dtype=np.uint8)
			else: raise IndexError(f"Expected an array shape of 1, instead got {len(xy.shape)}")
		elif type(xy) is tuple or type(xy) is list:
			if len(xy) != 2: raise IndexError(f"Expected a list/tuple length of 2, instead got {len(xy)}")
			xy = np.array(xy, dtype=np.uint8)
		else: raise TypeError(f"Expected type: {np.ndarray} or {tuple} or {list}, instead got type: {type(xy)}")
		return self.grid[xy[1], xy[0]]

	def put(self, xy, value):
		if type(xy) is np.ndarray:
			if len(xy.shape) == 1:
				if xy.shape[0] != 2: raise IndexError(f"Expected an array size of 2, instead got {xy.shape[0]}")
				xy = np.asarray(xy, dtype=np.uint8)
			else: raise IndexError(f"Expected an array shape of 1, instead got {len(xy.shape)}")
		elif type(xy) is tuple or type(xy) is list:
			if len(xy) != 2: raise IndexError(f"Expected a list/tuple length of 2, instead got {len(xy)}")
			xy = np.array(xy, dtype=np.uint8)
		else: raise TypeError(f"Expected type: {np.ndarray} or {tuple} or {list}, instead got type: {type(xy)}")
		if type(value) is np.ndarray: value = value.tolist()
		self.grid[xy[1], xy[0]] = value
	#Add scrollbar, vertical and horizontal

class tab(form_object):
	def __init__(self, name, coordinates, contents):
		super().__init__(name, coordinates)
		self.content = contents
		self.open_status = False

class tabGroup:
	def __init__(self, name):
		if type(name) is not str: raise TypeError(f"Expected type: {str}, instead got type: {type(name)}")
		self.name = name
		self.tabs = None

	def add_tab(self, name, coordinates, contents):
		if self.tabs is None: self.tabs = np.empty((1,), dtype=object)
		else:
			for val in self.tabs:
				if name.lower() == val.name.lower(): raise NameError(f"Tab with name '{name}' already exists")
			n = self.tabs.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.tabs
		self.tabs[self.tabs.shape[0]-1] = tab(name, coordinates, contents)

	def remove_tab(self, name):
		if self.tabs is None: raise IndexError("Tabs is empty")
		n = self.tabs.shape[0]
		if n == 1 and self.tabs[0].name == name:
			self.tabs = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.tabs:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Tab with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.tabs = _temp

	def get_tab(self, name):
		if self.tabs is None: raise IndexError("Tabs is empty")
		for val in self.tabs:
			if val.name == name: return val
		else: raise NameError(f"Tab with name '{name}' does not exist")

	def open_tab(self, name):
		_temp = self.get_tab(name)
		if not _temp.open_status:
			for val in self.tabs:
				val.open_status = False
			_temp.open_status = True

class combobox(form_object):
	def __init__(self, name, coordinates):
		super().__init__(name, coordinates)
		self.values = None
		self.selected_index = None
		self.selected_value = None

	def change_selected_index(self, index):
		if self.values is None: raise IndexError("Values is empty")
		if self.values.shape[0]-1 > index: raise IndexError(f"Index {index} is out of range for values")
		if type(index) is not int: raise TypeError(f"Expected type: {int}, instead got type: {type(index)}")
		self.selected_index = index
		self.selected_value = self.values[self.selected_index]

	def add_value(self, name):
		if type(name) is not str: raise TypeError(f"Expected type: {str}, instead got type: {type(name)}")
		if self.values is None:
			self.values = np.empty((1,), dtype=str)
			self.change_selected_index(0)
		else:
			n = self.values.shape[0]
			_temp = np.empty((n+1), dtype=str)
			_temp[:-1] = self.values
		self.values[self.values.shape[0]-1] = name

	def remove_value(self, name):
		if type(name) is not str: raise TypeError(f"Expected type: {str}, instead got type: {type(name)}")
		if self.values is None: raise IndexError("Values is empty")
		n = self.values.shape[0]
		if n == 1 and self.values[0].name == name:
			self.values = None
			self.selected_index = None
			self.selected_value = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.values:
				if val.name == name:
					if i == self.selected_index:
						shift = True
					else:
						shift = False
					continue
				elif np.equal(i, n): raise NameError(f"Value with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else:
				self.values = _temp
				if shift: self.change_selected_index(0)
	
	def change_selected_value(self, name):
		if self.values is None: raise IndexError("Values is empty")
		for i,val in enumerate(self.values):
			if val == name:
				self.selected_index = i
				self.selected_value = self.values[self.selected_index]
				break
		else: raise NameError(f"Value '{name}' does not exist")

class textbox(form_object):
	def __init__(self, name, coordinates):
		super().__init__(name, coordinates)
		self.text = None
		self.lines = 0

	def add_line(self, text):
		if type(text) is not str: raise TypeError(f"Expected type: {str}, instead got type: {type(text)}")
		if self.text is None: self.text = np.empty((1,), dtype=str)
		else:
			n = self.text.shape[0]
			_temp = np.empty((n+1), dtype=str)
			_temp[:-1] = self.text
		self.text[self.text.shape[0]-1] = text
		self.lines = self.text.shape[0]

	def remove_line(self, line_number=None):
		if self.text is None: raise IndexError("Text is empty")
		if line_number is None: line_number = self.text.shape[0]
		if type(line_number) is not int: raise TypeError(f"Expected type: {int}, instead got type: {type(line_number)}")
		if line_number < 0 or line_number > self.lines: raise IndexError(f"Line {line_number} is out of range.")
		n = self.text.shape[0]
		if n == 1:
			self.text = None
			self.lines = 0
		else:
			_temp = np.empty((n-1), dtype=str)
			i = 0
			for row in self.text:
				i += 1
				if i == line_number: continue
				_temp[i-1] = row
			else:
				self.text = _temp
				self.lines = self.text.shape[0]


	#def insert_line(self, text, line_number):

class numberbox(form_object):
	def __init__(self, name, coordinates, data_type=np.float32):
		super().__init__(name, coordinates)
		self.value = 0
		self.data_type = data_type
		self.value = self.data_type(self.value)

	def add_value(self, value):
		self.value = self.data_type(np.add(self.value, value))

	def subtract_value(self, value):
		self.value = self.data_type(np.subtract(self.value, value))

	def multiply_value(self, value):
		self.value = self.data_type(np.multiply(self.value, value))

	def divide_value(self, value):
		self.value = self.data_type(np.true_divide(self.value, value))

	def floor_divide_value(self, value):
		self.value = self.data_type(np.floor_divide(self.value, value))

	def set_value(self, value):
		self.value = self.data_type(value)

	def increment_value(self):
		self.value = self.data_type(np.add(self.value, 1))

	def decrement_value(self):
		self.value = self.data_type(np.subtract(self.value, 1))

class button(form_object):
	def __init__(self, name, coordinates, text=None):
		super().__init__(name, coordinates)
		if text is None: self.text = self.name
		elif type(text) is str: self.text = text
		else: raise TypeError(f"Expected type: {str}, instead got type: {type(text)}")

class label(form_object):
	def __int__(self, name, coordinates, text=None):
		super().__init__(name, coordinates)
		if text is None: self.text = self.name
		elif type(text) is str: self.text = text
		else: raise TypeError(f"Expected type: {str}, instead got type: {type(text)}")

class Form:
	def __init__(self, name):
		if name in FORM_LIST: raise NameError(f"Form '{name}' already exists")
		self.name = name
		self.checkboxes = None
		self.radiobuttonGroups = None
		self.grids = None
		self.tabGroups = None
		self.comboboxes = None
		self.textboxes = None
		self.numberboxes = None
		self.buttons = None
		self.labels = None

	def add_checkbox(self, name, coordinates):
		if self.checkboxes is None: self.checkboxes = np.empty((1,), dtype=object)
		else:
			#Check for duplicates
			for val in self.checkboxes:
				if name.lower() == val.name.lower(): raise NameError(f"Checkbox with name '{name}' already exists")
			n = self.checkboxes.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.checkboxes
			self.checkboxes = _temp
		self.checkboxes[self.checkboxes.shape[0]-1] = checkbox(name, coordinates)
	def remove_checkbox(self, name):
		if self.checkboxes is None: raise IndexError("Checkboxes is empty")
		n = self.checkboxes.shape[0]
		if n == 1 and self.checkboxes[0].name == name:
			self.checkboxes = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.checkboxes:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Checkbox with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.checkboxes = _temp
	def get_checkbox(self, name):
		if self.checkboxes is None: raise IndexError("Checkboxes is empty")
		for val in self.checkboxes:
			if val.name == name: return val
		else: raise NameError(f"Checkboxes with name '{name}' does not exist")

	def add_radiobuttonGroup(self, name):
		if self.radiobuttonGroups is None: self.radiobuttonGroups = np.empty((1,), dtype=object)
		else:
			for val in self.radiobuttonGroups:
				if name.lower() == val.name.lower(): raise NameError(f"Radiobutton group with name '{name}' already exists")
			n = self.radiobuttonGroups.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.radiobuttonGroups
			self.radiobuttonGroups = _temp
		self.radiobuttonGroups[self.radiobuttonGroups.shape[0]-1] = radiobuttonGroup(name)
	def remove_radiobuttonGroup(self, name):
		if self.radiobuttonGroups is None: raise IndexError("Radiobutton groups is empty")
		n = self.radiobuttonGroups.shape[0]
		if n == 1 and self.radiobuttonGroups[0].name == name:
			self.radiobuttonGroups = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.radiobuttonGroups:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Radiobutton group with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.radiobuttonGroups = _temp
	def get_radiobuttonGroup(self, name):
		if self.radiobuttonGroups is None: raise IndexError("Radiobutton groups is empty")
		for val in self.radiobuttonGroups:
			if val.name == name: return val
		else: raise NameError(f"Radiobutton group with name '{name}' does not exist")

	def add_grid(self, name, coordinates, size):
		if self.grids is None: self.grids = np.empty((1,), dtype=object)
		else:
			for val in self.grids:
				if name.lower() == val.name.lower(): raise NameError(f"Grid with name '{name}' already exists")
			n = self.grids.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.grids
			self.grids = _temp
		self.grids[self.grids.shape[0]-1] = grid(name, coordinates, size)
	def remove_grid(self, name):
		if self.grids is None: raise IndexError("Grids is empty")
		n = self.grids.shape[0]
		if n == 1 and self.grids[0].name == name:
			self.grids = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.grids:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Grid with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.grids = _temp
	def get_grid(self, name):
		if self.grids is None: raise IndexError("Grids is empty")
		for val in self.grids:
			if val.name == name: return val
		else: raise NameError(f"Grid with name '{name}' does not exist")

	def add_tabGroup(self, name):
		if self.tabGroups is None: self.tabGroups = np.empty((1,), dtype=object)
		else:
			for val in self.tabGroups:
				if name.lower() == val.name.lower(): raise NameError(f"Tab group with name '{name}' already exists")
			n = self.tabGroups.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.tabGroups
			self.tabGroups = _temp
		self.tabGroups[self.tabGroups.shape[0]-1] = tabGroup(name)
	def remove_tabGroup(self, name):
		if self.tabGroups is None: raise IndexError("Tab groups is empty")
		n = self.tabGroups.shape[0]
		if n == 1 and self.tabGroups[0].name == name:
			self.tabGroups = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.tabGroups:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Tab group with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.tabGroups = _temp
	def get_tabGroup(self, name):
		if self.tabGroups is None: raise IndexError("Tab groups is empty")
		for val in self.tabGroups:
			if val.name == name: return val
		else: raise NameError(f"Tab group with name '{name}' does not exist")

	def add_combobox(self, name, coordinates):
		if self.comboboxes is None: self.comboboxes = np.empty((1,), dtype=object)
		else:
			for val in self.comboboxes:
				if name.lower() == val.name.lower(): raise NameError(f"Combobox with name '{name}' already exists")
			n = self.comboboxes.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.comboboxes
			self.comboboxes = _temp
		self.comboboxes[self.comboboxes.shape[0]-1] = combobox(name, coordinates)
	def remove_combobox(self, name):
		if self.comboboxes is None: raise IndexError("Comboboxes is empty")
		n = self.comboboxes.shape[0]
		if n == 1 and self.comboboxes[0].name == name:
			self.comboboxes = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.comboboxes:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Combobox with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.comboboxes = _temp
	def get_combobox(self, name):
		if self.comboboxes is None: raise IndexError("Comboboxes is empty")
		for val in self.comboboxes:
			if val.name == name: return val
		else: raise NameError(f"Combobox with name '{name}' does not exist")

	def add_textbox(self, name, coordinates):
		if self.textboxes is None: self.textboxes = np.empty((1,), dtype=object)
		else:
			for val in self.textboxes:
				if name.lower() == val.name.lower(): raise NameError(f"Textbox with name '{name}' already exists")
			n = self.textboxes.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.textboxes
			self.textboxes = _temp
		self.textboxes[self.textboxes.shape[0]-1] = textbox(name, coordinates)
	def remove_textbox(self, name):
		if self.textboxes is None: raise IndexError("Textboxes is empty")
		n = self.textboxes.shape[0]
		if n == 1 and self.textboxes[0].name == name:
			self.textboxes = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.textboxes:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Textbox with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.textboxes = _temp
	def get_textbox(self, name):
		if self.textboxes is None: raise IndexError(f"Textboxes is empty")
		for val in self.textboxes:
			if val.name == name: return val
		else: raise NameError(f"Textbox with name '{name}' does not exist")

	def add_numberbox(self, name, coordinates):
		if self.numberboxes is None: self.numberboxes = np.empty((1,), dtype=object)
		else:
			for val in self.numberboxes:
				if name.lower() == val.name.lower(): raise NameError(f"Numberbox with name '{name}' already exists")
			n = self.numberboxes.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.numberboxes
			self.numberboxes = _temp
		self.numberboxes[self.numberboxes.shape[0]-1] = numberbox(name, coordinates)
	def remove_numberbox(self, name):
		if self.numberboxes is None: raise IndexError("Numberboxes is empty")
		n = self.numberboxes.shape[0]
		if n == 1 and self.numberboxes[0].name == name:
			self.numberboxes = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.numberboxes:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Numberbox with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.numberboxes = _temp
	def get_numberbox(self, name):
		if self.numberboxes is None: raise IndexError("Numberboxes is empty")
		for val in self.numberboxes:
			if val.name == name: return val
		else: raise NameError(f"Numberbox with name '{name}' does not exist")

	def add_button(self, name, coordinates, text):
		if self.buttons is None: self.buttons = np.empty((1,), dtype=object)
		else:
			for val in self.buttons:
				if name.lower() == val.name.lower(): raise NameError(f"Button with name '{name}' already exists")
			n = self.buttons.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.buttons
			self.buttons = _temp
		self.buttons[self.buttons.shape[0]-1] = button(name, coordinates, text)
	def remove_button(self, name):
		if self.buttons is None: raise IndexError("Buttons is empty")
		n = self.buttons.shape[0]
		if n == 1 and self.buttons[0].name == name:
			self.buttons = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.buttons:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Button with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.buttons = _temp
	def get_button(self, name):
		if self.buttons is None: raise IndexError("Buttons is empty")
		for val in self.buttons:
			if val.name == name: return val
		else: raise NameError(f"Button with name '{name}' does not exist")

	def add_label(self, name, coordinates, text):
		if self.labels is None: self.labels = np.empty((1,), dtype=object)
		else:
			for val in self.labels:
				if name.lower() == val.name.lower(): raise NameError(f"Label with name '{name}' already exists")
			n = self.labels.shape[0]
			_temp = np.empty((n+1), dtype=object)
			_temp[:-1] = self.labels
			self.labels = _temp
		self.labels[self.labels.shape[0]-1] = label(name, coordinates, text)
	def remove_label(self, name):
		if self.labels is None: raise IndexError("Labels is empty")
		n = self.labels.shape[0]
		if n == 1 and self.labels[0].name == name:
			self.labels = None
		else:
			_temp = np.empty((n-1), dtype=object)
			i = 0
			for val in self.labels:
				if val.name == name: continue
				elif np.equal(i, n): raise NameError(f"Label with name '{name}' does not exist")
				_temp[i] = val
				i += 1
			else: self.labels = _temp
	def get_label(self, name):
		if self.labels is None: raise IndexError("Labels is empty")
		for val in self.labels:
			if val.name == name: return val
		else: raise NameError(f"Label with name '{name}' does not exist")


units_form = Form("Units")
