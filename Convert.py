__author__ = 'jsh0x'
__version__ = '1.0.0'

import numpy as np



def hex2rgb(color):
	if type(color) is str and len(color) == 6: return np.array((np.int(color[:2], base=16), np.int(color[2:4], base=16), np.int(color[4:], base=16)), dtype=np.int)
	elif type(color) is not str: raise TypeError(f"Value 'color' incompatible type: {type(color)}")
	else: raise ValueError(f"Value 'color' incompatible hex-string length: {len(color)}")

def hex2rgba(color):
	if type(color) is str and len(color) == 8: return np.array((np.int(color[:2], base=16), np.int(color[2:4], base=16), np.int(color[4:6], base=16), np.int(color[6:], base=16)), dtype=np.int)
	elif type(color) is not str: raise TypeError(f"Value 'color' incompatible type: {type(color)}")
	else: raise ValueError(f"Value 'color' incompatible hex-string length: {len(color)}")

def rgb2hex(color):
	if (type(color) is tuple or type(color) is list or type(color) is np.ndarray) and len(color) == 3:
		if type(color[0]) is not int and type(color[0]) is not float: TypeError(f"Red value in 'color' incompatible type: {type(color[0])}")
		elif color[0] > 255 or color[0] < 0: raise ValueError(f"Red value in 'color' out of range 0 - 255: {color[0]}")
		if type(color[1]) is not int and type(color[1]) is not float: TypeError(f"Green value in 'color' incompatible type: {type(color[1])}")
		elif color[1] > 255 or color[1] < 0: raise ValueError(f"Green value in 'color' out of range 0 - 255: {color[1]}")
		if type(color[2]) is not int and type(color[2]) is not float: TypeError(f"Blue value in 'color' incompatible type: {type(color[2])}")
		elif color[2] > 255 or color[2] < 0: raise ValueError(f"Blue value in 'color' out of range 0 - 255: {color[2]}")
		return hex(color[0]).format('x')[1]+hex(color[1]).format('x')[1]+hex(color[2]).format('x')[1]
	elif type(color) is not tuple and type(color) is not list and type(color) is not np.ndarray: raise TypeError(f"Value 'color' incompatible type: {type(color)}")
	else: raise ValueError(f"Value 'color' incompatible {type(color)} length: {len(color)}")

def rgba2hex(color):
	if (type(color) is tuple or type(color) is list or type(color) is np.ndarray) and len(color) == 4:
		if type(color[0]) is not int and type(color[0]) is not float: TypeError(f"Red value in 'color' incompatible type: {type(color[0])}")
		elif color[0] > 255 or color[0] < 0: raise ValueError(f"Red value in 'color' out of range 0 - 255: {color[0]}")
		if type(color[1]) is not int and type(color[1]) is not float: TypeError(f"Green value in 'color' incompatible type: {type(color[1])}")
		elif color[1] > 255 or color[1] < 0: raise ValueError(f"Green value in 'color' out of range 0 - 255: {color[1]}")
		if type(color[2]) is not int and type(color[2]) is not float: TypeError(f"Blue value in 'color' incompatible type: {type(color[2])}")
		elif color[2] > 255 or color[2] < 0: raise ValueError(f"Blue value in 'color' out of range 0 - 255: {color[2]}")
		if type(color[3]) is not int and type(color[3]) is not float: TypeError(f"Alpha value in 'color' incompatible type: {type(color[3])}")
		elif color[3] > 255 or color[3] < 0: raise ValueError(f"Alpha value in 'color' out of range 0 - 255: {color[3]}")
		return hex(color[0]).format('x')[1]+hex(color[1]).format('x')[1]+hex(color[2]).format('x')[1]+hex(color[3]).format('x')[1]
	elif type(color) is not tuple and type(color) is not list and type(color) is not np.ndarray: raise TypeError(f"Value 'color' incompatible type: {type(color)}")
	else: raise ValueError(f"Value 'color' incompatible {type(color)} length: {len(color)}")
