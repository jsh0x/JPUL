__author__ = 'jsh0x'
__version__ = '1.0.0'
import numpy as np

def mode(ndarray):
	if type(ndarray) is not np.ndarray: raise TypeError
	value, count = np.unique(ndarray, return_counts=True)
	return value[count.tolist().index(count.max())]