__author__ = 'jsh0x'
__version__ = '2.0.0'
import numpy as np

def mode(ndarray):
	if type(ndarray) is not np.ndarray: raise TypeError
	value, count = np.unique(ndarray, return_counts=True)
	return value[count.tolist().index(count.max())]

def partial_derivative(ndarray):
	if type(ndarray) is not np.ndarray: raise TypeError
	if 2 > len(ndarray.shape) > 3: raise IndexError
	if len(ndarray.shape) == 2:
		cs = np.asarray(ndarray, dtype=np.float32)
		mod = np.floor_divide(cs.shape[0], 2)
		mod2 = np.floor_divide(np.multiply(cs.shape[0], cs.shape[1])-1, 2)
		c = cs[mod, mod]
		temp = np.hstack((cs.flatten()[:mod2], cs.flatten()[mod2+1:]))
		retval = np.rint(np.average(np.abs(np.subtract(c, temp))))
	elif len(ndarray.shape) == 3:
		retval = np.empty((3,), dtype=np.uint8)
		mod = np.floor_divide(ndarray.shape[0], 2)
		mod2 = np.floor_divide(np.multiply(ndarray.shape[0], ndarray.shape[1])-1, 2)
		for i,cs in enumerate(ndarray.T):
			cs = np.asarray(cs.T, dtype=np.float32)
			c = cs[mod, mod]
			temp = np.hstack((cs.flatten()[:mod2], cs.flatten()[mod2+1:]))
			retval[i] = np.rint(np.average(np.abs(np.subtract(c, temp))))
	return retval

def total_differential(ndarray, kernel_size=(3,3)):
	if type(ndarray) is not np.ndarray: raise TypeError
	if 1 >= len(ndarray.shape) > 3: raise IndexError
	mod = int((kernel_size[0]-1)/2)
	if len(ndarray.shape) == 2:
		retval = np.empty((ndarray.shape[0]-(mod*2), ndarray.shape[1]-(mod*2)), dtype=np.uint8)
	elif len(ndarray.shape) == 3:
		retval = np.empty((ndarray.shape[0]-(mod*2), ndarray.shape[1]-(mod*2), 3), dtype=np.uint8)
	y_range = np.arange(mod,ndarray.shape[0]-mod)
	x_range = np.arange(mod,ndarray.shape[1]-mod)
	for y in y_range:
		for x in x_range:
			k = ndarray[y-mod:y+mod+1, x-mod:x+mod+1]
			retval[y-mod, x-mod] = partial_derivative(k)
	return retval