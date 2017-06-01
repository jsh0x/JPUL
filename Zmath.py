__author__ = 'jsh0x'
__version__ = '2.0.0'

from sys import maxsize
from math import gcd
from typing import Union,Iterable,Tuple
from fractions import Fraction
import numpy as np
from matplotlib import pyplot as plt



def symmetrical(x1, x2):
	if (even(x1) and even(x2)) or ((even(x1) and not even(x2)) or (even(x2) and not even(x1))):
		return True
	else:
		return False

def even(x: int) -> bool:
	if np.equal(np.remainder(x, 2), 0):
		return True
	else:
		return False

def diff(x1: int, x2: int) -> int:
	return np.absolute(np.subtract(x1, x2))

def dimensional_slicer(array: np.ndarray, kernel_dim: Union[int, Iterable[int]]) -> np.ndarray:
	#TODO
	d = 1
	if type(kernel_dim) is not int:
		try:
			d = kernel_dim.ndim
		except Exception:
			try:
				d = len(kernel_dim)
			except Exception:
				raise TypeError
	"""if type(kernel_dim) is int:
		if kernel_dim > min(array.shape):
			raise IndexError
		d = 1
	elif type(kernel_dim) is np.ndarray:
		if kernel_dim.ndim > array.ndim:
			raise IndexError
		d = kernel_dim.ndim
	else:
		if len(kernel_dim) > array.ndim:
			raise IndexError
		d = len(kernel_dim)"""
	if d == 1:
		pass
	elif d == 2:
		w,h = kernel_dim
		iterator = np.empty([array.shape[0],array.shape[1],2,w,h], dtype=np.intp)
		for (x,y),v in np.ndenumerate(array[::]):
			x_array = np.array([range(x,x+w)] * h)
			y_array = np.empty_like(x_array)
			for i,j in enumerate(range(y,y+h)):
				y_array[i] = [j] * w

			iterator[x,y] = y_array,x_array
		return iterator

def mode(ndarray: np.ndarray):
	if type(ndarray) is not np.ndarray: raise TypeError
	value, count = np.unique(ndarray, return_counts=True)
	return value[count.tolist().index(count.max())]

def partial_derivative(ndarray: np.ndarray):
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

def total_differential(ndarray: np.ndarray, kernel_size=(3,3)):
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

def get_local_max(array: np.ndarray, legacy=False) -> Tuple[np.ndarray, np.ndarray]:
	if len(array.shape) > 1: raise IndexError
	#retval = [i for i in np.arange(1, array.shape[0]-1) if np.less(array[i-1], array[i]) and np.less(array[i+1], array[i])]
	i = [i for i in np.arange(1, array.shape[0] - 1) if np.less(array[i - 1], array[i]) and np.less(array[i + 1], array[i])]
	if legacy:
		return i
	indices = np.zeros_like(array)
	indices[i] = 1
	values = array[i]
	return (indices, values)
	#iterator = np.stack((np.arange(array[1:-1].size),np.arange(array[1:-1].size),np.arange(array[1:-1].size)), axis=1)
	#retval = np.zeros_like(array, dtype=np.int)
	#iterator = np.empty([array.shape[0]-2,3], dtype=np.intp)
	#for i in np.arange(iterator.shape[0]):
	#	iterator[i] = np.arange(i,i+3)

def get_local_min(array: np.ndarray, legacy=False) -> Tuple[np.ndarray, np.ndarray]:
	if len(array.shape) > 1: raise IndexError
	#retval = [i for i in np.arange(1, array.shape[0]-1) if np.greater(array[i-1], array[i]) and np.greater(array[i+1], array[i])]
	i = [i for i in np.arange(1, array.shape[0]-1) if np.greater(array[i-1], array[i]) and np.greater(array[i+1], array[i])]
	if legacy:
		return i
	indices = np.zeros_like(array)
	indices[i] = 1
	values = array[i]
	return (indices, values)

def reduce_fraction(fraction: Tuple[int, int]) -> Tuple[int, int]:
	numerator,denominator = fraction
	if denominator == 0: raise ZeroDivisionError
	retval = Fraction(numerator,denominator).limit_denominator(100)
	return (retval.numerator, retval.denominator)

def rot90cw(a: np.ndarray, n: int= 1, axes=(0,1)) -> np.ndarray:
	return np.rot90(a, np.subtract(4, np.remainder(n, 4)), axes)

def rot90ccw(a: np.ndarray, n: int= 1, axes=(0,1)) -> np.ndarray:
	return np.rot90(a, n, axes)

def line_pair(a: np.ndarray, target_width: int, target_height: int, iterations: int) -> np.ndarray:
	section_width = np.floor_divide(target_width, np.multiply(iterations, 2))
	section_height = np.floor_divide(target_height, np.multiply(iterations, 2))
	print(a.shape, target_width, target_height, section_width, section_height, iterations)
	if even(iterations):
		if iterations != 2:
			a = line_pair(a[:np.multiply(section_height, np.floor_divide(iterations, 2)), :np.multiply(section_width, np.floor_divide(iterations, 2))], np.multiply(section_width, iterations), np.multiply(section_height, iterations), np.floor_divide(iterations, 2))
		for i in range(np.floor_divide(iterations, 2)):
			#plt.imshow(a, cmap='gray')
			#plt.show()
			if a.shape == (target_height, target_width):
				break
			if even(target_width):
				if even(target_height):
					center = np.zeros((a.shape[0], target_width - a.shape[1]))
					center[:-1,:1] = 1
					a = np.hstack((a[:,:target_width - a.shape[1]], center))
					a = np.vstack((a, rot90cw(a, 2)))
				else:
					center = np.zeros((target_height - a.shape[0], a.shape[1]))
					a = np.vstack((a, center))
					a = np.hstack((a, rot90cw(a, 2)))
			else:
				if even(target_height):
					center = np.zeros((a.shape[0], target_width - a.shape[1]))
					center[:-1, :1] = 1
					a = np.hstack((a, center))
					a = np.vstack((a, rot90cw(a, 2)))
				else:
					old_height = a.shape[0]
					center = np.zeros((target_height - a.shape[0], a.shape[1]))
					a = np.vstack((a, center))
					center = np.zeros((a.shape[0], 1))
					center[old_height] = 1
					a = np.hstack((a, center, rot90cw(a, 2)))
		return a
	elif not even(iterations):
		a = line_pair(a[section_height:-section_height, section_width:-section_width], np.multiply(np.multiply(2, section_width), np.subtract(iterations, 1)), np.multiply(np.multiply(2, section_height), np.subtract(iterations, 1)), np.subtract(iterations, 1))
		sub_section_height = np.floor_divide(np.subtract(target_height, a.shape[0]), 2)
		sub_section_width = np.floor_divide(np.subtract(target_width, a.shape[1]), 2)
		padding = np.zeros((sub_section_height, a.shape[1]))
		a = np.vstack((padding, a, padding))

		padding = np.zeros((np.subtract(target_height, sub_section_height), sub_section_width))
		sub_a = np.identity(sub_section_width)
		sub_a_left = np.vstack((sub_a, padding))
		sub_a_right = np.vstack((padding, sub_a))
		a = np.hstack((sub_a_left, a, sub_a_right))
		return a

def line_algorithm(width: int, height: int) -> np.ndarray:
	retval = np.zeros((height, width, 2))
	max_dim = max(width, height)
	min_dim = min(width, height)
	base = np.identity(np.floor_divide(max_dim, 2))
	print(base.shape)
	if width != height:
		dim_d = diff(max_dim, min_dim)
		if even(min_dim):
			d = np.subtract(dim_d, 2)
		else:
			d = np.subtract(dim_d, 1)
		for split in np.arange(1, 1000000, 2):
			print(split)
			if np.greater_equal(np.multiply(split, 2), dim_d):
				break
		else:
			pass  # AKA panic.
		sections = np.add(np.floor_divide(np.subtract(split, 1), 2), 1)
		section_width = np.floor_divide(base.shape[1], sections)
		sub_base = np.identity(section_width)
		sub_base = line_pair(sub_base, 7, 7, sections)

		plt.imshow(sub_base, cmap='gray')
		plt.show()
		if even(max_dim):
			pass


		else:
			pass


		if height > width:
			base = np.fliplr(rot90cw(base))


	retval[:, :, 0] = base
	retval[:, :, 1] = np.fliplr(base)
	return retval
#a = np.identity(30)
#b = line_pair(a, 60, 60, 5)
#plt.imshow(b, cmap='gray')
#plt.show()
#12 iterations
a = line_algorithm(11,7)

print(a)
for x in range(1,20,2):
	print(x)

