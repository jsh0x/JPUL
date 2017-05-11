__author__ = 'jsh0x'
__version__ = '1.0.0'

from sys import maxsize
from secrets import randbelow
from string import digits, ascii_letters, punctuation
import numpy as np



def generate_char_map(key_length=6):
	retval = {}
	chars = digits+ascii_letters+punctuation
	val = set()
	while len(val) < 94:
		val.add(int(str(randbelow(maxsize))[:key_length]))
	for k,v in zip(val, chars):
		retval[k] = v
	return retval

def generate_key():
	retval = 0
	while len(str(retval)) < 8:
		retval = randbelow(maxsize)
	return retval

def encrypt(char_map:dict, key:int, value:str):
	#TODO: Similarity return for NaCl
	if type(char_map) is not dict: raise TypeError
	if type(key) is not int: raise TypeError
	if type(value) is not str: raise TypeError
	if len(char_map) != 94: raise IndexError
	for k,v in char_map.items():
		if type(k) is not int: raise TypeError
		if type(v) is not str: raise TypeError
		if k >= maxsize: raise KeyError
		if len(v) > 2: raise ValueError
	retval = ""
	for char in value:
		for k, v in char_map.items():
			if v == char:
				k = np.int64(k)
				key = np.int64(key)
				retval += np.binary_repr(np.add(k, key))
	return int(retval, 2)

def decrypt(char_map:dict, key:int, value:str):
	if type(char_map) is not dict: raise TypeError
	if type(key) is not int: raise TypeError
	if type(value) is not str: raise TypeError
	if len(char_map) != 94: raise IndexError
	for k,v in char_map.items():
		if type(k) is not int: raise TypeError
		if type(v) is not str: raise TypeError
		if k >= maxsize: raise KeyError
		if len(v) > 2: raise ValueError
	step = 19
	value = np.binary_repr(np.int(value))
	retval = ""
	for i in range(0, len(value)-step, step):
		retval += char_map[np.subtract(int(value[i:i+step], base=2), key)]
	return retval