__author__ = 'jsh0x'
__version__ = '1.0.0'


import numpy as np
from Convert import hex2rgb


#Initial Variables
#



def get_array(text):
	if type(text) is not str: raise TypeError("Value 'text' must be type str, instead got type {}".format(type(text)))
	if len(text) == 0: raise ValueError("Value 'text' is empty!")
