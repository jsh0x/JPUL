__author__ = 'jsh0x'
__version__ = '1.0.0'

import os
import cv2
import numpy as np
from PIL import Image



def OCR(input_string, haystack_image):
	if type(input_string) is not str: raise TypeError(f"Value 'input_string' must be type str, instead got type {type(input_string)}")
	if len(input_string) == 0: raise ValueError("Value 'input_string' is empty")
	if type(haystack_image) is str and not os.path.exists(haystack_image): raise FileNotFoundError(f"Image directory '{haystack_image}' not found")
	elif type(haystack_image) is str: img = np.array(Image.open(haystack_image))
	elif type(haystack_image) is Image.Image: img = np.array(haystack_image)
	elif type(haystack_image) is np.ndarray: img = haystack_image
	else: raise TypeError(f"Value 'haystack_image' must be a PIL Image-Object, an image directory, or an image array, instead got {type(haystack_image)}")
	return
