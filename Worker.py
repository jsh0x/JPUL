__version__ = '2.0.0'
__author__ = 'jsh0x'

import os, sys, logging, datetime
import cv2
import pyperclip
import numpy as np
#from JPUL import (SyteLine as sl, SQL as sql)
#from JPUL.Image import OCR

class Unit:
	def __init__(self, serial_number):
		self.serial_number = serial_number
