__author__ = 'jsh0x'
__version__ = '2.0.0'

import os, sys, logging, datetime
import cv2
import pyperclip
import numpy as np
#from JPUL import (SyteLine as sl, SQL as sql)
#from JPUL.Image import OCR

class Unit:
	def __init__(self, serial_number:str, build:int, carrier:str, product:str, suffix:str):
		if type(serial_number) is not str: raise TypeError
		if len(serial_number) != 7: raise ValueError
		if type(build) is not int: raise TypeError
		if build < 100: raise ValueError
		if carrier:
			if carrier != "Verizon" and carrier != "Sprint": raise ValueError
		if type(product) is not str: raise TypeError
		if type(suffix) is not str: raise TypeError
		self.serial_number = serial_number
		self.build = build
		self.carrier = carrier
		self.product = product
		self.suffix = suffix
		self.parts = None

		self.check()
		build_prefix, serial_number_prefix = self.get_prefix()
		build_suffix = self.get_suffix()
		if carrier: build_carrier = carrier.upper()[0]
		else: build_carrier = ""
		self.full_build = f"{build_prefix}{build}{build_carrier}{build_suffix}"
		self.full_serial_number = f"{serial_number_prefix}{serial_number}"

	def check(self):
		#Check if serial_number matches product
		self.serial_number, self.product
		#Check if product needs a carrier
		self.product, self.carrier
		#Check if build matches product and serial_number
		self.build, self.product, self.serial_number
	def get_suffix(self):
		if self.suffix == "Monitoring": return "-M"
		elif self.suffix == "Demo": return "-DEMO"
		elif self.suffix == "Refurbished": return "-R"
		else: return ""
	def get_prefix(self):
		#Returns build_prefix and serial_number_prefix for product and build
		self.product, self.build
	def convert_build(self, build:int):
		pass
		self.build = build
	def add_parts(self, parts:np.ndarray):
		pass
		self.parts += parts
	def remove_parts(self, parts:np.ndarray):
		pass
		self.parts -= parts
	def clear_parts(self):
		self.parts = None
	def update_build(self):
		self.check()
		if self.carrier: build_carrier = self.carrier.upper()[0]
		else: build_carrier = ""
		build_prefix, serial_number_prefix = self.get_prefix()
		build_suffix = self.get_suffix()
		self.full_build = f"{build_prefix}{self.build}{build_carrier}{build_suffix}"

class MonitoringUnit(Unit):
	def __init__(self, serial_number:str, build:int, product:str, carrier:str=None):
		super().__init__(serial_number=serial_number, build=build, carrier=carrier, product=product, suffix="Monitoring")
	def make_RTS(self):
		pass
		self.suffix = "RTS"
	def make_Refurbished(self):
		pass
		self.suffix = "Refurbished"

class DirectUnit(Unit):
	def __init__(self, serial_number:str, build:int, product:str, customer_number:int, carrier:str=None):
		super().__init__(serial_number=serial_number, build=build, carrier=carrier, product=product, suffix="Direct")

class RTSUnit(Unit):
	def __init__(self, serial_number:str, build:int, product:str, customer_number:int, carrier:str=None):
		super().__init__(serial_number=serial_number, build=build, carrier=carrier, product=product, suffix="RTS")

class DemoUnit(Unit):
	def __init__(self, serial_number:str, build:int, product:str, carrier:str=None):
		super().__init__(serial_number=serial_number, build=build, carrier=carrier, product=product, suffix="Demo")
