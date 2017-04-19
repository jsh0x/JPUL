__author__ = 'jsh0x'
__version__ = '1.0.0'

import os
from secrets import randbelow as rand
import cv2
import numpy as np
from PIL import Image, ImageDraw
from Convert import hex2rgb, index2coord, rgb2hex
from Constants import CHARACTER_ARRAYS
from Zmath import total_differential as t_diff
import datetime



def space_conflict(roi1, roi2):
	roi1 = [(x,y) for y in np.arange(roi1[1],roi1[3]) for x in np.arange(roi1[0],roi1[2])]
	roi2 = [(x,y) for y in np.arange(roi2[1],roi2[3]) for x in np.arange(roi2[0],roi2[2]) if (x,y) in roi1]
	if len(roi2) > 0: return True
	else: return False

def get_gradient(image):
	"""finds the anti-derivative of an image's RGB values"""
	if type(image) is str and not os.path.exists(image): raise FileNotFoundError(f"Image directory '{image}' not found")
	elif type(image) is str: img = np.array(Image.open(image).convert('RGB'), dtype=np.int16)
	elif type(image) is Image.Image: img = np.array(image, dtype=np.int16)
	elif type(image) is np.ndarray: img = image
	else: raise TypeError(f"Value 'image' must be a PIL Image-Object, an image directory, or an image array, instead got {type(image)}")
	ret_img = t_diff(img)
	"""ret_img = np.zeros((img.shape[0]-1, img.shape[1]-1, 3))
	ret_dict = {}
	for y in np.arange(ret_img.shape[0]):
		for x in np.arange(ret_img.shape[1]):
			y2 = ret_img.shape[0]-y-1
			x2 = ret_img.shape[1]-x-1
			if str(y) + "," + str(x) not in ret_dict.keys():
				ret_dict[str(y) + "," + str(x)] = []
			if str(y) + "," + str(x2) not in ret_dict.keys():
				ret_dict[str(y) + "," + str(x2)] = []
			if str(y2) + "," + str(x) not in ret_dict.keys():
				ret_dict[str(y2) + "," + str(x)] = []
			if str(y2) + "," + str(x2) not in ret_dict.keys():
				ret_dict[str(y2) + "," + str(x2)] = []
			ret_dict[str(y) + "," + str(x)].append(np.abs(np.subtract(img[y, x, :],img[y+1, x+1, :])))
			ret_dict[str(y2) + "," + str(x2)].append(np.abs(np.subtract(img[y2, x2, :], img[y2, x2, :])))
			ret_dict[str(y) + "," + str(x2)].append(np.abs(np.subtract(img[y, x2, :], img[y + 1, x2, :])))
			ret_dict[str(y2) + "," + str(x)].append(np.abs(np.subtract(img[y2, x, :], img[y2, x + 1, :])))
	for k in ret_dict.keys():
		v = ret_dict[k]
		y,x = k.split(',')
		y = int(y)
		x = int(x)
		r = 0
		g = 0
		b = 0
		for r2,g2,b2 in v:
			r += r2
			g += g2
			b += b2
		r /= len(v)
		g /= len(v)
		b /= len(v)
		ret_img[y,x] = np.array([r,g,b], dtype=np.uint8)
	return ret_img.mean(axis=2)"""
	return ret_img

def OCR(input_string, haystack_image, tolerance_threshold=10, roi=None):
	#HIGHLY recommend tolerance thresholds < 20
	if type(input_string) is not str: raise TypeError(f"Value 'input_string' must be type str, instead got type {type(input_string)}")
	if len(input_string) == 0: raise ValueError("Value 'input_string' is empty")
	if type(haystack_image) is str and not os.path.exists(haystack_image): raise FileNotFoundError(f"Image directory '{haystack_image}' not found")
	elif type(haystack_image) is str: img = np.array(Image.open(haystack_image).convert('RGB'), dtype=np.int16)
	elif type(haystack_image) is Image.Image: img = np.array(haystack_image.convert('RGB'), dtype=np.int16)
	elif type(haystack_image) is np.ndarray: img = haystack_image
	else: raise TypeError(f"Value 'haystack_image' must be a PIL Image-Object, an image directory, or an image array, instead got {type(haystack_image)}")
	retval_dict = {}
	im = Image.fromarray(np.asarray(img, dtype=np.uint8))
	draw = ImageDraw.Draw(im)
	if roi is None:
		char = input_string[0]
		img2 = np.copy(img)
		char_array = CHARACTER_ARRAYS[char]
		char_array2 = np.zeros((char_array.shape[0], char_array.shape[1], 3))
		for y in np.arange(char_array.shape[0]):
			for x in np.arange(char_array.shape[1]):
				char_array2[y,x]=hex2rgb(char_array[y,x])
		char_array3 = char_array2.mean(axis=2)
		for y in np.arange(img2.shape[0]-char_array3.shape[0]):
			for x in np.arange(img2.shape[1]-char_array3.shape[1]):
				try: krn = (img2[y:y+char_array3.shape[0],x:x+char_array3.shape[1]]).mean(axis=2)
				except IndexError: krn = (img2[y:y+char_array3.shape[0],x:x+char_array3.shape[1]])
				if np.abs(np.subtract(krn, char_array3)).mean() < tolerance_threshold:
					if roi is None: roi = [(x, y, x + char_array.shape[1], y + char_array.shape[0])]
					else: roi.append((x, y, x + char_array.shape[1], y + char_array.shape[0]))
		else:
			old = tolerance_threshold
			for char in input_string[1:]:
				if char == ('l' or 'I'):
					tolerance_threshold = 10
				else: tolerance_threshold = old
				for i in range(len(roi)):
					x1, y1, x2, y2 = roi.pop(0)
					#color = (rand(256), rand(256), rand(256))
					#color2 = (rand(256), rand(256), rand(256))
					#size=2
					char_array = CHARACTER_ARRAYS[char]
					y3 = np.add(np.floor_divide((y2-y1),2), y1)
					y4 = y3 + char_array.shape[0]
					y3 = y3 - char_array.shape[0]
					x4 = np.add(np.add(x2,char_array.shape[1]), np.floor_divide(char_array.shape[1], 2))
					img2 = np.copy(img[min(y1,y3):max(y2,y4), x1:x4])
					print(min(y1,y3),max(y2,y4), x1,x4)
					#draw.ellipse((x1 - size, y1 - size, x1 + size, y1 + size), fill=color)
					#draw.ellipse((x2 - size, y2 - size, x2 + size, y2 + size), fill=color)
					#draw.ellipse((x1 - size, min(y1, y3) - size, x1 + size, min(y1, y3) + size), fill=color2)
					#draw.ellipse((x3 - size, max(y2, y4) - size, x3 + size, max(y2, y4) + size), fill=color2)
					char_array2 = np.zeros((char_array.shape[0], char_array.shape[1], 3))
					for y in np.arange(char_array.shape[0]):
						for x in np.arange(char_array.shape[1]):
							char_array2[y, x] = hex2rgb(char_array[y, x])
					char_array3 = char_array2.mean(axis=2)
					for y in np.arange(img2.shape[0] - char_array3.shape[0]):
						for x in np.arange(img2.shape[1] - char_array3.shape[1]):
							try: krn = (img2[y:y + char_array3.shape[0], x:x + char_array3.shape[1]]).mean(axis=2)
							except IndexError: krn = (img2[y:y + char_array3.shape[0], x:x + char_array3.shape[1]])
							if np.abs(np.subtract(krn, char_array3)).mean() < tolerance_threshold:
								print(char, x, y, np.abs(np.subtract(krn, char_array3)).mean())
								roi.append((min(x+x1, x1), min(y+y1, y1), max(x + char_array.shape[1] + x1, x2), min(y + char_array.shape[0] + y1, y2)))
			else:
				print(len(roi))
				target = 0
				count = 0
				for char in input_string:
					char_array = CHARACTER_ARRAYS[char]
					target += char_array.shape[1]
				for x1, y1, x2, y2 in roi:
					if target - (x2-x1) < 10:
						count += 1
						print(target, x2-x1, y2-y1)
						color = (255, 0, 0)
						draw.rectangle((x1, y1, x2, y2), outline=color)
	print(count)
	im.show()



#OCR('SRO', 'Q:/autopaper/Untitled4.png', tolerance_threshold=12)
#OCR('Unit', get_gradient('Q:/autopaper/Untitled.jpg'), tolerance_threshold=24)
now=datetime.datetime.today()
im2 = np.array(Image.open('Q:/autopaper/Untitled2.png').convert('L'))
#print(im2.shape)
im = Image.fromarray(get_gradient(im2))
#print(datetime.datetime.today()-now)
#im.show()
im.save('Untitled2.png')
# TODO: Handle multiple words separated by spaces
# COLOR= 108.622931 seconds
# GRAY=  39.256156 seconds
