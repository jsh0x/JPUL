__author__ = 'jsh0x'
__version__ = '2.0.0'

import os
import sys
from queue import Queue
from threading import Thread
from typing import Union, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw
from convert import hex2rgb
from constants import CHARACTER_ARRAYS, Numeric
from zmath import total_differential as t_diff, get_local_max as local_max, get_local_min as local_min
import datetime
from matplotlib import pyplot as plt



"""def kernel(image: np.ndarray, f: function, k: Tuple[int, int]=(3,3)) -> np.ndarray:
	border_x = np.floor_divide(k[0],2)
	border_y = np.floor_divide(k[1],2)
	retval = np.empty((np.subtract(image.shape[0],np.multiply(border_y, 2)), np.subtract(image.shape[1]-np.multiply(border_x, 2))))
	for y in np.arange(border_y, np.subtract(image.shape[0], border_y)):
		for x in np.arange(border_x, np.subtract(image.shape[1], border_x)):
			val = image[np.subtract(y,border_y):np.add(np.subtract(y,border_y),k[1]), np.subtract(x,border_x):np.add(np.subtract(x,border_x),k[0])].copy()
			retval[np.subtract(y,border_y), np.subtract(x,border_x)] = f(val)
	return retval"""

def show_image(image: np.ndarray, x: int = None, y: int = None, w: int = None, h: int = None, gray: bool = True):
	if not x or not y or not w or not h:
		if gray:
			plt.imshow(image, cmap='gray')
		elif not gray:
			im = Image.fromarray(image.astype(np.uint8))
			im.show()
	else:
		if gray:
			plt.imshow(image[y:y+h,x:x+w], cmap='gray')
		elif not gray:
			plt.imshow(image[y:y + h, x:x + w])
	plt.show()

def reduce_image_whitespace(image: np.ndarray, threshold: int = 32, kernel: Tuple[int, int] = (5, 5)):
	w = image.shape[1]
	h = image.shape[0]
	step = kernel[0]
	k = np.floor_divide(step, 2)
	slices = [(slice(y-k,y+k+1), slice(x-k,x+k+1)) for y in np.arange(k, h-k-1, step, dtype=np.intp) for x in np.arange(k, w-k-1, step, dtype=np.intp) if image[y-k:y+k+1,x-k:x+k+1].max() < threshold]
	for y,x in slices:
		image[y,x] = 0
	return image

def space_conflict(roi, roi_list):
	retval = [roi2 for roi2 in roi_list if (np.abs(np.subtract(roi[0],roi2[0])) <= roi[2]-roi[0]) and (np.abs(np.subtract(roi[1],roi2[1])) <= roi[3]-roi[1])]
	if len(retval) <= 1: return None
	else: return retval

def get_gradient(image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
	"""finds the anti-derivative of an image's RGB values"""
	if type(image) is str and not os.path.exists(image): raise FileNotFoundError(f"Image directory '{image}' not found")
	elif type(image) is str: img = np.array(Image.open(image).convert('RGB'), dtype=np.int16)
	elif type(image) is Image.Image:
		if image.mode == 'RGB': img = np.array(image.convert('L'), dtype=np.int16)
		elif image.mode == 'L': img = np.array(image, dtype=np.int16)
	elif type(image) is np.ndarray: img = image
	else: raise TypeError(f"Value 'image' must be a PIL Image-Object, an image directory, or an image array, instead got {type(image)}")
	#ret_img = t_diff(img)
	#return ret_img
	ret_img = np.empty((img.shape[0]-2, img.shape[1]-2, 4))
	h,w = img.shape
	img2 = img[1:h-1,1:w-1]
	ret_img[...,0] = np.absolute(np.subtract(img2, img[1:h-1,2:]))
	ret_img[...,1] = np.absolute(np.subtract(img2, img[1:h-1,:w-2]))
	ret_img[...,2] = np.absolute(np.subtract(img2, img[2:,1:w-1]))
	ret_img[...,3] = np.absolute(np.subtract(img2, img[:h-2,1:w-1]))
	ret_img = np.mean(ret_img, axis=2)
	return np.rint(np.where((ret_img-ret_img.astype(int)) >= 0.5, ret_img+0.5, ret_img))
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

def get_bounds(roi_list):
	minx = sys.maxsize
	miny = sys.maxsize
	maxx = 0
	maxy = 0
	for x1,y1,x2,y2 in roi_list:
		minx = min(minx, x1)
		miny = min(miny, y1)
		maxx = max(maxx, x2)
		maxy = max(maxy, y2)
	return tuple((minx, miny, maxx, maxy))

def OCR(input_string, haystack_image, tolerance_threshold=10, roi=None):
	#HIGHLY recommend tolerance thresholds < ?
	if type(input_string) is not str: raise TypeError(f"Value 'input_string' must be type str, instead got type {type(input_string)}")
	if len(input_string) == 0: raise ValueError("Value 'input_string' is empty")
	if type(haystack_image) is str and not os.path.exists(haystack_image): raise FileNotFoundError(f"Image directory '{haystack_image}' not found")
	elif type(haystack_image) is str: img = np.array(Image.open(haystack_image).convert('L'), dtype=np.int16)
	elif type(haystack_image) is Image.Image: img = np.array(haystack_image.convert('L'), dtype=np.int16)
	elif type(haystack_image) is np.ndarray: img = np.asarray(haystack_image, dtype=np.int16)
	else: raise TypeError(f"Value 'haystack_image' must be a PIL Image-Object, an image directory, or an image array, instead got {type(haystack_image)}")
	retval_dict = {}
	master_list = []
	old = tolerance_threshold
	im = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
	draw = ImageDraw.Draw(im)
	for char in input_string:
		if roi is None:
			img2 = np.copy(img)
			char_array = CHARACTER_ARRAYS[char]
			for y in np.arange(img2.shape[0]-char_array.shape[0]):
				for x in np.arange(img2.shape[1]-char_array.shape[1]):
					krn = (img2[y:y+char_array.shape[0],x:x+char_array.shape[1]])
					if np.abs(np.subtract(krn, char_array)).mean() < tolerance_threshold:
						print(char, x, y, np.abs(np.subtract(krn, char_array)).mean())
						if roi is None: roi = [(x, y, x + char_array.shape[1], y + char_array.shape[0])]
						else: roi.append((x, y, x + char_array.shape[1], y + char_array.shape[0]))
			roi2 = roi
		else:

			if char == 'l' or char == 'I':
				tolerance_threshold = 20
			else:
				tolerance_threshold = old
			"""
			im2 = Image.fromarray(haystack_image).convert("RGB")
			draw = ImageDraw.Draw(im2)"""
			#378,417
			"""print(len(roi))
			conflict_set = set()
			for roi2 in roi:
				res = space_conflict(roi2, roi)
				if res is not None:
					conflict_set.add(tuple(res))"""
			#print(conflict_set)
			"""print(len(conflict_set))
			for sc in conflict_set:
				bnd = get_bounds(sc)
				for s in sc:
					try: roi.remove(s)
					except: pass
				roi.append(bnd)
			print(len(roi))"""
			"""
			for x1,y1,x2,y2 in roi:
				draw.rectangle((x1,y1,x2,y2), outline=(255, 0, 0))
			#im2.show()"""
			roi2 = []
			for x1, y1, x2, y2 in roi:
				print(x1,y1,x2,y2)
				#color = (rand(256), rand(256), rand(256))
				#color2 = (rand(256), rand(256), rand(256))
				#size=2
				#rem_list.append((x1, y1, x2, y2))
				char_array = CHARACTER_ARRAYS[char]
				y4 = y2 + char_array.shape[0]
				y3 = y2 - char_array.shape[0]
				x4 = np.add(x2, char_array.shape[1])
				x3 = np.subtract(x2, np.floor_divide(char_array.shape[1], 2))
				img2 = np.copy(img[y3:y4, x3:x4])
				# img2 = np.copy(img[min(y1,y3):max(y2,y4), x1:x4])
				#print(min(y1,y3),max(y2,y4), x1,x4)
				#draw.ellipse((x1 - size, y1 - size, x1 + size, y1 + size), fill=color)
				#draw.ellipse((x2 - size, y2 - size, x2 + size, y2 + size), fill=color)
				#draw.ellipse((x1 - size, min(y1, y3) - size, x1 + size, min(y1, y3) + size), fill=color2)
				#draw.ellipse((x3 - size, max(y2, y4) - size, x3 + size, max(y2, y4) + size), fill=color2)
				for y in np.arange(img2.shape[0]-char_array.shape[0]):
					for x in np.arange(img2.shape[1] - char_array.shape[1]):
						krn = (img2[y:y+char_array.shape[0], x:x+char_array.shape[1]])
						if krn.shape == char_array.shape:
							if np.abs(np.subtract(krn, char_array)).mean() < tolerance_threshold:
								print(char, x, y, np.abs(np.subtract(krn, char_array)).mean(), (x2-x, y2-y, (x2-x)+char_array.shape[1], (y2-y)-char_array.shape[0]))
								roi2.append((x2-x, y2-y, (x2-x)+char_array.shape[1], (y2-y)-char_array.shape[0]))
				#for y in np.arange(img2.shape[0]-char_array.shape[0]):
				#	krn = (img2[y:y + char_array.shape[0], :char_array.shape[1]])
				#	if np.abs(np.subtract(krn, char_array)).mean() < tolerance_threshold:
				#		print(char, x, y, np.abs(np.subtract(krn, char_array)).mean())
				#		roi2.append((x2, y + y2, x4, y + char_array.shape[0] + y2,))
		"""conflict_set = set()
		for roi3 in roi2:
			res = space_conflict(roi3, roi2)
			if res is not None:
				conflict_set.add(tuple(res))
		for sc in conflict_set:
			bnd = get_bounds(sc)
			for s in sc:
				try:
					roi2.remove(s)
				except:
					pass
			roi2.append(bnd)"""
		master_list.append(roi2)
		roi = roi2
	conflict_set = set()
	print(len(master_list))
	"""for roi in master_list:
		for roi2 in roi:
			res = space_conflict(roi2, roi)
			if res is not None:
				conflict_set.add(tuple(res))
		for sc in conflict_set:
			bnd = get_bounds(sc)
			for s in sc:
				try: roi.remove(s)
				except: pass
			roi.append(bnd)"""
	print(len(master_list))
	count = 0
	for roi in master_list:
		count+= 1
		print(count)
		for x1, y1, x2, y2 in roi:
			color = (255, 0, 0)
			print(x1,y1,x2,y2)
			draw.rectangle((x1, y1, x2, y2), outline=color)
	im.show()

def get_diagonals(a: np.ndarray, flipped=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	tmp = np.zeros_like(a, dtype=np.int32)
	lmx = tmp.copy()
	lmn = tmp.copy()
	lmx2 = tmp.copy()
	lmn2 = tmp.copy()
	if flipped:
		a = np.fliplr(a)
	if a.shape[0] < a.shape[1]:
		for i in np.arange(a.shape[0] - 3, 0, -1, dtype=np.intp):
			diagonal = np.diagonal(a[i:, :a.shape[0] - i])
			di = np.diag_indices(a[i:, :a.shape[0] - i].shape[0])
			lmx[i:, :a.shape[0] - i][di] = local_max(diagonal)[0]
			lmn[i:, :a.shape[0] - i][di] = local_min(diagonal)[0]
			lmx2[i:, :a.shape[0] - i][di] = np.where(np.equal(diagonal, np.max(diagonal)), 1, 0)
			lmn2[i:, :a.shape[0] - i][di] = np.where(np.equal(diagonal, np.min(diagonal)), 1, 0)
		for i in np.arange(a.shape[1] - a.shape[0] + 1, dtype=np.intp):
			diagonal = np.diagonal(a[:, i:a.shape[0] + i])
			di = np.diag_indices(a[:, i:a.shape[0] + i].shape[0])
			lmx[:, i:a.shape[0] + i][di] = local_max(diagonal)[0]
			lmn[:, i:a.shape[0] + i][di] = local_min(diagonal)[0]
			lmx2[:, i:a.shape[0] + i][di] = np.where(np.equal(diagonal, np.max(diagonal)), 1, 0)
			lmn2[:, i:a.shape[0] + i][di] = np.where(np.equal(diagonal, np.min(diagonal)), 1, 0)
		for i in np.arange(a.shape[0] - 1, 2, -1, dtype=np.intp):
			diagonal = np.diagonal(a[:i, a.shape[1] - i:])
			di = np.diag_indices(a[:i, a.shape[1] - i:].shape[0])
			lmx[:i, a.shape[1] - i:][di] = local_max(diagonal)[0]
			lmn[:i, a.shape[1] - i:][di] = local_min(diagonal)[0]
			lmx2[:i, a.shape[1] - i:][di] = np.where(np.equal(diagonal, np.max(diagonal)), 1, 0)
			lmn2[:i, a.shape[1] - i:][di] = np.where(np.equal(diagonal, np.min(diagonal)), 1, 0)
	elif a.shape[0] > a.shape[1]:
		for i in np.arange(a.shape[1] - 3, 0, -1, dtype=np.intp):
			diagonal = np.diagonal(a[:a.shape[1] - i, i:])
			di = np.diag_indices(a[:a.shape[1] - i, i:].shape[0])
			lmx[:a.shape[1] - i, i:][di] = local_max(diagonal)[0]
			lmn[:a.shape[1] - i, i:][di] = local_min(diagonal)[0]
			lmx2[:a.shape[1] - i, i:][di] = np.where(np.equal(diagonal, np.max(diagonal)), 1, 0)
			lmn2[:a.shape[1] - i, i:][di] = np.where(np.equal(diagonal, np.min(diagonal)), 1, 0)
		for i in np.arange(a.shape[0] - a.shape[1] + 1, dtype=np.intp):
			diagonal = np.diagonal(a[i:a.shape[1] + i, :])
			di = np.diag_indices(a[i:a.shape[1] + i, :].shape[0])
			lmx[i:a.shape[1] + i, :][di] = local_max(diagonal)[0]
			lmn[i:a.shape[1] + i, :][di] = local_min(diagonal)[0]
			lmx2[i:a.shape[1] + i, :][di] = np.where(np.equal(diagonal, np.max(diagonal)), 1, 0)
			lmn2[i:a.shape[1] + i, :][di] = np.where(np.equal(diagonal, np.min(diagonal)), 1, 0)
		for i in np.arange(a.shape[1] - 1, 2, -1, dtype=np.intp):
			diagonal = np.diagonal(a[a.shape[0] - i:, :i])
			di = np.diag_indices(a[a.shape[0] - i:, :i].shape[0])
			lmx[a.shape[0] - i:, :i][di] = local_max(diagonal)[0]
			lmn[a.shape[0] - i:, :i][di] = local_min(diagonal)[0]
			lmx2[a.shape[0] - i:, :i][di] = np.where(np.equal(diagonal, np.max(diagonal)), 1, 0)
			lmn2[a.shape[0] - i:, :i][di] = np.where(np.equal(diagonal, np.min(diagonal)), 1, 0)
	if flipped:
		lmx = np.fliplr(lmx)
		lmn = np.fliplr(lmn)
		lmx2 = np.fliplr(lmx2)
		lmn2 = np.fliplr(lmn2)
	return lmx, lmn, lmx2, lmn2

"""
im = Image.open('Q:/autopaper/Untitled2.png').convert('RGB')
im2 = get_gradient(im)

d = np.floor_divide(np.hypot(im2.shape[0],im2.shape[1]), 2)
print(np.hypot(im2.shape[0], im2.shape[1]))
for i in np.arange(-im2.shape[0]+1, im2.shape[1], dtype=np.intp):
	for j in np.ndenumerate(np.diagonal(im2, i)):
		print(i,j)
quit()"""

#print(im2.shape)
#OCR("General", im2, 31)

#im = Image.fromarray(get_gradient(im2))
#print(datetime.datetime.today()-now)
#im.show()
#im.save('Untitled2.png')
# TODO: Handle multiple words separated by spaces
# COLOR= 108.622931 seconds
# GRAY=  39.256156 seconds

def temp_thread(chunk, xy, h, w, min_dim, lmxh, lmnh, lmxv, lmnv, lmxd, lmnd, threshold, char_array, q):
	for y in np.arange(h, chunk.shape[0] - h):
		for x in np.arange(w, chunk.shape[1] - w):
			array = chunk[y, x - w:x - w + char_array.shape[1]]
			lmxh2 = np.array(local_max(array,legacy=True))
			lmnh2 = np.array(local_min(array,legacy=True))
			#TODO: HERE TOO
			if (lmxh2[0].shape[0] == lmxh.shape[0] and lmnh2[0].shape[0] == lmnh.shape[0]) and (
				np.less_equal(np.abs(np.sum(np.subtract(lmxh2[0], lmxh))), lmxh.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnh2[0], lmnh))), lmnh.shape[0])):
				array = chunk[y - h:y - h + char_array.shape[0], x]
				lmxv2 = np.array(local_max(array,legacy=True))
				lmnv2 = np.array(local_min(array,legacy=True))

				if (lmxv2[0].shape[0] == lmxv.shape[0] and lmnv2[0].shape[0] == lmnv.shape[0]) and (
					np.less_equal(np.abs(np.sum(np.subtract(lmxv2[0], lmxv))), lmxv.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnv2[0], lmnv))), lmnv.shape[0])):
					array = chunk[y - min_dim:y - min_dim + char_array.shape[0], x - min_dim:x - min_dim + char_array.shape[1]]
					lmxd2 = np.array(local_max(array,legacy=True))
					lmnd2 = np.array(local_min(array,legacy=True))

					if (lmxd2[0].shape[0] == lmxd.shape[0] and lmnd2[0].shape[0] == lmnd.shape[0]) and (
						np.less_equal(np.abs(np.sum(np.subtract(lmxd2[0], lmxd))), lmxd.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnd2[0], lmnd))), lmnd.shape[0])):
						array = chunk[y - h:y - h + char_array.shape[0], x - w:x - w + char_array.shape[1]]
						if np.less(np.mean(np.abs(np.subtract(array, char_array))), threshold):
							q.put((x, y))

def dev(input_string: str, haystack_image, threshold=0.8, roi=None, bold=False):
	string_width = 0
	string_height = 8
	contains_upper = False
	contains_under = False
	for i in range(len(input_string)):
		char = input_string[i]
		val_h = CHARACTER_ARRAYS[char].shape[0]
		val_w = CHARACTER_ARRAYS[char].shape[1]
		if val_h >= 10: contains_upper = True
		if char == "y" or char == "j" or char == "q" or char == "p" or char == "g": contains_under = True
		elif not bold:
			if i == 0: string_width += val_w
			elif input_string[i-1].isupper(): string_width += val_w
			elif input_string[i-1].islower(): string_width += (val_w-1)
			elif input_string[i-1].isnumeric(): string_width += (val_w - 1)
			else: string_width += (val_w - 1)
		else: string_width += (val_w-1)
		print(string_width)
	if contains_upper: string_height += 3
	if contains_under: string_height += 2

	needle_image = np.zeros((string_height, string_width), dtype=np.int16)
	next_point = 0
	for i in range(len(input_string)):
		char = input_string[i]
		val_h = CHARACTER_ARRAYS[char].shape[0]
		val_w = CHARACTER_ARRAYS[char].shape[1]
		if val_h == 10: base_h = 1
		elif val_h == 11: base_h = 0
		elif contains_upper: base_h = 3
		else: base_h = 0

		print(char, next_point, next_point+val_w)
		needle_image[base_h:base_h+val_h, next_point:next_point+val_w] = np.add(needle_image[base_h:base_h+val_h, next_point:next_point+val_w], CHARACTER_ARRAYS[char])
		if not bold and char.isupper(): next_point += val_w#-1
		elif i < (len(input_string)-1) and bold:
			if CHARACTER_ARRAYS[input_string[i]].shape[1] < 7: next_point += (val_w-2)
			elif input_string[i] == '0': next_point += (val_w-2)
			else: next_point += (val_w-1)
		else: next_point += (val_w-1)
	w, h = needle_image.shape[::-1]
	if bold:
		needle_image = np.where(needle_image[:,:] > 0, needle_image[:,:]+16, needle_image[:,:])
		needle_image = np.where(needle_image[:, :] > 128, needle_image[:, :] + 16, needle_image[:,:])
	#for y, x in np.ndindex(haystack_image.shape[0] - h, haystack_image.shape[1] - w):
	#	if np.mean(np.abs(np.subtract(haystack_image[y:y + h, x:x + w], needle_image))) < 37:
	#		print(np.mean(np.abs(np.subtract(haystack_image[y:y + h, x:x + w], needle_image))))
	res = np.array([(x,y,w,h) for y,x in np.ndindex(haystack_image.shape[0]-h, haystack_image.shape[1]-w) if np.mean(np.abs(np.subtract(haystack_image[y:y + h, x:x + w], needle_image)))<threshold])
	for x,y,w,h in res:
		print(np.mean(np.abs(np.subtract(haystack_image[y:y + h, x:x + w], needle_image))), np.abs(np.mean(np.subtract(haystack_image[y:y + h, x:x + w], needle_image))))
		img = np.empty((h*2, w), dtype=np.int16)
		img2 = haystack_image[y:y + h, x:x + w]
		img3 = needle_image
		print("ROWS")
		for row,row2 in zip(img2, img3):
			print(np.mean(row))
			print(np.mean(row2))
			print(np.subtract(np.mean(row), np.mean(row2)))
			print()
		print("COLUMNS")
		for col,col2 in zip(img2[:], img3[:]):
			print(np.mean(col))
			print(np.mean(col2))
			print(np.subtract(np.mean(col), np.mean(col2)))
			print()
		img[:h] = img2
		img[h:] = img3
		#for i in range(3):
		#	print(np.abs(np.subtract(img[4+i,:], needle_image[4+i,:])).tolist())
		#print()
		plt.imshow(img, cmap='gray')
		plt.show()
	quit()
	k2 = []
	for xy,roi in k2:
		roi_list = [xy]
		for char in input_string[1:]:
			needle_image = CHARACTER_ARRAYS[char]
			w, h = needle_image.shape[::-1]
			for y, x in np.ndindex(roi.shape[0], roi.shape[1] - w):
				print(np.mean(np.abs(np.subtract(roi[y:y + h, x:x + w], needle_image))))
			k3 = np.array([((xy[0]+(xy[2]-1)+x,xy[1]+y,w,h), roi[:,x+(w-1):]) for y, x in np.ndindex(roi.shape[0], roi.shape[1]-w) if np.mean(np.abs(np.subtract(roi[y:y + h, x:x + w], needle_image))) < threshold])
			for a in k3:
				roi_list.append(a)
			#plt.imshow(k, cmap='gray')
			#plt.show()
	print(roi_list)



	quit()
	if roi is None:
		roi = []
		res = cv2.matchTemplate(haystack_image, needle_image, cv2.TM_CCOEFF_NORMED)
		loc = np.where(res >= threshold)
		for pt in zip(*loc[::-1]):
			roi.append([(pt[0]+(w/2), (pt[1]+(h/2))-CHARACTER_ARRAYS[input_string[i+1]].shape[0],
		                (pt[0]+(w/2))+(CHARACTER_ARRAYS[input_string[i+1]].shape[1]*2),
		                (pt[1]+(h/2))+CHARACTER_ARRAYS[input_string[i+1]].shape[0]),
		                (pt[0], pt[1], pt[0]+w, pt[1]+h)])
		if not roi: return None
	else:
		roi3 = []
		for roi2 in roi:
			print(roi2)
			x1,y1,x2,y2 = roi2[0]
			old = roi2[1:]
			print(old)
			if x1 < 0:
				x1 = 0
			if y1 < 0:
				y1 = 0
			res = cv2.matchTemplate(haystack_image[int(y1):int(y2), int(x1):int(x2)], needle_image, cv2.TM_CCOEFF_NORMED)
			print(char, res.max())
			loc = np.where(res >= threshold)
			for pt in zip(*loc[::-1]):
				plt.imshow(haystack_image[int(pt[1]+y1):int(pt[1]+y1 + h), int(pt[0]+x1):int(pt[0]+x1 + w)], cmap='gray')
				#plt.show()
				roi3.append([(pt[0]+x1 + (w / 2), (pt[1]+y1 + (h / 2)) - CHARACTER_ARRAYS[input_string[i + 1]].shape[0],
				            (pt[0]+x1 + (w / 2)) + (CHARACTER_ARRAYS[input_string[i + 1]].shape[1] * 3),
				            (pt[1]+y1 + (h / 2)) + CHARACTER_ARRAYS[input_string[i + 1]].shape[0]),
				             old+[(int(pt[0]+x1), int(pt[1]+y1), int(pt[0]+x1 + w), int(pt[1]+y1 + h))]])
		roi = roi3
		if not roi: return None
	if i == (len(input_string)-2):
		return roi
"""im = np.array(Image.open("figure_1.png"), dtype=np.int16)
im2 = np.array(Image.open("figure_2.png"), dtype=np.int16)
im2 = np.where(im2[:,:] > 0, im2[:,:]+16, 0)
im3 = np.abs(np.subtract(im,im2))
im4 = np.mean(np.abs(np.subtract(im,im2)))
for row,row2 in zip(im,im2):
	print(row.mean())
	print(row2.mean())
	print()
print(np.mean(im), np.mean(im2))"""
def dev2(input_string: str, haystack_image: np.ndarray, threshold: int, max_distance=5):
	retval = {}
	master_vals = {}
	for i in np.arange(len(input_string)):
		char = input_string[i]
		char_array = CHARACTER_ARRAYS[char]
		h = np.floor_divide(char_array.shape[0], 2)
		w = np.floor_divide(char_array.shape[1], 2)
		min_dim = min(h, w)
		di = np.diag_indices(min_dim)
		lmxh = np.array(local_max(char_array[h]))
		lmnh = np.array(local_min(char_array[h]))
		lmxv = np.array(local_max(char_array[:, w]))
		lmnv = np.array(local_min(char_array[:, w]))
		lmxd = np.array(local_max(char_array[h - min_dim:h + min_dim, w - min_dim:w + min_dim][di]))
		lmnd = np.array(local_min(char_array[h - min_dim:h + min_dim, w - min_dim:w + min_dim][di]))
		if i > 0:
			prev_char = input_string[i-1]
			prev_char_array = CHARACTER_ARRAYS[prev_char]
			ph = np.floor_divide(prev_char_array.shape[0], 2)
			pw = np.floor_divide(prev_char_array.shape[1], 2)
			for x,y in master_vals[prev_char]:
				roi = haystack_image[y-ph-h:y+ph+h,x+pw-1:x+pw+(3*w)]
				show_image(roi)
				for y in np.arange(h, roi.shape[0] - h):
					for x in np.arange(w, roi.shape[1] - w):
						array = roi[y, x - w:x - w + char_array.shape[1]]
						lmxh2 = np.array(local_max(array))
						lmnh2 = np.array(local_min(array))

						if (lmxh2.shape[0] == lmxh.shape[0] and lmnh2.shape[0] == lmnh.shape[0]) and (np.less_equal(np.abs(np.sum(np.subtract(lmxh2, lmxh))), lmxh.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnh2, lmnh))), lmnh.shape[0])):
							array = roi[y - h:y - h + char_array.shape[0], x]
							lmxv2 = np.array(local_max(array))
							lmnv2 = np.array(local_min(array))

							if (lmxv2.shape[0] == lmxv.shape[0] and lmnv2.shape[0] == lmnv.shape[0]) and (np.less_equal(np.abs(np.sum(np.subtract(lmxv2, lmxv))), lmxv.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnv2, lmnv))), lmnv.shape[0])):
								array = roi[y - min_dim:y - min_dim + char_array.shape[0], x - min_dim:x - min_dim + char_array.shape[1]][di]
								lmxd2 = np.array(local_max(array))
								lmnd2 = np.array(local_min(array))

								if (lmxd2.shape[0] == lmxd.shape[0] and lmnd2.shape[0] == lmnd.shape[0]) and (np.less_equal(np.abs(np.sum(np.subtract(lmxd2, lmxd))), lmxd.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnd2, lmnd))), lmnd.shape[0])):
									array = roi[y - h:y - h + char_array.shape[0], x - w:x - w + char_array.shape[1]]
									if np.less(np.mean(np.abs(np.subtract(array, char_array))), threshold):
										master_vals[char].append((x, y))
		else:
			step_x = np.floor_divide(haystack_image.shape[1], 6)
			step_y = np.floor_divide(haystack_image.shape[0], 6)
			q = Queue()
			for i in np.arange(0, haystack_image.shape[0], step_y):
				for j in np.arange(0, haystack_image.shape[1], step_x):
					section = haystack_image[i:i+step_y, j:j+step_x].view()
					if section.shape[0] > char_array.shape[0] and section.shape[1] > char_array.shape[1]:
						worker = Thread(target=temp_thread, args=(section,(i,j), h, w, min_dim, lmxh, lmnh, lmxv, lmnv, lmxd, lmnd, threshold, char_array, q))
						worker.setDaemon(True)
						worker.start()
			worker.join()
			while not q.empty():
				print(q.get())


			#TODO: HERE
			quit()
			for y in np.arange(h, haystack_image.shape[0] - h):
				for x in np.arange(w, haystack_image.shape[1] - w):
					array = haystack_image[y, x - w:x - w + char_array.shape[1]]
					lmxh2 = np.array(local_max(array))
					lmnh2 = np.array(local_min(array))
				
					if (lmxh2.shape[0] == lmxh.shape[0] and lmnh2.shape[0] == lmnh.shape[0]) and (np.less_equal(np.abs(np.sum(np.subtract(lmxh2, lmxh))), lmxh.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnh2, lmnh))), lmnh.shape[0])):
						array = haystack_image[y - h:y - h + char_array.shape[0], x]
						lmxv2 = np.array(local_max(array))
						lmnv2 = np.array(local_min(array))

						if (lmxv2.shape[0] == lmxv.shape[0] and lmnv2.shape[0] == lmnv.shape[0]) and (np.less_equal(np.abs(np.sum(np.subtract(lmxv2, lmxv))), lmxv.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnv2, lmnv))), lmnv.shape[0])):
							array = haystack_image[y - min_dim:y - min_dim + char_array.shape[0], x - min_dim:x - min_dim + char_array.shape[1]][di]
							lmxd2 = np.array(local_max(array))
							lmnd2 = np.array(local_min(array))

							if (lmxd2.shape[0] == lmxd.shape[0] and lmnd2.shape[0] == lmnd.shape[0]) and (np.less_equal(np.abs(np.sum(np.subtract(lmxd2, lmxd))), lmxd.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnd2, lmnd))), lmnd.shape[0])):
								array = haystack_image[y - h:y - h + char_array.shape[0], x - w:x - w + char_array.shape[1]]
								if np.less(np.mean(np.abs(np.subtract(array, char_array))), threshold):
									master_vals[char].append((x, y))
	return master_vals
	master_vals2 = {}
	for i in range(len(master_vals.keys())-1):
		k1 = list(master_vals.keys())[i]
		k2 = list(master_vals.keys())[i+1]
		for x1,y1 in master_vals[k1]:
			char_array1 = CHARACTER_ARRAYS[k1]
			w1 = np.floor_divide(char_array1.shape[1], 2)
			h1 = np.floor_divide(char_array1.shape[0], 2)
			for x2,y2 in sorted(master_vals[k2]):
				char_array2 = CHARACTER_ARRAYS[k2]
				w2 = np.floor_divide(char_array2.shape[1], 2)
				h2 = np.floor_divide(char_array2.shape[0], 2)
				x_dist = np.subtract(x2-w2, x1-w1+char_array1.shape[1])
				y_dist = np.subtract(y2, y1)
				print(x1, y1)
				print(x2, y2)
				print(x_dist, y_dist, np.hypot(x_dist, y_dist))
				print()
				if np.hypot(x_dist, y_dist) <= max_distance:
					pass

	return master_vals