__author__ = 'jsh0x'
__version__ = '1.0.0'

import os
from secrets import randbelow as rand
import cv2
import numpy as np
from PIL import Image, ImageDraw
from Convert import hex2rgb, index2coord, rgb2hex
from Constants import CHARACTER_ARRAYS

master_img = None

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
	ret_img = np.zeros((img.shape[0]-1, img.shape[1]-1, 3))
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
	return ret_img.mean(axis=2)



def OCR(input_string, haystack_image, tolerance_threshold=10):
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
	for char in input_string:
		img2 = np.copy(img)
		color = (rand(256), rand(256), rand(256))
		if len(retval_dict.keys()) > 0:
			for val in retval_dict.values():
				for x1,y1,x2,y2 in val:
					char_array = CHARACTER_ARRAYS[char]
					#cy,cx = y1+(y2/2), x1+(x2/2)
					img2 = np.copy(img[y1-(int(char_array.shape[0]/2)):y2+(int(char_array.shape[0]/2)), x1:x2+(char_array.shape[1]*2)])
		retval_dict[char] = []
		char_array = CHARACTER_ARRAYS[char]
		char_array2 = np.zeros((char_array.shape[0], char_array.shape[1], 3))
		for y in np.arange(char_array.shape[0]):
			for x in np.arange(char_array.shape[1]):
				char_array2[y,x]=hex2rgb(char_array[y,x])
		char_array3 = char_array2.mean(axis=2)-char_array2.mean(axis=2).min()
		global master_img
		char_array3 = master_img
		for y in np.arange(img2.shape[0]-char_array3.shape[0]):
			for x in np.arange(img2.shape[1]-char_array3.shape[1]):
				krn = (img2[y:y+char_array3.shape[0],x:x+char_array3.shape[1]]).mean(axis=2)
				krn-=krn.min()
				if np.abs(np.subtract(krn, char_array3)).mean() < tolerance_threshold:
					draw.rectangle((x, y, x + char_array.shape[1], y + char_array.shape[0]), outline=(255, 0, 0))
	im.show()
	"""maxx = 0
	maxy = 0
	minx = 1000000
	miny = 1000000
	print(list(retval_dict.values()))
	for val in retval_dict.values():
		for x1, y1, x2, y2 in val:
			minx = min(x1, minx)
			miny = min(y1, miny)
			maxx = max(x2, maxx)
			maxy = max(y2, maxy)
	draw.rectangle((minx, miny, maxx, maxy), outline=color)
	im.show()"""
	"""for i,row in enumerate(img[:]):  # splits image into rows
			row2 = np.copy(row)
			for j in np.arange(0, row2.shape[0]-char_array3.shape[0]):  # iterates through a section of the row equal to the width of the char
				sl = row2[j:j+char_array3.shape[0]-1]-row2[j:j+char_array3.shape[0]-1].min()  # equalizes minimum
				#print(sl.tolist())
				#print(char_array3[0]-char_array3[0].min().tolist())
				#print()
				if np.abs(np.subtract(sl, char_array3[0])) < tolerance_threshold:
					print(np.abs(np.subtract(sl, char_array3[0])))
					print(i,j)"""
	#if np.sum((np.abs(row2-rgb)), axis=1).min() < tolerance_threshold:
		#	j = np.sum((np.abs(img[i, :]-rgb)), axis=1).tolist().index(np.sum(np.abs(img[i, :]-rgb), axis=1).min())  # finds the index of the value row-wise
		#	val_list.append([i,j])

	"""for py,px in val_list:
			#break_out = False
			if np.average(np.subtract(img[py:py+char_array.shape[0], px-np.floor_divide(char_array.shape[1],2):px-np.floor_divide(char_array.shape[1],2)+char_array.shape[1]],char_array2)) >= tolerance_threshold:
				val_list.remove([py, px])
				break
		im = Image.fromarray(np.asarray(img, dtype=np.uint8))
		draw = ImageDraw.Draw(im)
		rm_list = []
		for y,x in val_list:
			for y2, x2 in val_list:
				if space_conflict((x-np.floor_divide(char_array.shape[1],2),y,(x-np.floor_divide(char_array.shape[1],2))+char_array.shape[1],y+char_array.shape[0]),
					              (x2-np.floor_divide(char_array.shape[1],2),y2,(x2-np.floor_divide(char_array.shape[1],2))+char_array.shape[1],y2+char_array.shape[0])) and x != x2 and y != y2:
					rm_list.append(((y,x),(y2,x2)))
		for yx1,yx2 in rm_list:
			y1,x1 = yx1
			y2,x2 = yx2
			try:
				val1 = np.average(np.subtract(img[y1:y1+char_array.shape[0], x1-np.floor_divide(char_array.shape[1],2):(x1-np.floor_divide(char_array.shape[1],2))+char_array.shape[1]], char_array2))
				val2 = np.average(np.subtract(img[y2:y2+char_array.shape[0], x2-np.floor_divide(char_array.shape[1],2):(x2-np.floor_divide(char_array.shape[1],2))+char_array.shape[1]], char_array2))
			except:pass
			else:
				if val1 > val2:
					try: val_list.remove(yx1)
					except: pass
				elif val2 > val1:
					try: val_list.remove(yx2)
					except: pass
				else:
					try:
						val_list.remove(yx1)
						val_list.remove(yx2)
					except: pass
		for y,x in val_list:
			draw.rectangle((x,y,x+char_array.shape[1],y+char_array.shape[0]),outline=(255,0,0))
		im.show()
		return
		#252,241
		#quit()"""
	"""temp_array = np.zeros((char_array.shape[0],char_array.shape[1],3))
		for y in np.arange(char_array.shape[0]):
			for x in np.arange(char_array.shape[1]):
				r,g,b = hex2rgb(char_array[y,x])
				temp_array[y,x,0] = r
				temp_array[y,x,1] = g
				temp_array[y,x,2] = b
		char_array = temp_array
		y1 = np.arange(0, img.shape[0]-char_array.shape[0])
		x1 = np.arange(0, img.shape[1]-char_array.shape[1])
		for y in y1:
			for x in x1:
				#if np.allclose(img[y:y + char_array.shape[0], x:x + char_array.shape[1], :], char_array):
				#	print(f"{char}@ ({x}, {y}, {x+char_array.shape[1]}, {y+char_array.shape[0]})")
					#print(char+"", x, x + char_array.shape[1], y, y + char_array.shape[0])
				#print(x,x+char_array.shape[1],y,y+char_array.shape[0],np.average(np.abs(np.subtract(img[y:y+char_array.shape[0], x:x+char_array.shape[1], :], char_array))))
				if np.average(np.abs(np.subtract(img[y:y+char_array.shape[0], x:x+char_array.shape[1], :], char_array))) < tolerance_thresh:
					retval_dict[char].append((x,y,x+char_array.shape[1],y+char_array.shape[0]))
					#print(f"{char}: ({x}, {y}, {x+char_array.shape[1]}, {y+char_array.shape[0]}) = {np.average(np.abs(np.subtract(img[y:y+char_array.shape[0], x:x+char_array.shape[1], :], char_array)))}")
	for k in retval_dict.keys():
		for k2 in retval_dict.keys():
			try:rm_dict = [(max((np.average(np.abs(np.subtract(img[c1[1]:c1[3], c1[0]:c1[2], :], char_array))), c1, k),(np.average(np.abs(np.subtract(img[c2[1]:c2[3], c2[0]:c2[2], :], char_array))), c2, k2))) for c1 in retval_dict[k] for c2 in retval_dict[k2] if space_conflict(c1, c2)]
			except: pass
	print(rm_dict)"""

#OCR('MDN', 'Q:/autopaper/Untitled.png')
#im_array = get_gradient('Q:/autopaper/Untitled.png')
#im = Image.fromarray(np.asarray(im_array, dtype=np.uint8))


char_array = CHARACTER_ARRAYS['S']
maxh = max(CHARACTER_ARRAYS['S'].shape[1], CHARACTER_ARRAYS['R'].shape[1], CHARACTER_ARRAYS['O'].shape[1])
print(maxh)
s = np.zeros((char_array.shape[0], maxh, 3))
for y in np.arange(char_array.shape[0]):
	for x in np.arange(char_array.shape[1]):
		s[y,x]=hex2rgb(char_array[y,x])

char_array = CHARACTER_ARRAYS['R']
r = np.zeros((char_array.shape[0], maxh, 3))
for y in np.arange(char_array.shape[0]):
	for x in np.arange(char_array.shape[1]):
		r[y,x]=hex2rgb(char_array[y,x])

char_array = CHARACTER_ARRAYS['O']
o = np.zeros((char_array.shape[0], maxh, 3))
for y in np.arange(char_array.shape[0]):
	for x in np.arange(char_array.shape[1]):
		o[y,x]=hex2rgb(char_array[y,x])
c = np.hstack((s,r,o))
print(s.shape, r.shape, o.shape)

dst = get_gradient(c)
master_img = dst
OCR('S', 'Q:/autopaper/Untitled4.png', 20)