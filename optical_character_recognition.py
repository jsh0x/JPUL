__author__ = 'jsh0x'
__version__ = '3.0.0'

import sqlite3 as sql
from typing import Union
from string import digits,ascii_letters,punctuation
import numpy as np
from zmath import get_local_max as local_max, get_local_min as local_min
#from image import get_gradient


#Global Variables
CHARACTERS = digits+ascii_letters+punctuation
DIRECTIONS = ('horizontal', 'vertical', 'diagonal')
"""conn = sql.connect('character_info.db')
c = conn.cursor()
c.execute('DROP TABLE character_arrays')
c.execute('DROP TABLE character_shapes')
c.execute('CREATE TABLE character_arrays (char, image_array, direction, local_max, local_min, local_max2, local_min2, global_max, global_min)')
c.execute('CREATE TABLE character_shapes (char, shape)')
conn.commit()
conn.close()"""
quit()

def remember() -> dict:
	retdict = {}
	for char in CHARACTERS:
		retdict[char] = {'shape': None, 'local_maxes': None, 'local_mins': None}
		conn = sql.connect('character_info.db')
		c = conn.cursor()
		c.execute(f"SELECT image_array,local_maxes,local_mins FROM character_arrays WHERE char = '{char}'")
		sample_list = tuple(c.fetchall())
		conn.close()

		image_array = np.zeros_like(sample_list[0][0], dtype=np.int32)
		local_max_array = np.zeros_like(sample_list[0][0], dtype=np.int32)
		for i,(array,lmx,lmn) in enumerate(sample_list):
			full_array += np.array(sample)
			pass
	return retval


def teach(char: str, image: np.ndarray):
	if type(char) is not str: raise TypeError
	if len(char) > 1: raise ValueError
	conn = sql.connect('character_info.db')
	c = conn.cursor()
	c.execute(f"SELECT shape FROM character_shapes WHERE char = '{char}'")
	shape = tuple(c.fetchone())
	conn.close()
	if image.shape != shape: raise IndexError
	tmp = np.zeros_like(image, dtype=np.int32)
	gmx = np.where(np.equal(image, np.max(image.flatten())), 1, 0)
	gmn = np.where(np.equal(image, np.min(image.flatten())), 1, 0)
	for direction in DIRECTIONS:
		lmx = tmp.copy()
		lmn = tmp.copy()
		lmx2 = tmp.copy()
		lmn2 = tmp.copy()
		if direction == 'horizontal':
			for i in np.arange(image.shape[0], dtype=np.intp):
				row = image[i]
				lmx[i] = local_max(row)[0]
				lmn[i] = local_min(row)[0]
				lmx2[i] = np.where(np.equal(row, np.max(row)), 1, 0)
				lmn2[i] = np.where(np.equal(row, np.min(row)), 1, 0)
		elif direction == 'vertical':
			for i in np.arange(image.shape[1], dtype=np.intp):
				column = image[:,i]
				lmx[i] = local_max(column)[0]
				lmn[i] = local_min(column)[0]
				lmx2[i] = np.where(np.equal(column, np.max(column)), 1, 0)
				lmn2[i] = np.where(np.equal(column, np.min(column)), 1, 0)
		elif direction == 'diagonal':
			for i in np.arange(image.shape[1], dtype=np.intp):
				column = image[:,i]
				lmx[i] = local_max(column)[0]
				lmn[i] = local_min(column)[0]
				lmx2[i] = np.where(np.equal(column, np.max(column)), 1, 0)
				lmn2[i] = np.where(np.equal(column, np.min(column)), 1, 0)

		conn = sql.connect('character_info.db')
		c = conn.cursor()
		c.execute(f"INSERT INTO character_arrays VALUES ('{char}', {image}, '{direction}', {lmx}, {lmn}, {lmx2}, {lmn2}, {gmx}, {gmn})")
		conn.commit()
		conn.close()
#char, image_array, direction, local_max, local_min, local_max2, local_min2, global_max, global_min

#def test():
#	pass


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
			prev_char = input_string[i - 1]
			prev_char_array = CHARACTER_ARRAYS[prev_char]
			ph = np.floor_divide(prev_char_array.shape[0], 2)
			pw = np.floor_divide(prev_char_array.shape[1], 2)
			for x, y in master_vals[prev_char]:
				roi = haystack_image[y - ph - h:y + ph + h, x + pw - 1:x + pw + (3 * w)]
				show_image(roi)
				for y in np.arange(h, roi.shape[0] - h):
					for x in np.arange(w, roi.shape[1] - w):
						array = roi[y, x - w:x - w + char_array.shape[1]]
						lmxh2 = np.array(local_max(array))
						lmnh2 = np.array(local_min(array))

						if (lmxh2.shape[0] == lmxh.shape[0] and lmnh2.shape[0] == lmnh.shape[0]) and (
							np.less_equal(np.abs(np.sum(np.subtract(lmxh2, lmxh))), lmxh.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnh2, lmnh))), lmnh.shape[0])):
							array = roi[y - h:y - h + char_array.shape[0], x]
							lmxv2 = np.array(local_max(array))
							lmnv2 = np.array(local_min(array))

							if (lmxv2.shape[0] == lmxv.shape[0] and lmnv2.shape[0] == lmnv.shape[0]) and (
								np.less_equal(np.abs(np.sum(np.subtract(lmxv2, lmxv))), lmxv.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnv2, lmnv))), lmnv.shape[0])):
								array = roi[y - min_dim:y - min_dim + char_array.shape[0], x - min_dim:x - min_dim + char_array.shape[1]][di]
								lmxd2 = np.array(local_max(array))
								lmnd2 = np.array(local_min(array))

								if (lmxd2.shape[0] == lmxd.shape[0] and lmnd2.shape[0] == lmnd.shape[0]) and (
									np.less_equal(np.abs(np.sum(np.subtract(lmxd2, lmxd))), lmxd.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnd2, lmnd))), lmnd.shape[0])):
									array = roi[y - h:y - h + char_array.shape[0], x - w:x - w + char_array.shape[1]]
									if np.less(np.mean(np.abs(np.subtract(array, char_array))), threshold):
										master_vals[char].append((x, y))
		else:
			step_x = np.floor_divide(haystack_image.shape[1], 6)
			step_y = np.floor_divide(haystack_image.shape[0], 6)
			q = Queue()
			for i in np.arange(0, haystack_image.shape[0], step_y):
				for j in np.arange(0, haystack_image.shape[1], step_x):
					section = haystack_image[i:i + step_y, j:j + step_x].view()
					if section.shape[0] > char_array.shape[0] and section.shape[1] > char_array.shape[1]:
						worker = Thread(target=temp_thread, args=(section, (i, j), h, w, min_dim, lmxh, lmnh, lmxv, lmnv, lmxd, lmnd, threshold, char_array, q))
						worker.setDaemon(True)
						worker.start()
			worker.join()
			while not q.empty():
				print(q.get())

			# TODO: HERE
			quit()
			for y in np.arange(h, haystack_image.shape[0] - h):
				for x in np.arange(w, haystack_image.shape[1] - w):
					array = haystack_image[y, x - w:x - w + char_array.shape[1]]
					lmxh2 = np.array(local_max(array))
					lmnh2 = np.array(local_min(array))

					if (lmxh2.shape[0] == lmxh.shape[0] and lmnh2.shape[0] == lmnh.shape[0]) and (
						np.less_equal(np.abs(np.sum(np.subtract(lmxh2, lmxh))), lmxh.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnh2, lmnh))), lmnh.shape[0])):
						array = haystack_image[y - h:y - h + char_array.shape[0], x]
						lmxv2 = np.array(local_max(array))
						lmnv2 = np.array(local_min(array))

						if (lmxv2.shape[0] == lmxv.shape[0] and lmnv2.shape[0] == lmnv.shape[0]) and (
							np.less_equal(np.abs(np.sum(np.subtract(lmxv2, lmxv))), lmxv.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnv2, lmnv))), lmnv.shape[0])):
							array = haystack_image[y - min_dim:y - min_dim + char_array.shape[0], x - min_dim:x - min_dim + char_array.shape[1]][di]
							lmxd2 = np.array(local_max(array))
							lmnd2 = np.array(local_min(array))

							if (lmxd2.shape[0] == lmxd.shape[0] and lmnd2.shape[0] == lmnd.shape[0]) and (
								np.less_equal(np.abs(np.sum(np.subtract(lmxd2, lmxd))), lmxd.shape[0]) and np.less_equal(np.abs(np.sum(np.subtract(lmnd2, lmnd))), lmnd.shape[0])):
								array = haystack_image[y - h:y - h + char_array.shape[0], x - w:x - w + char_array.shape[1]]
								if np.less(np.mean(np.abs(np.subtract(array, char_array))), threshold):
									master_vals[char].append((x, y))
	return master_vals
	master_vals2 = {}
	for i in range(len(master_vals.keys()) - 1):
		k1 = list(master_vals.keys())[i]
		k2 = list(master_vals.keys())[i + 1]
		for x1, y1 in master_vals[k1]:
			char_array1 = CHARACTER_ARRAYS[k1]
			w1 = np.floor_divide(char_array1.shape[1], 2)
			h1 = np.floor_divide(char_array1.shape[0], 2)
			for x2, y2 in sorted(master_vals[k2]):
				char_array2 = CHARACTER_ARRAYS[k2]
				w2 = np.floor_divide(char_array2.shape[1], 2)
				h2 = np.floor_divide(char_array2.shape[0], 2)
				x_dist = np.subtract(x2 - w2, x1 - w1 + char_array1.shape[1])
				y_dist = np.subtract(y2, y1)
				print(x1, y1)
				print(x2, y2)
				print(x_dist, y_dist, np.hypot(x_dist, y_dist))
				print()
				if np.hypot(x_dist, y_dist) <= max_distance:
					pass

	return master_vals
