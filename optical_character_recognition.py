__author__ = 'jsh0x'
__version__ = '3.0.0'

import sqlite3 as sql
from typing import Union, Dict, Tuple
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from zmath import get_local_max as local_max, get_local_min as local_min, diff
from image import get_gradient, get_diagonals, show_image
from constants import CHARACTERS, STANDARD_COLORS


#Global Variables
DIRECTIONS = ('horizontal', 'vertical', 'diagonal_negative', 'diagonal_positive')
"""conn = sql.connect('character_info.db')
c = conn.cursor()
c.execute('DROP TABLE character_arrays')
c.execute('DROP TABLE character_shapes')
c.execute('CREATE TABLE character_arrays (char, image_array, direction, local_max, local_min, local_max2, local_min2, global_max, global_min)')
c.execute('CREATE TABLE character_shapes (char, shape)')
conn.commit()
conn.close()"""



def markup_plot(ax: plt.axis()):
	line = ax.lines[0]
	x_data,y_data = line.get_xdata(),line.get_ydata()
	lmn = local_min(y_data)
	lmx = local_max(y_data)
	lmni = [x_data[i] for i in np.arange(lmn[0].shape[0]) if lmn[0][i] == 1]
	lmxi = [x_data[i] for i in np.arange(lmx[0].shape[0]) if lmx[0][i] == 1]
	ymin, ymax = plt.ylim()
	for i, mx in zip(lmxi, lmx[1]):
		if np.asarray(mx, dtype=np.int32)+25 >= ymax:
			ax.plot(i, mx-25, '^k')
		else:
			ax.plot(i, mx+25, '^k')
	for i, mn in zip(lmni, lmn[1]):
		if np.asarray(mn, dtype=np.int32)-25 <= ymin:
			ax.plot(i, mn+25, 'vk')
		else:
			ax.plot(i, mn-25, 'vk')


def remember(char: str=None) -> dict:
	if not char:
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
			local_min_array = np.zeros_like(sample_list[0][0], dtype=np.int32)
			local_max_array = np.zeros_like(sample_list[0][0], dtype=np.int32)
			for i,(array,lmx,lmn) in enumerate(sample_list):
				#full_array += np.array(sample)
				pass
		#return retval


def teach(char: str, image: np.ndarray):
	temp = {}
	if type(char) is not str: raise TypeError
	if len(char) > 1: raise ValueError
	conn = sql.connect('character_info.db')
	c = conn.cursor()
	c.execute(f"SELECT shape FROM character_shapes WHERE char = '{char}'")
	#shape = tuple(c.fetchone())
	shape = (9,9)
	conn.close()
	if image.shape != shape: raise IndexError(f'{image.shape} != {shape}')
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
		elif direction == 'diagonal_negative':
			lmx,lmn, lmx2, lmn2 = get_diagonals(image)
		elif direction == 'diagonal_positive':
			lmx, lmn, lmx2, lmn2 = get_diagonals(image, flipped=True)

		conn = sql.connect('character_info.db')
		c = conn.cursor()
		#c.execute(f"INSERT INTO character_arrays VALUES ('{char}', {image}, '{direction}', {lmx}, {lmn}, {lmx2}, {lmn2}, {gmx}, {gmn})")
		#conn.commit()
		conn.close()
		temp[direction] = (lmx, lmn, lmx2, lmn2)
	return temp
#char, image_array, direction, local_max, local_min, local_max2, local_min2, global_max, global_min
def display_peaks_and_valleys(a: np.ndarray):
	"""disp_val = np.empty((list(a_dict.values())[0][0].shape[0], list(a_dict.values())[0][0].shape[1], 3), dtype=np.int16)
	disp_val[...,0] = ao
	disp_val[...,1] = ao
	disp_val[...,2] = ao
	for k,v in a_dict.items():
		v2, v3, v4, v5 = v
		temp = [(x,y) for (x,y),val in np.ndenumerate(v2) if val==1]
		for x,y in temp:
			if 'horizontal' in k:
				disp_val[y,x] = (255, 0, 0)
				#disp_val[...,0] = np.where(v3 == 1, 64, disp_val[...,0])
			elif 'vertical' in k:
				disp_val[y, x] = (0, 255, 0)
				#disp_val[...,1] = np.where(v3 == 1, 64, disp_val[...,1])
			elif 'diagonal_negative' in k:
				disp_val[y, x] = (0, 0, 255)
				#disp_val[...,2] = np.where(v3 == 1, 64, disp_val[...,2])
			elif 'diagonal_positive' in k:
				disp_val[y, x] = (255, 255, 0)
	im = Image.fromarray(np.asarray(disp_val[..., ::-1], dtype=np.uint8)).convert('RGB')
	a = np.array(im)
	for y in np.arange(a.shape[0]):
		for x in np.arange(a.shape[1]):
			direction = None
			if a[y,x,0] == 255 and a[y,x,1] == 0 and a[y,x,2] == 0:
				line = a[y,:].view()
				direction = 'horizontal'
			elif a[y,x,0] == 0 and a[y,x,1] == 255 and a[y,x,2] == 0:
				line = a[:,x].view()
				direction = 'vertical'
			elif a[y,x,0] == 0 and a[y,x,1] == 0 and a[y,x,2] == 255:
				temp_x = x
				temp_y = y
				while temp_x > 0 and temp_y > 0:
					temp_x -= 1
					temp_y -= 1
				min_x = temp_x
				min_y = temp_y

				temp_x = x
				temp_y = y
				while temp_x < a.shape[1] and temp_y < a.shape[0]:
					temp_x += 1
					temp_y += 1
				max_x = temp_x
				max_y = temp_y
				x_range = np.arange(min_x, max_x)
				y_range = np.arange(min_y, max_y)
				xy = (y_range, x_range)
				line = a[xy].view()
				direction = 'diagonal_negative'
			elif a[y,x,0] == 255 and a[y,x,1] == 255 and a[y,x,2] == 0:
				temp_x = x
				temp_y = y
				while temp_x > 0 and temp_y < a.shape[0]:
					temp_x -= 1
					temp_y -= 1
				min_x = temp_x
				max_y = temp_y

				temp_x = x
				temp_y = y
				while temp_x < a.shape[1] and temp_y > 0:
					temp_x += 1
					temp_y += 1
				max_x = temp_x
				min_y = temp_y
				x_range = np.arange(min_x, max_x)
				y_range = np.arange(min_y, max_y)
				xy = (y_range, x_range)
				line = a[xy].view()
				direction = 'diagonal_positive'
			else:
				continue
			for i in np.arange(line.shape[0]):
				if direction == 'horizontal':
					if line[i,0] < 200:
						line[i,0] = line[i,0]+32
					if line[i,1] >= 32:
						line[i,1] = line[i,1]-32
					if line[i,2] >= 32:
						line[i,2] = line[i,2]-32
				elif direction == 'vertical':
					if line[i,0] >= 32:
						line[i,0] = line[i,0]-32
					if line[i,1] < 200:
						line[i,1] = line[i,1]+32
					if line[i,2] >= 32:
						line[i,2] = line[i,2]-32
				elif direction == 'diagonal_negative':
					if line[i,0] >= 32:
						line[i,0] = line[i,0]-32
					if line[i,1] >= 32:
						line[i,1] = line[i,1]-32
					if line[i,2] < 200:
						line[i,2] = line[i,2]+32
				elif direction == 'diagonal_positive':
					if line[i,0] < 200:
						line[i,0] = line[i,0]+32
					if line[i,1] < 200:
						line[i,1] = line[i,1]+32
					if line[i,2] >= 32:
						line[i,2] = line[i,2]-32
	im = Image.fromarray(a).convert('RGB')
	plt.imshow(im)
	plt.show()"""

	f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
	ax1.set_title('Sharing both axes')
	# Fine-tune figure; make subplots close to each other and hide x ticks for
	# all but bottom plot.


	cx = np.floor_divide(a.shape[1], 2)
	cy = np.floor_divide(a.shape[0], 2)
	min_dim = min(a.shape[0], a.shape[1])
	di = np.diag_indices(min_dim)
	b = np.fliplr(a.copy())

	ax1.plot(np.arange(a.shape[1]), a[cy,:], 'r-')
	#markup_plot(ax1)
	lmn = local_min(a[cy,:])
	lmx = local_max(a[cy,:])
	lmni = [np.arange(a.shape[1])[i] for i in np.arange(lmn[0].shape[0]) if lmn[0][i] == 1]
	lmxi = [np.arange(a.shape[1])[i] for i in np.arange(lmx[0].shape[0]) if lmx[0][i] == 1]
	ymin, ymax = plt.ylim()
	for i,mx in zip(lmxi,lmx[1]):
		if np.asarray(mx, dtype=np.int32)+25 >= ymax:
			ax1.plot(i, mx-25, '^k')
		else:
			ax1.plot(i, mx+25, '^k')
	for i,mn in zip(lmni,lmn[1]):
		if np.asarray(mn, dtype=np.int32)-25 <= ymin:
			ax1.plot(i, mn+25, 'vk')
		else:
			ax1.plot(i, mn-25, 'vk')

	ax2.plot(np.arange(a.shape[0]), a[:,cx], 'g-')
	lmn = local_min(a[:,cx])
	lmx = local_max(a[:,cx])
	lmni = [np.arange(a.shape[0])[i] for i in np.arange(lmn[0].shape[0]) if lmn[0][i] == 1]
	lmxi = [np.arange(a.shape[0])[i] for i in np.arange(lmx[0].shape[0]) if lmx[0][i] == 1]
	ymin, ymax = plt.ylim()
	for i,mx in zip(lmxi,lmx[1]):
		if np.asarray(mx, dtype=np.int32)+25 >= ymax:
			ax2.plot(i, mx-25, '^k')
		else:
			ax2.plot(i, mx+25, '^k')
	for i,mn in zip(lmni,lmn[1]):
		if np.asarray(mn, dtype=np.int32)-25 <= ymin:
			ax2.plot(i, mn+25, 'vk')
		else:
			ax2.plot(i, mn-25, 'vk')

	ax3.plot(np.arange(min_dim), a[di], 'b-')
	lmn = local_min(a[di])
	lmx = local_max(a[di])
	lmni = [np.arange(min_dim)[i] for i in np.arange(lmn[0].shape[0]) if lmn[0][i] == 1]
	lmxi = [np.arange(min_dim)[i] for i in np.arange(lmx[0].shape[0]) if lmx[0][i] == 1]
	ymin, ymax = plt.ylim()
	for i,mx in zip(lmxi,lmx[1]):
		if np.asarray(mx, dtype=np.int32) + 25 >= ymax:
			ax3.plot(i, mx-25, '^k')
		else:
			ax3.plot(i, mx+25, '^k')
	for i,mn in zip(lmni,lmn[1]):
		if np.asarray(mn, dtype=np.int32) - 25 <= ymin:
			ax3.plot(i, mn+25, 'vk')
		else:
			ax3.plot(i, mn-25, 'vk')

	ax4.plot(np.arange(min_dim), b[di], 'c-')
	lmn = local_min(b[di])
	lmx = local_max(b[di])
	lmni = [np.arange(min_dim)[i] for i in np.arange(lmn[0].shape[0]) if lmn[0][i] == 1]
	lmxi = [np.arange(min_dim)[i] for i in np.arange(lmx[0].shape[0]) if lmx[0][i] == 1]
	ymin, ymax = plt.ylim()
	for i,mx in zip(lmxi,lmx[1]):
		if np.asarray(mx, dtype=np.int32) + 25 >= ymax:
			ax4.plot(i, mx-25, '^k')
		else:
			ax4.plot(i, mx+25, '^k')
	for i,mn in zip(lmni,lmn[1]):
		if np.asarray(mn, dtype=np.int32) - 25 <= ymin:
			ax4.plot(i, mn+25, 'vk')
		else:
			ax4.plot(i, mn-25, 'vk')


	temp = np.asarray(a[cy,:], dtype=np.int32)
	temp = np.abs(np.subtract(temp[:-1], temp[1:]))
	ax1.plot(np.arange(1, a.shape[1]), temp, 'r--')

	temp = np.asarray(a[:,cx], dtype=np.int32)
	temp = np.abs(np.subtract(temp[:-1], temp[1:]))
	ax2.plot(np.arange(1, a.shape[0]), temp, 'g--')

	temp = np.asarray(a[di], dtype=np.int32)
	temp = np.abs(np.subtract(temp[:-1], temp[1:]))
	ax3.plot(np.arange(1, min_dim), temp, 'b--')

	temp = np.asarray(b[di], dtype=np.int32)
	temp = np.abs(np.subtract(temp[:-1], temp[1:]))
	ax4.plot(np.arange(1, min_dim), temp, 'c--')
	f.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	#ax1.tick_params(axis='y', left='off', labelleft='off')
	#ax2.tick_params(axis='y', left='off', labelleft='off')
	#ax3.tick_params(axis='y', left='off', labelleft='off')
	#ax4.tick_params(axis='y', left='off', labelleft='off')
	#plt.minorticks_on()
	"""h_array = np.zeros_like(a[1:], dtype=np.int16)
	for i in np.arange(h_array.shape[0]):
		rows = np.asarray(a[i:i+2,:], dtype=np.int32)
		val = np.abs(np.subtract(rows[0],rows[1]))
		h_array[i] = val
	y_vals = (h_array.mean(axis=1)).round(1)
	x_vals = np.arange(h_array.shape[0])
	plt.plot(x_vals, y_vals, 'r-')"""

	"""v_array = np.zeros_like(a[:,1:], dtype=np.int16)
	for i in np.arange(v_array.shape[1]):
		cols = np.asarray(a[:,i:i+2], dtype=np.int32)
		val = np.abs(np.subtract(cols[:,0], cols[:,1]))
		v_array[:,i] = val
	y_vals = (v_array.mean(axis=1)).round(1)
	x_vals = np.arange(v_array.shape[0])
	plt.plot(x_vals, y_vals, 'g-')"""
	plt.show()
	#show_image(disp_val[...,::-1], gray=False)
#def test():
#	pass
G1im = np.array(Image.open('G1.png').convert('L'))
G2im = np.array(Image.open('G2.png').convert('L'))
G3im = np.array(Image.open('G3.png').convert('L'))
G4im = np.array(Image.open('G4.png').convert('L'))
G5im = np.array(Image.open('G5.png').convert('L'))
G1 = teach('G', G1im)
G2 = teach('G', G2im)
G3 = teach('G', G3im)
G4 = teach('G', G4im)
G5 = teach('G', G5im)
display_peaks_and_valleys(G1im)
display_peaks_and_valleys(G1im)
display_peaks_and_valleys(G1im)
display_peaks_and_valleys(G1im)
display_peaks_and_valleys(G1im)


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
