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

def look_test(char: np.ndarray):
	char = np.asarray(get_gradient(char), dtype=np.int32)
	retval = []
	img2 = np.array(Image.open('Untitled2.png').convert('L'))
	img = np.asarray(get_gradient(img2), dtype=np.int32)
	for y in np.arange(img.shape[0]-char.shape[0]):
		for x in np.arange(img.shape[1]-char.shape[1]):
			k = np.asarray(img[y:y+char.shape[0], x:x+char.shape[1]].copy(), dtype=np.int32)
			val = np.mean(np.abs(np.subtract(k.copy(),char.copy())))
			if val < 20:
				print(x, y, val)
				retval.append(img2[y:y+char.shape[0]+1, x:x+char.shape[1]+1])
	return retval


def markup_plot(ax: plt.axis, line_num: int=0, dashed=False, mark_local=False, mark_global=True):
	line = ax.lines[line_num]
	x_data,y_data = line.get_xdata(),line.get_ydata()
	lmn = local_min(y_data)
	lmx = local_max(y_data)
	ymin, ymax = plt.ylim()
	center = np.floor_divide(diff(ymax, ymin), 2)

	if mark_global:
		if dashed:
			style = 'dashdot'
		else:
			style = 'dotted'
		ax.hlines(y=max(y_data), xmin=x_data[0], xmax=x_data[-1], linestyles=style, label=str(max(y_data)), alpha=0.3)
		ax.hlines(y=min(y_data), xmin=x_data[0], xmax=x_data[-1], linestyles=style, label=str(min(y_data)), alpha=0.3)

	if mark_local:
		lmni = [x_data[i] for i in np.arange(lmn[0].shape[0]) if (lmn[0][i] == 1)]
		lmxi = [x_data[i] for i in np.arange(lmx[0].shape[0]) if (lmx[0][i] == 1)]

		for i, mx in zip(lmxi, lmx[1]):
			ax.plot(i, mx, '^k', alpha=0.6)
			if mx > center:
				ax.text(i, mx-40, str(mx), ha='center', va='center')
			else:
				ax.text(i, mx+40, str(mx), ha='center', va='center')
		for i, mn in zip(lmni, lmn[1]):
			ax.plot(i, mn, 'vk', alpha=0.6)
			if mn < center:
				ax.text(i, mn+40, str(mn), ha='center', va='center')
			else:
				ax.text(i, mn-40, str(mn), ha='center', va='center')

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

def compare_levels(a1: np.ndarray, a2: Tuple[np.ndarray], title: str, one_window=True):
	columns = len(a2)
	if one_window:
		f, ax_t = plt.subplots(4, columns, sharex=True, sharey=True)
		for i in np.arange(columns):
			(ax1, ax2, ax3, ax4) = ax_t[:, i]
			ax1.set_title(title + str(i + 1))

			cx = np.floor_divide(a2[i].shape[1], 2)
			cy = np.floor_divide(a2[i].shape[0], 2)
			min_dim = min(a2[i].shape[0], a2[i].shape[1])
			di = np.diag_indices(min_dim)
			b1 = np.fliplr(a1.copy())
			b2 = np.fliplr(a2[i].copy())

			# Horizontal
			ax1.plot(np.arange(a1.shape[1]), a1[cy, :], '.r-')
			temp1 = get_gradient(a1[cy, :])
			ax1.plot(np.arange(1, a1.shape[1]), temp1, '.r--')
			ax1.plot(np.arange(a2[i].shape[1]), a2[i][cy, :], '.r:', alpha=0.3)
			temp2 = get_gradient(a2[i][cy, :])
			ax1.plot(np.arange(1, a2[i].shape[1]), temp2, '.r-.', alpha=0.3)
			ax1.vlines(np.arange(a1.shape[1]), a1[cy, :], a2[i][cy, :], linestyles='dashed')
			ax1.vlines(np.arange(1, a1.shape[1]), temp1, temp2, linestyles='dotted')

			# Vertical
			ax2.plot(np.arange(a1.shape[0]), a1[:, cx], '.g-')
			temp1 = get_gradient(a1[:, cx])
			ax2.plot(np.arange(1, a1.shape[0]), temp1, '.g--')
			ax2.plot(np.arange(a2[i].shape[0]), a2[i][:, cx], '.g:', alpha=0.3)
			temp2 = get_gradient(a2[i][:, cx])
			ax2.plot(np.arange(1, a2[i].shape[0]), temp2, '.g-.', alpha=0.3)
			ax2.vlines(np.arange(a1.shape[0]), a1[:, cx], a2[i][:, cx], linestyles='dashed')
			ax2.vlines(np.arange(1, a1.shape[0]), temp1, temp2, linestyles='dotted')

			# Negative Diagonal
			ax3.plot(np.arange(min_dim), a1[di], '.b-')
			temp1 = get_gradient(a1[di])
			ax3.plot(np.arange(1, min_dim), temp1, '.b--')
			ax3.plot(np.arange(min_dim), a2[i][di], '.b:', alpha=0.3)
			temp2 = get_gradient(a2[i][di])
			ax3.plot(np.arange(1, min_dim), temp2, '.b-.', alpha=0.3)
			ax3.vlines(np.arange(min_dim), a1[di], a2[i][di], linestyles='dashed')
			ax3.vlines(np.arange(1, min_dim), temp1, temp2, linestyles='dotted')

			# Positive Diagonal
			ax4.plot(np.arange(min_dim), b1[di], '.c-')
			temp1 = get_gradient(b1[di])
			ax4.plot(np.arange(1, min_dim), temp1, '.c--')
			ax4.plot(np.arange(min_dim), b2[di], '.c:', alpha=0.3)
			temp2 = get_gradient(b2[di])
			ax4.plot(np.arange(1, min_dim), temp2, '.c-.', alpha=0.3)
			ax4.vlines(np.arange(min_dim), b1[di], b2[di], linestyles='dashed')
			ax4.vlines(np.arange(1, min_dim), temp1, temp2, linestyles='dotted')

		f.subplots_adjust(hspace=0)
		# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
		plt.show()
	else:
		for c in np.arange(4):
			f, ax_t = plt.subplots(columns, sharex=True, sharey=True)
			for i,ax in enumerate(ax_t):
				#ax.set_title(title + str(i + 1))
				cx = np.floor_divide(a2[i].shape[1], 2)
				cy = np.floor_divide(a2[i].shape[0], 2)
				min_dim = min(a2[i].shape[0], a2[i].shape[1])
				di = np.diag_indices(min_dim)
				b1 = np.fliplr(a1.copy())
				b2 = np.fliplr(a2[i].copy())

				if c == 0:
					ax.plot(np.arange(a1.shape[1]), a1[cy, :], '.r-')
					temp1 = get_gradient(a1[cy, :])
					ax.plot(np.arange(1, a1.shape[1]), temp1, '.r--')
					ax.plot(np.arange(a2[i].shape[1]), a2[i][cy, :], '.b:')
					temp2 = get_gradient(a2[i][cy, :])
					ax.plot(np.arange(1, a2[i].shape[1]), temp2, '.b-.')
					ax.vlines(np.arange(a1.shape[1]), a1[cy, :], a2[i][cy, :], linestyles='dashed')
					ax.vlines(np.arange(1, a1.shape[1]), temp1, temp2, linestyles='dotted')

				elif c == 1:
					ax.plot(np.arange(a1.shape[0]), a1[:, cx], '.g-')
					temp1 = get_gradient(a1[:, cx])
					ax.plot(np.arange(1, a1.shape[0]), temp1, '.g--')
					ax.plot(np.arange(a2[i].shape[0]), a2[i][:, cx], '.r:')
					temp2 = get_gradient(a2[i][:, cx])
					ax.plot(np.arange(1, a2[i].shape[0]), temp2, '.r-.')
					ax.vlines(np.arange(a1.shape[0]), a1[:, cx], a2[i][:, cx], linestyles='dashed')
					ax.vlines(np.arange(1, a1.shape[0]), temp1, temp2, linestyles='dotted')

				elif c == 2:
					ax.plot(np.arange(min_dim), a1[di], '.b-')
					temp1 = get_gradient(a1[di])
					ax.plot(np.arange(1, min_dim), temp1, '.b--')
					ax.plot(np.arange(min_dim), a2[i][di], '.g:')
					temp2 = get_gradient(a2[i][di])
					ax.plot(np.arange(1, min_dim), temp2, '.g-.')
					ax.vlines(np.arange(min_dim), a1[di], a2[i][di], linestyles='dashed')
					ax.vlines(np.arange(1, min_dim), temp1, temp2, linestyles='dotted')

				elif c == 3:
					ax.plot(np.arange(min_dim), b1[di], '.c-')
					temp1 = get_gradient(b1[di])
					ax.plot(np.arange(1, min_dim), temp1, '.c--')
					ax.plot(np.arange(min_dim), b2[di], '.m:')
					temp2 = get_gradient(b2[di])
					ax.plot(np.arange(1, min_dim), temp2, '.m-.')
					ax.vlines(np.arange(min_dim), b1[di], b2[di], linestyles='dashed')
					ax.vlines(np.arange(1, min_dim), temp1, temp2, linestyles='dotted')

			f.subplots_adjust(hspace=0)
			# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
			plt.show()

def display_levels(a: Union[np.ndarray, Tuple[np.ndarray]], title: str):
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
	if type(a) is Tuple:
		columns = len(a)
	else:
		columns = 1
	columns = len(a)
	f, ax_t = plt.subplots(4, columns, sharex=True, sharey=True)
	for i in np.arange(columns):
		(ax1, ax2, ax3, ax4) = ax_t[:,i]
		ax1.set_title(title+str(i+1))

		cx = np.floor_divide(a[i].shape[1], 2)
		cy = np.floor_divide(a[i].shape[0], 2)
		min_dim = min(a[i].shape[0], a[i].shape[1])
		di = np.diag_indices(min_dim)
		b = np.fliplr(a[i].copy())

		ax1.plot(np.arange(a[i].shape[1]), a[i][cy,:], '.r-')
		markup_plot(ax1)
		temp = get_gradient(a[i][cy, :])
		ax1.plot(np.arange(1, a[i].shape[1]), temp, '.r--')

		ax2.plot(np.arange(a[i].shape[0]), a[i][:,cx], '.g-')
		markup_plot(ax2)
		temp = get_gradient(a[i][:,cx])
		ax2.plot(np.arange(1, a[i].shape[0]), temp, '.g--')

		ax3.plot(np.arange(min_dim), a[i][di], '.b-')
		markup_plot(ax3)
		temp = get_gradient(a[i][di])
		ax3.plot(np.arange(1, min_dim), temp, '.b--')

		ax4.plot(np.arange(min_dim), b[di], '.c-')
		markup_plot(ax4)
		temp = get_gradient(b[di])
		ax4.plot(np.arange(1, min_dim), temp, '.c--')

	f.subplots_adjust(hspace=0)
	#plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	plt.show()

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
imgs = look_test(G1im)
#display_levels((G1im,G2im,G3im,G4im,G5im), 'G')
compare_levels(G1im, imgs, 'G')
#TODO: Label all points?


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
