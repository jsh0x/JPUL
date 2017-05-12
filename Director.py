__author__ = 'jsh0x'
__version__ = '1.0.0'

import datetime
from threading import Thread
import numpy as np
import Worker as worker, SyteLine as sl, SQL as sql

number_of_workers = 4
current_workers = np.empty((number_of_workers,), dtype=object)
current_types = np.empty_like(current_workers, dtype=np.str_)
current_statuses = np.empty_like(current_workers, dtype=np.str_)
current_processes = np.empty_like(current_workers, dtype=np.str_)
current_units = np.empty_like(current_workers, dtype=object)
current_queue = np.empty_like(current_workers, dtype=np.ndarray)


def display():
	global current_workers, current_types, current_statuses, current_processes, current_units, current_queue
	management_array = np.vstack((current_workers, current_types, current_statuses, current_processes, current_units, current_queue))
	col_wrk = " Worker "
	col_typ = " Type "
	col_sta = " Status "
	col_prc = " Process "
	col_unt = " Unit "
	col_que = " In-Queue "
	col_headers = np.array([col_wrk, col_typ, col_sta, col_prc, col_unt, col_que], dtype=np.str_)
	max_w = np.zeros_like(col_headers, dtype=int)
	for i,row in enumerate(management_array):
		max_w[i] = np.maximum(len(str(col_headers[i])), max_w[i])
		for col in row:
			max_w[i] = np.maximum(len(str(col))+2, max_w[i])

	total_w = col_headers.shape[0]
	for col in col_headers: total_w += len(col)
	underline = ("-"*total_w)
	row_template = ("|{}"*col_headers.shape[0])+"|"

	print(row_template.format(*col_headers))
	print(underline)

	for row in management_array.T:
		slot_used = True
		val_list = []
		for i,(header,col) in enumerate(zip(col_headers, row)):
			if slot_used and not col: slot_used = False
			w = len(str(header))
			if not slot_used: col = "-"*(w-2)
			val_list.append(str(col).center(w))
		print(row_template.format(*val_list))
	print(underline)

def get_time() -> datetime.datetime:
	return datetime.datetime.today()

def update_queue():
	now = get_time()
	priority = np.zeros((2,5,3), dtype=np.int32)
	#TODO: Add condition for users on weekends and night-shift
	statuses = ('Queued', 'Skipped')
	suffixes = ('Monitoring', 'RTS', 'Direct', 'Demo', 'Refurbished')
	speeds = ('Fast', 'Medium', 'Slow')
	for i,status in enumerate(statuses):
		for j,sfx in enumerate(suffixes):
			for k,speed in enumerate(speeds):
				priority[i,j,k] = len(sql.query(f"SELECT [Serial Number],[Build],[Operation] FROM [PyComm] WHERE [Status] = '{status}' AND [Suffix] = '{sfx}' AND [Speed] = '{speed}' ORDER BY [DateTime] DESC"))
	skipped = priority[1]
	queued = priority[0]
	monitoring_queued = queued[0]
	directs_queued = queued[1:3]
	demos_queued = queued[3]
	refurbished_queued = queued[4]
	fast_queued = queued[:,0]
	medium_queued = queued[:,1]
	slow_queued = queued[:,2]

	if number_of_workers == 1:
		current_types[0] = "QuickQuery"
		if now.weekday < 5:  # If it's a weekday
			if now.hour < 6:  # If it's early-morning
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
				elif np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[0] = "Troubleshoot"
			elif 17 < now.hour:  # If it's night
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
				elif np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[0] = "Troubleshoot"
		elif now.weekday >= 5:  # If it's the weekend
			if np.greater(queued.sum, 0):  # If there are any units in the queue
				current_types[0] = "Paperwork"
				if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
					current_queue[0] = directs_queued
				elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
					current_queue[0] = demos_queued
				elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
					current_queue[0] = refurbished_queued
				else:  # If there are any monitoring units in the queue
					current_queue[0] = monitoring_queued
			elif np.greater(skipped.sum, 0):  # If there are any units that have been skipped
				current_types[0] = "Troubleshoot"
	elif number_of_workers == 2:
		current_types[0] = "QuickQuery"
		current_types[1] = "QuickQuery"
		if now.weekday < 5:  # If it's a weekday
			if now.hour < 6:  # If it's early-morning
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
				if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[0] = "Troubleshoot"
			elif 5 < now.hour < 18:  # If it's the middle of the day
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
				if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[0] = "Troubleshoot"
			elif 17 < now.hour:  # If it's night
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
					if np.greater(queued.sum, 1):
						current_types[1] = "Paperwork"
				if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[0] = "Troubleshoot"
		elif now.weekday >= 5:  # If it's the weekend
			if np.greater(queued.sum, 0):  # If there are any units in the queue
				current_types[0] = "Paperwork"
				if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
					current_queue[0] = directs_queued
				elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
					current_queue[0] = demos_queued
				elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
					current_queue[0] = refurbished_queued
				else:  # If there are any monitoring units in the queue
					current_queue[0] = monitoring_queued
				if np.greater(queued.sum, 1):
					current_types[1] = "Paperwork"
					if np.greater(directs_queued.sum, 1):  # If there are any direct/RTS units in the queue
						current_queue[1] = directs_queued
					elif np.greater(demos_queued.sum, 1):  # If there are any demo units in the queue
						current_queue[1] = demos_queued
					elif np.greater(refurbished_queued.sum, 1):  # If there are any refurbished units in the queue
						current_queue[1] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[1] = monitoring_queued
			if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
				current_types[1] = "Troubleshoot"
	elif number_of_workers == 3:
		current_types[0] = "QuickQuery"
		current_types[1] = "QuickQuery"
		current_types[2] = "QuickQuery"
		if now.weekday < 5:  # If it's a weekday
			if now.hour < 6:  # If it's early-morning
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
					if np.greater(queued.sum, 1):
						current_types[1] = "Paperwork"
						if np.greater(directs_queued.sum, 1):  # If there are any direct/RTS units in the queue
							current_queue[1] = directs_queued
						elif np.greater(demos_queued.sum, 1):  # If there are any demo units in the queue
							current_queue[1] = demos_queued
						elif np.greater(refurbished_queued.sum, 1):  # If there are any refurbished units in the queue
							current_queue[1] = refurbished_queued
						else:  # If there are any monitoring units in the queue
							current_queue[1] = monitoring_queued
				if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[1] = "Troubleshoot"
			elif 5 < now.hour < 18:  # If it's the middle of the day
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
				elif np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[0] = "Troubleshoot"
			elif 17 < now.hour:  # If it's night
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
					if np.greater(queued.sum, 1):
						current_types[1] = "Paperwork"
						if np.greater(directs_queued.sum, 1):  # If there are any direct/RTS units in the queue
							current_queue[1] = directs_queued
						elif np.greater(demos_queued.sum, 1):  # If there are any demo units in the queue
							current_queue[1] = demos_queued
						elif np.greater(refurbished_queued.sum, 1):  # If there are any refurbished units in the queue
							current_queue[1] = refurbished_queued
						else:  # If there are any monitoring units in the queue
							current_queue[1] = monitoring_queued
				if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[1] = "Troubleshoot"
		elif now.weekday >= 5:  # If it's the weekend
			if np.greater(queued.sum, 0):  # If there are any units in the queue
				current_types[0] = "Paperwork"
				if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
					current_queue[0] = directs_queued
				elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
					current_queue[0] = demos_queued
				elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
					current_queue[0] = refurbished_queued
				else:  # If there are any monitoring units in the queue
					current_queue[0] = monitoring_queued
				if np.greater(queued.sum, 1):
					current_types[1] = "Paperwork"
					if np.greater(directs_queued.sum, 1):  # If there are any direct/RTS units in the queue
						current_queue[1] = directs_queued
					elif np.greater(demos_queued.sum, 1):  # If there are any demo units in the queue
						current_queue[1] = demos_queued
					elif np.greater(refurbished_queued.sum, 1):  # If there are any refurbished units in the queue
						current_queue[1] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[1] = monitoring_queued
			if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
				current_types[1] = "Troubleshoot"
	elif number_of_workers == 4:  # If there are 4 virtual machines available
		current_types[0] = "QuickQuery"
		current_types[1] = "QuickQuery"
		current_types[2] = "QuickQuery"
		current_types[3] = "QuickQuery"
		if now.weekday < 5:  # If it's a weekday
			if now.hour < 6:  # If it's early-morning
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
					if np.greater(queued.sum, 1):
						current_types[1] = "Paperwork"
						if np.greater(directs_queued.sum, 1):
							current_queue[1] = directs_queued
						elif np.greater(demos_queued.sum, 1):
							current_queue[1] = demos_queued
						elif np.greater(refurbished_queued.sum, 1):
							current_queue[1] = refurbished_queued
						else:
							current_queue[1] = monitoring_queued
						if np.greater(queued.sum, 2):
							current_types[2] = "Paperwork"
							if np.greater(directs_queued.sum, 2):  # If there are any direct/RTS units in the queue
								current_queue[2] = directs_queued
							elif np.greater(demos_queued.sum, 2):  # If there are any demo units in the queue
								current_queue[2] = demos_queued
							elif np.greater(refurbished_queued.sum, 2):  # If there are any refurbished units in the queue
								current_queue[2] = refurbished_queued
							else:  # If there are any monitoring units in the queue
								current_queue[2] = monitoring_queued
				if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[2] = "Troubleshoot"
			elif 5 < now.hour < 18:  # If it's the middle of the day
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
					if np.greater(queued.sum, 1):
						current_types[1] = "Paperwork"
						if np.greater(directs_queued.sum, 1):  # If there are any direct/RTS units in the queue
							current_queue[1] = directs_queued
						elif np.greater(demos_queued.sum, 1):  # If there are any demo units in the queue
							current_queue[1] = demos_queued
						elif np.greater(refurbished_queued.sum, 1):  # If there are any refurbished units in the queue
							current_queue[1] = refurbished_queued
						else:  # If there are any monitoring units in the queue
							current_queue[1] = monitoring_queued
			elif 17 < now.hour:  # If it's night
				if np.greater(queued.sum, 0):  # If there are any units in the queue
					current_types[0] = "Paperwork"
					if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
						current_queue[0] = directs_queued
					elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
						current_queue[0] = demos_queued
					elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
						current_queue[0] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[0] = monitoring_queued
					if np.greater(queued.sum, 1):
						current_types[1] = "Paperwork"
						if np.greater(directs_queued.sum, 1):  # If there are any direct/RTS units in the queue
							current_queue[1] = directs_queued
						elif np.greater(demos_queued.sum, 1):  # If there are any demo units in the queue
							current_queue[1] = demos_queued
						elif np.greater(refurbished_queued.sum, 1):  # If there are any refurbished units in the queue
							current_queue[1] = refurbished_queued
						else:  # If there are any monitoring units in the queue
							current_queue[1] = monitoring_queued
				if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
					current_types[2] = "Troubleshoot"
					if np.greater(skipped.sum, 1):
						current_types[3] = "Troubleshoot"
		elif now.weekday >= 5:  # If it's the weekend
			if np.greater(queued.sum, 0):  # If there are any units in the queue
				current_types[0] = "Paperwork"
				if np.greater(directs_queued.sum, 0):  # If there are any direct/RTS units in the queue
					current_queue[0] = directs_queued
				elif np.greater(demos_queued.sum, 0):  # If there are any demo units in the queue
					current_queue[0] = demos_queued
				elif np.greater(refurbished_queued.sum, 0):  # If there are any refurbished units in the queue
					current_queue[0] = refurbished_queued
				else:  # If there are any monitoring units in the queue
					current_queue[0] = monitoring_queued
				if np.greater(queued.sum, 1):
					current_types[1] = "Paperwork"
					if np.greater(directs_queued.sum, 1):  # If there are any direct/RTS units in the queue
						current_queue[1] = directs_queued
					elif np.greater(demos_queued.sum, 1):  # If there are any demo units in the queue
						current_queue[1] = demos_queued
					elif np.greater(refurbished_queued.sum, 1):  # If there are any refurbished units in the queue
						current_queue[1] = refurbished_queued
					else:  # If there are any monitoring units in the queue
						current_queue[1] = monitoring_queued
			if np.greater(skipped.sum, 0):  # If there are any units that have been skipped
				current_types[2] = "Troubleshoot"
				if np.greater(skipped.sum, 1):
					current_types[3] = "Troubleshoot"