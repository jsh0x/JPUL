__author__ = 'jsh0x'
__version__ = '1.0.0'

import datetime
from threading import Thread
import numpy as np
#from . import Worker, SyteLine, SQL

number_of_workers = 4
current_workers = np.empty((number_of_workers,), dtype=object)
current_types = np.empty_like(current_workers, dtype=np.str_)
current_statuses = np.empty_like(current_workers, dtype=np.str_)
current_processes = np.empty_like(current_workers, dtype=np.str_)
current_units = np.empty_like(current_workers, dtype=object)
current_queue = np.empty_like(current_workers, dtype=object)
management_array = np.vstack((current_workers, current_types, current_statuses,current_processes,current_units, current_queue))

def display():
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

def update_queue():
	#SQL query for number of skipped
	skipped = 1
	#SQL query for number of monitoring in-queue
	monitoring = 1
	#SQL query for number of directs in-queue
	directs = 1
	#SQL query for number of RTS in-queue
	rts = 1
	#SQL query for number of demos in-queue
	demo = 1

#worker, status, process, unit = management_array

display()