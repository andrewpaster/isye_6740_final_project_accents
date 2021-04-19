#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 20:38:11 2021

@author: Andrew
"""

import librosa
import os
import librosa.display
import numpy as np
import pandas as pd

import pathlib
import librosa
from multiprocessing import Pool

DIRNAME = pathlib.Path().absolute()
LOCALPATH = 'isye_6740_final_project_accents/test/'
PATHNAME = os.path.join(DIRNAME, LOCALPATH)

def analyze_file(file_name):

	try:

		file_path = os.path.join(PATHNAME, file_name)
		formants = pfp.formants_at_interval(file_path, winlen=0.001)
		median_row = int(len(formants) / 2)
		accent = file_name.split('_')[3]
		gender = file_name.split('_')[1]
		word = file_name.split('_')[0]
		speaker = file_name.split('_')[2].split('-')[0]
		window = len(formants)

		y, sr = librosa.load(file_path, sr=16000)
		duration = librosa.get_duration(y=y, sr=sr)

		meta_data = np.array([median_row, accent, gender, word, speaker, window, duration]).reshape(1,7)

		formants_avg = np.average(formants, axis=0).reshape(1,4)
		formants_median = formants[median_row].reshape(1,4)
		formants_stacked = np.hstack((formants_median, formants_avg)).reshape(1,8)
		formants_stacked = np.hstack((formants_stacked, meta_data)).reshape(1,15)
		
		return formants_stacked[0]

	except:

		print('could not use word {}'.format(file_name))
		
		output = [0]*15
		
		accent = file_name.split('_')[3]
		gender = file_name.split('_')[1]
		word = file_name.split('_')[0]
		speaker = file_name.split('_')[2].split('-')[0]

		output[9] = accent
		output[10] = gender
		output[11] = word
		output[12] = speaker

		return output


file_names = [x for x in os.listdir(PATHNAME)]
result = [None]*len(file_names)

pool = Pool(4)    
for i, temparray in enumerate(pool.imap(analyze_file, file_names)):
    result[i] = temparray
    if i % 500 == 0:
    	print(i, file_names[i])

pool.close()
pool.join()

columns = [['timestamp', 'f1', 'f2', 'f3', 'avg_timestamp', 'f1_avg', 'f2_avg', 'f3_avg', 'median_row', 'accent', 'gender', 'word', 'speaker', 'window', 'duration']]

df = pd.DataFrame(result, columns=columns)
df.to_csv('test_formants.csv', index=False)
