import praat_formants_python as pfp
import pathlib
import os 
import statistics
import pandas as pd
import numpy as np
import librosa
from multiprocessing import Pool
from itertools import product

#DIRNAME = pathlib.Path().absolute()
#LOCALPATH = 'isye_6740_final_project_accents/test/'
#PATHNAME = os.path.join(DIRNAME, LOCALPATH)

# uncomment to run on training set folder
# DIRNAME = pathlib.Path().absolute()
# LOCALPATH = 'isye_6740_final_project_accents/train/'
# PATHNAME = os.path.join(DIRNAME, LOCALPATH)


DIRNAME = pathlib.Path().absolute().parents[1]


def analyze_file(pathname, file_name):
	""" outputs forman information for a single recording

		inputs: file_name (str): name of file_name
		returns: output (list): list of formant values and file meta data
	"""
	try:

		# read in file and parse the meta data like speaker id, sex, accent
		file_path = os.path.join(pathname, file_name)
		formants = pfp.formants_at_interval(file_path, winlen=0.001)
		median_row = int(len(formants) / 2)
		accent = file_name.split('_')[3]
		gender = file_name.split('_')[1]
		word = file_name.split('_')[0]
		speaker = file_name.split('_')[2].split('-')[0]
		window = len(formants)

		# get the recording length
		y, sr = librosa.load(file_path, sr=16000)
		duration = librosa.get_duration(y=y, sr=sr)

		# store meta data
		meta_data = np.array([median_row, accent, gender, word, speaker, window, duration]).reshape(1,7)

		# get formant data from Praat
		formants_avg = np.average(formants, axis=0).reshape(1,4)
		formants_median = formants[median_row].reshape(1,4)
		formants_stacked = np.hstack((formants_median, formants_avg)).reshape(1,8)
		formants_stacked = np.hstack((formants_stacked, meta_data)).reshape(1,15)
		
		# return output as a list
		return formants_stacked[0]

	except:

		# when exception occurs, return list with 0 values and meta data
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


def get_file_formants(output_file='test_formants_eg.csv', localpath = 'isye_6740_final_project_accents/test/'):
    

    pathname = os.path.join(DIRNAME, localpath)

    # get all file names
    file_names = [x for x in os.listdir(pathname)]
    
    # make an empty list to hold results for each file
    result = [None]*len(file_names)
    
    # parallelize the file analysis
    pool = Pool(4)    
    for i, temparray in enumerate(pool.starmap(analyze_file, product(file_names, pathname))):
        result[i] = temparray
        if i % 500 == 0:
            	print(i, file_names[i])
    
    pool.close()
    pool.join()
    
    # output results to dataframe
    columns = [['timestamp', 'f1', 'f2', 'f3', 'avg_timestamp', 'f1_avg', 'f2_avg', 'f3_avg', 'median_row', 'accent', 'gender', 'word', 'speaker', 'window', 'duration']]
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(output_file, index=False)

## uncomment to run on training set data
# df.to_csv('train_formants.csv', index=False)


localpath = 'isye_6740_final_project_accents/test/'
pathname = os.path.join(DIRNAME, localpath)

# get_file_formants()
analyze_file(pathname, 'a_f_FAKS0-SX133-12759_DR1_03242021233521_timit.wav')


