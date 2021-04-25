import parselmouth
import pathlib
import os 
import pandas as pd
import numpy as np
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
        sound = parselmouth.Sound(file_path)
        duration = sound.get_total_duration()
        formant = sound.to_formant_burg(maximum_formant=8000)
        bandwidth = [formant.get_bandwidth_at_time(x, duration / 2.0) for x in [1,2,3,4,5]]
        value = [formant.get_value_at_time(x, duration / 2.0) for x in [1,2,3,4,5]]
    
        accent = file_name.split('_')[3]
        gender = file_name.split('_')[1]
        word = file_name.split('_')[0]
        speaker = file_name.split('_')[2].split('-')[0]
  
  
        # store meta data
        meta_data = np.array([accent, gender, word, speaker, duration]).reshape(1,5)
  
        # format results
        formants_stacked = np.hstack((value, bandwidth)).reshape(1,10)
        formants_stacked = np.hstack((formants_stacked, meta_data)).reshape(1,15)
          
        # return output as a list
        return formants_stacked[0]

    except:

        # when exception occurs, return list with 0 values and meta data
        print('could not use word {} \n'.format(file_name))
        
        output = [0]*15
        
        accent = file_name.split('_')[3]
        gender = file_name.split('_')[1]
        word = file_name.split('_')[0]
        speaker = file_name.split('_')[2].split('-')[0]

        output[10] = accent
        output[11] = gender
        output[12] = word
        output[13] = speaker

        return output


def get_file_formants(output_file='train_formants_phonemes.csv', localpath = 'isye_6740_final_project_accents/phoneme_train/'):
    

    pathname = os.path.join(DIRNAME, localpath)

    # get all file names
    print('getting file names')
    file_names = [x for x in os.listdir(pathname)]
    
    # make an empty list to hold results for each file
    result = [None]*len(file_names)
    
    print('analyzing files')
    # parallelize the file analysis
    pool = Pool(4)    
    for i, temparray in enumerate(pool.starmap(analyze_file, product([pathname], file_names))):
        result[i] = temparray
        if i % 500 == 0:
            print(i, file_names[i])
    
    pool.close()
    pool.join()
    
    # output results to dataframe
    columns = [['f1', 'f2', 'f3', 'f4', 'f5', 'bw1', 'bw2', 'bw3', 'bw4', 'bw5', 'accent', 'gender', 'word', 'speaker', 'duration']]
    df = pd.DataFrame(result, columns=columns)
    df.to_csv(output_file, index=False)


localpath = 'isye_6740_final_project_accents/phoneme_train/'
get_file_formants('train_formants_phonemes.csv', localpath)


