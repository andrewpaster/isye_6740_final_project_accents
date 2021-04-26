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


def analyze_american_file(file_info):
    """ outputs forman information for a single recording

        inputs: file_name (str): name of file_name
        returns: output (list): list of formant values and file meta data
    """
    try:
        # read in file and parse the meta data like speaker id, sex, accent
        pathname, file_name = file_info
        file_path = os.path.join(pathname, file_name)
        sound = parselmouth.Sound(file_path)
        duration = sound.get_total_duration()
        formant = sound.to_formant_burg(maximum_formant=8000)
        bandwidth = [formant.get_bandwidth_at_time(x, duration / 2.0) for x in [1,2,3,4,5]]
        value = [formant.get_value_at_time(x, duration / 2.0) for x in [1,2,3,4,5]]
        
        # get mfcc 
        if duration/4 <= 0.015:
            mfcc = sound.to_mfcc(window_length=duration/4, time_step=duration/12).to_array()
        
        else:
            mfcc = sound.to_mfcc().to_array()

        mfcc_avg = np.average(mfcc, axis=1).reshape(1,13)
        mfcc_median = np.median(mfcc, axis=1).reshape(1,13)
    
        split_file = file_name.split('_')
        accent = split_file[4]
        gender = split_file[2]
        word = split_file[1]
        phoneme = split_file[0]
        speaker = split_file[3].split('-')[0]
        sentence = split_file[3].split('-')[1]
  
  
        # store meta data
        meta_data = np.array([accent, gender, phoneme, word, sentence, speaker, duration]).reshape(1,7)
        value = np.array(value).reshape(1,5)
        bandwidth = np.array(value).reshape(1,5)
        
        # format results
        formants_stacked = np.hstack((meta_data, value, bandwidth, mfcc_avg, mfcc_median)).reshape(1,43)
          
        # return output as a list
        return formants_stacked[0]

    except:

        # when exception occurs, return list with 0 values and meta data
        pathname, file_name = file_info

        print('could not use word {} \n'.format(file_name))
        
        output = [0]*43
        
        split_file = file_name.split('_')
        accent = split_file[4]
        gender = split_file[2]
        word = split_file[1]
        phoneme = split_file[0]
        speaker = split_file[3].split('-')[0]
        sentence = split_file[3].split('-')[1]

        output[0] = accent
        output[1] = gender
        output[2] = phoneme
        output[3] = word
        output[4] = sentence
        output[5] = speaker

        return output


def analyze_british_file(file_info):
    """ outputs forman information for a single recording

        inputs: file_name (str): name of file_name
        returns: output (list): list of formant values and file meta data
    """
    try:
        # read in file and parse the meta data like speaker id, sex, accent
        pathname, file_name = file_info
        file_path = os.path.join(pathname, file_name)
        sound = parselmouth.Sound(file_path)
        duration = sound.get_total_duration()
        formant = sound.to_formant_burg(maximum_formant=8000)
        bandwidth = [formant.get_bandwidth_at_time(x, duration / 2.0) for x in [1,2,3,4,5]]
        value = [formant.get_value_at_time(x, duration / 2.0) for x in [1,2,3,4,5]]
        
        # get mfcc         
        if duration/4 <= 0.015:
            mfcc = sound.to_mfcc(window_length=duration/4, time_step=duration/12).to_array()
        
        else:
            mfcc = sound.to_mfcc().to_array()

        
        mfcc_avg = np.average(mfcc, axis=1).reshape(1,13)
        mfcc_median = np.median(mfcc, axis=1).reshape(1,13)
    
        split_file = file_name.split('_')
        accent = 'British'
        gender = split_file[3][0].lower()
        word = split_file[1]
        phoneme = split_file[0].replace(',', '').replace("'", '')
        speaker = split_file[3]
        recording = split_file[2]
  
  
        # store meta data
        meta_data = np.array([accent, gender, phoneme, word, recording, speaker, duration]).reshape(1,7)
        value = np.array(value).reshape(1,5)
        bandwidth = np.array(value).reshape(1,5)
        
        # format results
        formants_stacked = np.hstack((meta_data, value, bandwidth, mfcc_avg, mfcc_median)).reshape(1,43)
          
        # return output as a list
        return formants_stacked[0]

    except:

        # when exception occurs, return list with 0 values and meta data
        print('could not use word {} \n'.format(file_name))
        pathname, file_name = file_info

        output = [0]*43
        
        try:
            split_file = file_name.split('_')
            accent = 'British'
            gender = split_file[3][0].lower()
            word = split_file[1]
            phoneme = split_file[0].replace(',', '').replace("'", '')
            speaker = split_file[3]
            recording = split_file[2]
    
            output[0] = accent
            output[1] = gender
            output[2] = phoneme
            output[3] = word
            output[4] = recording
            output[5] = speaker
        
        except:
            pass

        return output



def get_file_formants(output_file='train_formants_phonemes.csv', localpath = 'isye_6740_final_project_accents/phoneme_train/', accent='american'):
    

    pathname = os.path.join(DIRNAME, localpath)

    # get all file names
    print('getting file names')
    file_names = [x for x in os.listdir(pathname)]
    print(len(file_names), 'number of files')
    
    # make an empty list to hold results for each file
    result = [None]*len(file_names)
    
    print('analyzing files')
    
    if accent == 'american':
        function = analyze_american_file
    else:
        function= analyze_british_file
        
    # parallelize the file analysis
    pool = Pool(4)    
    file_list = list(product([pathname], file_names))
    for i, temparray in enumerate(pool.imap(function, file_list)):
        result[i] = temparray
        if i % 500 == 0:
            print(i, file_names[i])
    
    pool.close()
    pool.join()
    
    # output results to dataframe
    columns = ['accent', 'gender', 'phoneme', 'word', 'recording', 'speaker', 'duration', 'f1', 'f2', 'f3', 'f4', 'f5', 'bw1', 'bw2', 'bw3', 'bw4', 'bw5']
    mfcc_columns = ['energy_avg'] + ['mfcc_avg_' + str(x + 1) for x in range(12)] + ['energy_median'] + ['mfcc_median_' + str(x + 1) for x in range(12)]
    columns = columns + mfcc_columns
    df = pd.DataFrame(result, columns=[columns])
    df.to_csv(output_file, index=False)


# localpath = 'isye_6740_final_project_accents/phoneme_test/'
# get_file_formants('test_formants_phonemes.csv', localpath)

# localpath = 'isye_6740_final_project_accents/phoneme_train/'
# get_file_formants('train_formants_phonemes.csv', localpath)

localpath = 'isye_6740_final_project_accents/british_phonemes/'
get_file_formants('british_formants_phonemes.csv', localpath, 'british')


# analyze_american_file('/Users/Andrew/Desktop/final_project_gtech/isye_6740_final_project_accents/phoneme_test/', 'd_discipline_m_MRJM4-SI2119-43570_DR7_04252021105315_timit.wav')
# analyze_british_file('/Users/Andrew/Desktop/final_project_gtech/isye_6740_final_project_accents/british_phonemes/', ":_weren't_A01_F1_144.49099999999999_144.521")

