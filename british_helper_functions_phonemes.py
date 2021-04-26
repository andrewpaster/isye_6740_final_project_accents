#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 21:57:38 2021

@author: Andrew
"""

# read in CSV files names from the Tabular folder

from os import listdir
from os.path import isfile, join
import pandas as pd
from scipy.io import wavfile


directory = '../data/british_data_raw/doc/Tabular'
file_names = listdir(directory)

SAMPLERATE = 16000


for file in file_names:
    
    letter = file.split('.')[0]
    
    df = pd.read_csv('{}/{}'.format(directory, file))
    
    recordings = df['recording'].unique()
    
    for recording in recordings:
    
        print(recording, ' currently being parsed')
        df_recording = df[df['recording'] == recording]
        df_recording = df_recording.copy().reset_index(drop=True)
    
        # Only A has subfolders
        if letter == 'A':
            sr, data = wavfile.read('../data/british_data_raw/Sounds/{letter}/{recording}/{recording}.wav'.format(directory=directory,letter=letter,recording=recording))
        else:
            sr, data = wavfile.read('../data/british_data_raw/Sounds/{letter}/{recording}.wav'.format(directory=directory,letter=letter,recording=recording))


        df_recording['end']=df_recording['duration'].cumsum()
        df_recording['start'] = df_recording['end'] - df_recording['duration']
        df_recording['sig_start'] = (df_recording['start']*SAMPLERATE).astype(int)
        df_recording['sig_end'] = (df_recording['end']*SAMPLERATE).astype(int)
        df_recording['word'] = df_recording['word'].str.lower()
    
        df_recording = df_recording[['recording','speaker','phoneme','word','duration','end','start','sig_start','sig_end']]
        
        for i, row in df_recording.iterrows():
            
            if i % 1000 == 0:
                print(i, ' phoneme')
            
            recording, speaker, phoneme, word, duration, end, start, sig_start, sig_end = row
            
            output_file = '{phoneme}_{word}_{recording}_{speaker}_{start}_{end}.wav'.format(phoneme=phoneme, speaker=speaker, recording=recording,
                                                                                        word=word, start=start, end=end)
            
            
            wavfile.write(data=data[sig_start:sig_end], filename='british_phonemes/' + output_file, rate=SAMPLERATE)