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

empty = pd.DataFrame(columns = ['recording','speaker','word','word_id','duration','start','end'])
spoken = []

for file in file_names:
    
    letter = file.split('.')[0]
    
    df = pd.read_csv('directory/{}'.format(file))
    
    recordings = df['recording'].unique()
    
    for recording in recordings:
    
        df_recording = df[df['recording'] == recording]
        df_recording = df_recording.copy().reset_index(drop=True)
    
        # Only A has subfolders
        if letter == 'A':
            data = wavfile.read('{directory}/{letter}/{recording}/{recording}.wav'.format(directory=directory,letter=letter,recording=recording))
        else:
            data = wavfile.read('{directory}/{letter}/{recording}.wav'.format(directory=directory,letter=letter,recording=recording))


        df_recording['end']=df_recording['duration'].cumsum()
        df_recording['start'] = df_recording['end'] - df_recording['duration']
        df_recording['sig_start'] = (df_recording['start']*SAMPLERATE).astype(int)
        df_recording['sig_end'] = (df_recording['end']*SAMPLERATE).astype(int)
        df_recording['word'] = df_recording['word'].str.lower()
        df_recording['phoneme'] = df_recording['phoneme'].str.lower()        
        df_recording = df_recording[['recording','speaker','phoneme','word','duration','end','start','sig_start','sig_end']]

        for row in df_recording.iterrows():
            
            recording, speaker, phoneme, word, duration, end, start, sig_start, sig_end = row
            
            output_file = '{phoneme}_{word}_{recording}_{speaker}_{start}_{end}'.format(phoneme=phoneme, speaker=speaker, recording=recording,
                                                                                        word=word, start=start, end=end)
            
            
            wav.writefile(data=data[sig_start, sig_end], filename='british_phenomes/' + output_file, rate=SAMPLERATE)