#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 21:57:38 2021

@author: Andrew
"""

# read in CSV files names from the Tabular folder

from os import listdir
from os.path import isfile, join


file_names = listdir('../data/british_raw_data/doc/Tabular')


empty = pd.DataFrame(columns = ['recording','speaker','word','word_id','duration','start','end'])
spoken = []

for file in file_names:
    