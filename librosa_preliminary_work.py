#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:28:20 2021

@author: Andrew
"""

import librosa
import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

y, sr_1 = librosa.load('train/greasy_m_MMPM0-SA1-39100_DR8_03242021232543_timit.wav')
z, sr_2 = librosa.load('train/greasy_m_MMSM0-SA1-12113_DR3_03242021232135_timit.wav')

f = librosa.feature.mfcc(y=y, sr=sr_1)
g = librosa.feature.mfcc(y=z, sr=sr_2)

f.shape
g.shape


fig, ax = plt.subplots()
img = librosa.display.specshow(f, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')


fig, ax = plt.subplots()
img = librosa.display.specshow(g, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')



#mfccs_array = np.empty((0, 12), float)
#indices= []
#columns = ['feature_' + str(x) for x in range(1,13,1)]
#gender = []
#word = []

#for filename in os.listdir('train/'):
#    
##    if "a" == filename.split('_')[0]: 
#        
#    try:
#        y, sr = librosa.load('train/' + filename, sr=None)
#        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)         
#        indices.append(filename.split('_')[3])
#        gender.append(filename.split('_')[1])
#        word.append(filename.split('_')[0])
#        mfccs_array = np.append(mfccs_array, np.average(mfccs, axis=1).reshape(1,12), axis=0)
#    except:
#        print('could not use word {}'.format(filename.split('_')[0]))
#
#df = pd.DataFrame(mfccs_array, columns=columns, index=indices)
#df['accent'] = df.index
#df['gender'] = gender
#df['word'] = word


#df.to_csv('all_words_mfcc.csv')

df = pd.read_csv('all_words_mfcc.csv')


sns.scatterplot(data=df, x='feature_5', y='feature_6', hue='word')

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[['feature_10', 'feature_11', 'feature_12']])
df_pca = pd.DataFrame(df_pca, columns=['c1', 'c2'])
df_pca.index = df.index
#df_pca['accent'] = df_pca.index
#df_pca['gender'] = gender

df_pca['accent'] = df['accent']
df_pca['gender'] = df['gender']
df_pca['word'] = df['word']

sns.scatterplot(data=df_pca[df['word'].isin(['wash'])], x='c1', y='c2', hue='accent', alpha=0.8)

sns.scatterplot(data=df_pca[df['word'].isin(['carry', 'greasy', 'wash'])], x='c1', y='c2', hue='word', alpha=0.9)


from sklearn.ensemble import RandomForestClassifier

parameters = {'n_estimators':[1, 100, 1000]}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, scoring='accuracy')
clf.fit(df[(df.accent.isin(['DR7', 'DR1'])) & (df.word=='a')].drop(['Unnamed: 0', 'gender', 'word', 'accent'],axis=1), df[(df.accent.isin(['DR7', 'DR1'])) & (df.word=='a')]['accent'])

print(clf.best_score_)

clf.fit(df[(df.accent.isin(['DR7', 'DR1'])) & (df.gender == 'm')].drop(['Unnamed: 0', 'gender', 'word', 'accent'],axis=1), df[(df.accent.isin(['DR7', 'DR1'])) & (df.gender == 'm')]['accent'])

print(clf.best_score_)
