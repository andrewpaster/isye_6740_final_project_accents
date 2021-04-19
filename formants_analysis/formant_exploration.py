#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:32:19 2021

@author: Andrew
"""

### Analysis of formant data features

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from itertools import combinations

dirname = 'formants_analysis'
if not os.path.exists(dirname):
    os.makedirs(dirname)

genderplots = dirname + '/gender_word_plots'
if not os.path.exists(genderplots):
    os.makedirs(genderplots)

wordplots = dirname + '/word_plots'
if not os.path.exists(wordplots):
    os.makedirs(wordplots)

accentplots = dirname + '/accent_plots'
if not os.path.exists(accentplots):
    os.makedirs(accentplots)

regionplots = dirname + '/region_plots'
if not os.path.exists(regionplots):
    os.makedirs(regionplots)

df_train = pd.read_csv('train_formants.csv')
df_test = pd.read_csv('test_formants.csv')


df_train['set'] = 'train'
df_test['set'] = 'test'

df = df_train.append(df_test).reset_index(drop=True)

# ratios of the formant features
df['ratio_f1_f2'] = df['f2'] / df['f1']
df['ratio_f1_f3'] = df['f3'] / df['f1']
df['ratio_f2_f3'] = df['f3'] / df['f2']
df['ratio_avg_f1_f2'] = df['f2_avg'] / df['f1_avg']
df['ratio_avg_f1_f3'] = df['f3_avg'] / df['f1_avg']
df['ratio_avg_f2_f3'] = df['f3_avg'] / df['f2_avg']

# human-readable names for accents
df['accent_name'] = None
df.loc[df.accent == 'DR1', 'accent_name'] = 'new_england'
df.loc[df.accent == 'DR2', 'accent_name'] = 'northern'
df.loc[df.accent == 'DR3', 'accent_name'] = 'north_midland'
df.loc[df.accent == 'DR4', 'accent_name'] = 'south_midland'
df.loc[df.accent == 'DR5', 'accent_name'] = 'southern'
df.loc[df.accent == 'DR6', 'accent_name'] = 'new_york_city'
df.loc[df.accent == 'DR7', 'accent_name'] = 'western'
df.loc[df.accent == 'DR8', 'accent_name'] = 'army_brat'

df['accent_group'] = None
df.loc[df.accent_name.isin(['new_england', 'new_york_city', 
                                        'northern', 'southern', 
                                        'north_midland', 'south_midland']), 
    'accent_group'] = 'eastern'
df.loc[df.accent_name.isin(['western']), 'accent_group'] = 'western'
df.loc[df.accent_name.isin(['army_brat']), 'accent_group'] = 'other'


df = df.groupby(['speaker', 'gender', 'accent_name', 'accent_group', 'word', 'set'], as_index=False)[['f1', 'f2', 'f3', 'f1_avg', 'f2_avg', 'f3_avg', 'ratio_f1_f2', 'ratio_f1_f3', 'ratio_f2_f3', 'ratio_avg_f1_f2', 'ratio_avg_f1_f3', 'ratio_avg_f2_f3']].mean()


# Top 25 words in traing and test set
df_train.word.value_counts()[0:25].plot(kind='bar')
plt.title('Most common spoken words in training set')
plt.ylabel('spoken word count')
plt.savefig(dirname + '/top_25_training_words.png')


df_test.word.value_counts()[0:25].plot(kind='bar')
plt.title('Most common spoken words in test set')
plt.ylabel('spoken word count')
plt.savefig(dirname + '/top_25_test_words.png')

top_word_list = set(list(df_train.word.value_counts()[0:25].index) + list(df_test.word.value_counts()[0:25].index))

# plot word by gender
def plot_by_gender(word, feature1, feature2):
    
    g = sns.FacetGrid(df.loc[df.word==word], col="set", hue='gender', col_order=['train', 'test'])
    g.map(sns.scatterplot, feature1, feature2, alpha=0.8)
    g.set_titles(col_template="{col_name} - " + word, row_template="{row_name}")
    g.add_legend()

    plt.savefig('{}/{}_gender_{}_{}.png'.format(genderplots, word, feature1, feature2))

for word in top_word_list:
    plot_by_gender(word, 'ratio_avg_f1_f2', 'ratio_avg_f2_f3')
    plot_by_gender(word, 'ratio_avg_f1_f2', 'ratio_avg_f1_f3')
    plot_by_gender(word, 'ratio_avg_f1_f3', 'ratio_avg_f2_f3')
    plot_by_gender(word, 'f1', 'f2')
    plot_by_gender(word, 'f1', 'f3')
    plot_by_gender(word,  'f2', 'f3')


# plot by word
def plot_by_word(words, feature1, feature2):
    # words - list of words
    
    g = sns.FacetGrid(df.loc[df.word.isin(words)], col='set', row='gender', hue='word', col_order=['train', 'test'])
    g.map(sns.scatterplot, feature1, feature2, alpha=0.8)
    g.set_titles(col_template="{col_name} - " + str(words), row_template="{row_name}")
    g.add_legend()
    plt.savefig('{}/{}_words_{}_{}.png'.format(wordplots, str(words), feature1, feature2))
    
for i in list(combinations(list(top_word_list)[0:5],3)):
    plot_by_word(i, 'ratio_avg_f1_f2', 'ratio_avg_f2_f3')
    plot_by_word(i, 'ratio_avg_f1_f2', 'ratio_avg_f1_f3')
    plot_by_word(i, 'ratio_avg_f1_f3', 'ratio_avg_f2_f3')


plot_by_word(list(top_word_list)[0:6], 'ratio_avg_f1_f2', 'ratio_avg_f2_f3')
plot_by_word(list(top_word_list)[0:6], 'ratio_avg_f1_f2', 'ratio_avg_f1_f3')
plot_by_word(list(top_word_list)[0:6], 'ratio_avg_f1_f3', 'ratio_avg_f2_f3')

plot_by_word(list(top_word_list)[0:6], 'f1_avg', 'f2_avg')
plot_by_word(list(top_word_list)[0:6], 'f1_avg', 'f3_avg')
plot_by_word(list(top_word_list)[0:6], 'f2_avg', 'f3_avg')


plot_by_word(list(top_word_list)[6:10], 'ratio_avg_f1_f2', 'ratio_avg_f2_f3')
plot_by_word(list(top_word_list)[6:10], 'ratio_avg_f1_f2', 'ratio_avg_f1_f3')
plot_by_word(list(top_word_list)[6:10], 'ratio_avg_f1_f3', 'ratio_avg_f2_f3')
plot_by_word(list(top_word_list)[6:10], 'f1_avg', 'f2_avg')
plot_by_word(list(top_word_list)[6:10], 'f1_avg', 'f3_avg')
plot_by_word(list(top_word_list)[6:10], 'f2_avg', 'f3_avg')


# plot by accent
def plot_by_accent(word, feature1, feature2):
    
    g = sns.FacetGrid(df.loc[df.word == word], col='set', row='gender', hue='accent_name', col_order=['train', 'test'])
    g.map(sns.scatterplot, feature1, feature2, alpha=0.8)
    g.set_titles(col_template="{col_name} - " + word, row_template="{row_name}")
    g.add_legend()
    plt.savefig('{}/{}_accents_{}_{}.png'.format(accentplots, word, feature1, feature2))

for word in top_word_list:    
    plot_by_accent(word, 'ratio_avg_f1_f2', 'ratio_avg_f2_f3')
    plot_by_accent(word, 'ratio_avg_f1_f2', 'ratio_avg_f1_f3')
    plot_by_accent(word, 'ratio_avg_f1_f3', 'ratio_avg_f2_f3')
    plot_by_accent(word, 'f1', 'f2')
    plot_by_accent(word, 'f1', 'f3')
    plot_by_accent(word, 'f2', 'f3')
    

# plot by region
def plot_by_region(word, feature1, feature2):
    
    g = sns.FacetGrid(df.loc[df.word == word], col='set', row='gender', hue='accent_group', col_order=['train', 'test'])
    g.map(sns.scatterplot, feature1, feature2, alpha=0.8)
    g.set_titles(col_template="{col_name} - " + word, row_template="{row_name}")
    g.add_legend()
    plt.savefig('{}/{}_accent_group_{}_{}.png'.format(regionplots, word, feature1, feature2))

for word in top_word_list:    
    plot_by_region(word, 'ratio_avg_f1_f2', 'ratio_avg_f2_f3')
    plot_by_region(word, 'ratio_avg_f1_f2', 'ratio_avg_f1_f3')
    plot_by_region(word, 'ratio_avg_f1_f3', 'ratio_avg_f2_f3')
    plot_by_region(word, 'f1', 'f2')
    plot_by_region(word, 'f1', 'f3')
    plot_by_region(word, 'f2', 'f3')