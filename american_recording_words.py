import pandas as pd
import os 
from collections import Counter
import re


training_csv = '../data/archive/train_data.csv'
test_csv = '../data/archive/test_data.csv'
data_dir = '../data/archive/data/'


df_training = pd.read_csv(training_csv)
df_test = pd.read_csv(test_csv)

df = df_training.append(df_test)

df = df[df['is_converted_audio'] == True].reset_index(drop=True)
df['filepath'] = df.apply(lambda row: os.path.join(data_dir,
                                                               row['test_or_train'],
                                                               row['dialect_region'],
                                                               row['speaker_id'],
                                                               row['filename']), axis=1)


df['textfilepath'] = df['filepath'].apply(lambda w: w.replace('.WAV.wav', '') + '.TXT')


train_words = Counter()
test_words = Counter()


for i, files in enumerate(df.loc[df['test_or_train'] == 'TRAIN', 'textfilepath']):
    
    if i % 100 == 0:
        print('{}th file processed'.format(i))
    
    with open(files) as f:
        
        for line in f.readlines():
            
            sentence = re.sub(r'[\.!?]', '', ' '.join(line.split(' ')[2:])).strip('\r\n')
            
            for word in sentence.split(' '):
                train_words[word.lower()] += 1
                


for i, files in enumerate(df.loc[df['test_or_train'] == 'TEST', 'textfilepath']):
    
    
    if i % 100 == 0:
        print('{}th file processed'.format(i))

    with open(files) as f:
        
        for line in f.readlines():
            
            sentence = re.sub(r'[\.!?]', '', ' '.join(line.split(' ')[2:])).strip('\r\n')
            
            for word in sentence.split(' '):
                test_words[word.lower()] += 1


df_train_counts = pd.DataFrame.from_dict(train_words, orient='index', columns=['counts'])
df_test_counts = pd.DataFrame.from_dict(test_words, orient='index', columns=['counts'])

df_train_counts['test_or_train'] = 'TRAIN'
df_test_counts['test_or_train'] = 'TEST'

df_train_counts.append(df_test_counts).to_csv('word_counts.csv', index_label='word')
