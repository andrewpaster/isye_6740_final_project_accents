import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import wavfile
import matplotlib.pyplot as plt
import time
from itertools import product

def timit_vocabulary(file_path):

    """
    Outputs a list of vocabulary in the timit data set based on the TIMITDIC.TXT file
    :param file_path: location of TIMITDIC.TXT
    :return: list of vocabulary
    """

    word_list = []
    with open(file_path) as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            if line[0] != ';':
                if line[0] == '-':
                    line = line[1:]

                word = line.split()[0]
                word_list.append(word)

    return word_list




def plot_wave_file(data_path, speaker_id, sentence_code):

    """
    Makes a waveform plot for a speaker_id - sentence code combination
    reference: https://www.kaggle.com/andregoios/darpa-timit-sample

    :param data_path: relative path to the data directory archive folder
            speaker_id: speaker id from the timit data set
            sentence code: sentence code of the file of interest
    :return: None - saves a png file of the waveform with the annotations

    """

    # TODO: make sure that you cannot ask for a speaker_id sentence_code combination that does not exist


    # get the data file meta data into a single data frame
    df_train = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    df_test = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    df_data = df_train.append(df_test)

    # search for the audio file of interest
    src_audio_file = df_data[(df_data['is_audio'] == True) &
                              (df_data['is_converted_audio'] == True) &
                              (df_data['speaker_id'] == speaker_id) &
                              (df_data['filename']).str.contains(sentence_code)]

    audio_file_path = src_audio_file['path_from_data_dir'].values[0]
    dialect_region = src_audio_file['dialect_region'].values[0]
    basename = src_audio_file['filename'].values[0].split('.')[0]

    # get the other files related to this WAV file recording
    audio_related_files = df_data[(df_data['filename'].str.contains(basename + '.').fillna(False)) &
                                   (df_data['dialect_region'] == dialect_region) &
                                   (df_data['speaker_id'] == speaker_id)]

    # load the word file name and phoneme file name that have the timestamps in them
    word_file_name = audio_related_files[audio_related_files['filename'].str.contains('.WRD')].iloc[0]
    phon_file_name = audio_related_files[audio_related_files['filename'].str.contains('.PHN')].iloc[0]

    word_df = pd.read_csv(os.path.join(data_path, 'data', word_file_name['path_from_data_dir']), sep=' ',
                                 header=None, names=['nsam_start', 'nsam_end', 'word'])
    word_df[['nsam_start', 'nsam_end']] = word_df[['nsam_start', 'nsam_end']].astype('i')

    phon_df = pd.read_csv(os.path.join(data_path, 'data', phon_file_name['path_from_data_dir']), sep=' ',
                                 header=None, names=['nsam_start', 'nsam_end', 'phoneme'])
    phon_df[['nsam_start', 'nsam_end']] = phon_df[['nsam_start', 'nsam_end']].astype('i')

    sr, ws = wavfile.read(os.path.join(data_path, 'data', audio_file_path))
    w = ws / max(ws)
    fig, ax = plt.subplots(2, figsize=(18, 8), sharex=True)
    ax[0].plot(w)
    ax[1].specgram(w, Fs=1, NFFT=2 ** 8, cmap='gray')
    for ir, row in word_df.iterrows():
        for axi in ax:
            axi.axvline(row['nsam_start'], color='r', alpha=.5)
        ax[0].annotate(row['word'], ((row['nsam_start'] + row['nsam_end']) / 2, .9),
                       bbox=dict(boxstyle="round", fc="0.9"), ha='center')
    for ir, row in phon_df.iterrows():
        for axi in ax:
            axi.axvline(row['nsam_start'], color='g', alpha=.5, lw=1)
        ax[1].annotate(row['phoneme'], ((row['nsam_start'] + row['nsam_end']) / 2, 0.45), ha='center')

    plt.savefig('{}_{}_{}'.format(speaker_id, sentence_code, dialect_region))

def phoneme_sounds(data_file,
                data_dir='../data/archive/data/',
                output_directory='train'):

    # reference: https://www.kaggle.com/mfekadu/extract-all-words-from-timit

    RATE = 16000  # KHz

    # get training data audio file meta data
    df = pd.read_csv(data_file)
    df = df[df['is_converted_audio'] == True].reset_index()
    df['filepath'] = df.apply(lambda row: os.path.join(data_dir,
                                                                   row['test_or_train'],
                                                                   row['dialect_region'],
                                                                   row['speaker_id'],
                                                                   row['filename']), axis=1)

    wav_files = df['filepath']
    print('reading audio data')
    audio_data = [wavfile.read(wav)[1] for wav in wav_files]

    
    print('aligning phoneme info')
    time_aligned_words = [parse_wrd_timestamps(w.replace('.WAV.wav', '') + '.PHN') for w in wav_files]
    
    print('parsing waves')
    word_aligned_audio = parse_word_waves(time_aligned_words, audio_data)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    i = 1
    for sentence in word_aligned_audio:
        for word_tup in sentence:
            timestamp = time.strftime("%m%d%Y%H%M%S", time.localtime())
            data, phoneme, word, speaker, sentence, region, train_or_test, gender = word_tup
            # gender = 'gender-speaker-id'
            # location = 'unknown-location'
            # loudness = 'unknown-loudness'
            # lastname = 'lastname-speaker-id'
            # firstname = 'firstname-speaker-id'
            nametag = 'timit'
            description = speaker + '-' + sentence + '-' + str(i)
            filename = phoneme + '_' + word + '_' + gender + '_' + description.replace('.PHN', '') + '_' + region
            filename += '_' + timestamp + '_' + nametag
            filename += '.wav'

            # filenames cannot have single quotes
            filename = filename.replace("'", '')

            wavfile.write(data=data, filename=output_directory + '/' + filename, rate=RATE)
            # print(data, filename)
            i += 1


# Given a file path to the .WRD file
# Output a list of tuples containing (start, end, word, speaker_id, sentence_id)
def parse_wrd_timestamps(wrd_path, verbose=False):
    
    print('wrd_path', wrd_path) if verbose else None
    speaker_id = wrd_path.split('/')[-2]
    sentence_id = wrd_path.split('/')[-1].replace('.WRD', '')
    region_id = wrd_path.split('/')[-3]
    train_or_test = wrd_path.split('/')[-4]
    gender = wrd_path.split('/')[-2][0].lower()

    phm_file = open(wrd_path)
    content = phm_file.read()
    content = content.split('\n')
    content = [x for x in content if x != '']

    wrd_file = open(wrd_path.replace('.PHN', '.WRD'))
    wrd_content = wrd_file.read()
    wrd_content = wrd_content.split('\n')
    wrd_content = [x for x in wrd_content if x != '']


    # print('content b4 tuple', content) if verbose else None
    
    # get the word that the phoneme belongs to
    final_content = []
    for record in product(content, wrd_content):
        
        phm_start, phm_end, phm = record[0].split(' ')
        wrd_start, wrd_end, wrd = record[1].split(' ')
        
        if phm_start >= wrd_start and phm_start < wrd_end:
            
            final_content.append([phm_start, phm_end, phm, wrd])
                
        
    # content = [(x[0].split(' ')[0]], x[0].split(' ')[1], x[0].split(' ')[2], x[1].split(' ')[2]) for x in product(content, wrd_content)\
    #            if (int(x[0].split(' ')[0]) >= int(x[1].split(' ')[0])) and (int(x[0].split(' ')[0]) < int(x[1].split(' ')[1]))]
    
    content = [tuple(foo + [speaker_id,
                                   sentence_id,
                                   region_id,
                                   train_or_test,
                                   gender]) for foo in final_content if foo != '']
    phm_file.close()
    wrd_file.close()

    return content


# Given both a time_aligned_words file && the output of read_audio()
# Output the another list of tuples containing (audio_data, label)
# e.g.
# [(array([ 2, 2, -3, ... , 3, 6, 1], dtype=int16), critical),
#   ... ((array([ 5, -6, 4, ... , 1, 3, 3], dtype=int16),maintenance)]
def parse_word_waves(time_aligned_words, audio_data, verbose=False):
    return [align_data(data, words, verbose) for data, words in zip(audio_data, time_aligned_words)]


# given numpy wave array and time alignment details
# output a list of each data with its word
def align_data(data, words, verbose=False):
    aligned = []
    print('len(data)', len(data)) if verbose else None
    print('len(words)', len(words)) if verbose else None
    print('data', data) if verbose else None
    print('words', words) if verbose else None
    for tup in words:
        print('tup', tup) if verbose else None
        start = int(tup[0])
        end = int(tup[1])
        phoneme = tup[2]
        word = tup[3]
        speaker_id = tup[4]
        sentence_id = tup[5]
        region_id = tup[6]
        train_test = tup[7]
        gender = tup[8]
        assert start >= 0
        assert end <= len(data)
        aligned.append((data[start:end], phoneme, word, speaker_id, sentence_id, region_id, train_test, gender))
    assert len(aligned) == len(words)
    return aligned


if __name__ == "__main__":

    # word_list = timit_vocabulary('../data/archive/TIMITDIC.TXT')
    # print(word_list[0:20])

    # plot_wave_file('../data/archive/', 'FALR0', 'SX335')
    # plot_wave_file('../data/archive/', 'MDCD0', 'SX335')
    # plot_wave_file('../data/archive/', 'FEME0', 'SX335')
    # plot_wave_file('../data/archive/', 'FGMB0', 'SX335')
    # plot_wave_file('../data/archive/', 'MBBR0', 'SX335')
    # plot_wave_file('../data/archive/', 'MTPF0', 'SX335')
    # plot_wave_file('../data/archive/', 'MRDM0', 'SX335')

    # phoneme_sounds('../data/archive/train_data.csv', '../data/archive/data/', 'phoneme_train')
    phoneme_sounds('../data/archive/test_data.csv', '../data/archive/data/', 'phoneme_test')