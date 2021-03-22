import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import wavfile
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    # word_list = timit_vocabulary('../data/archive/TIMITDIC.TXT')
    # print(word_list[0:20])

    plot_wave_file('../data/archive/', 'FALR0', 'SX335')
    plot_wave_file('../data/archive/', 'MDCD0', 'SX335')
    plot_wave_file('../data/archive/', 'FEME0', 'SX335')
    plot_wave_file('../data/archive/', 'FGMB0', 'SX335')
    plot_wave_file('../data/archive/', 'MBBR0', 'SX335')
    plot_wave_file('../data/archive/', 'MTPF0', 'SX335')
    plot_wave_file('../data/archive/', 'MRDM0', 'SX335')