def timit_vocabulary(file_path):

    """
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



if __name__ == "__main__":

    word_list = timit_vocabulary('../data/archive/TIMITDIC.TXT')
    print(word_list[0:20])