def main():




    # set paths to the dowloaded data as variables
    PATH_TO_CRAN_TXT = 'cran.all.1400'
    PATH_TO_CRAN_QRY = 'cran.qry'
    PATH_TO_CRAN_REL = 'cranqrel'

    from collections import defaultdict, Counter
    import re
    import json
    from io import StringIO
    import numpy as np
    import nltk
    from nltk.corpus import stopwords
    from pathlib import Path
    from operator import itemgetter
    import pickle

    cran_index = "cranfield-corpus"

    # get the text entries from the text and query files
    ID_marker = re.compile('\.I')


    def get_data(PATH_TO_FILE, marker):
        """
        Reads file and spilts text into entries at the ID marker '.I'.
        First entry is empty, so it's removed.
        'marker' contains the regex at which we want to split
        """
        with open(PATH_TO_FILE, 'r') as f:
            text = f.read().replace('\n', " ")
            lines = re.split(marker, text)
            lines.pop(0)
        return lines


    # get the data from paths
    cran_txt_list = get_data(PATH_TO_CRAN_TXT, ID_marker)
    cran_qry_list = get_data(PATH_TO_CRAN_QRY, ID_marker)

    # process text file

    cran_chunk_start = re.compile('\.[A,B,T,W]')
    cran_txt_data = defaultdict(dict)

    # read line by line the file and split preprocess it.
    for line in cran_txt_list:
        entries = re.split(cran_chunk_start, line)
        id = entries[0].strip()
        title = ''.join(entries[1])
        author = entries[2]
        publication_date = entries[3]
        text = ''.join(entries[4:])
        if len(text) < 2:
            continue
        if text.startswith(title) and len(text) > len(title):
            text = text[len(title):]
        cran_txt_data[id]['title'] = title
        cran_txt_data[id]['author'] = author
        cran_txt_data[id]['publication_date'] = publication_date
        cran_txt_data[id]['text'] = text

    # process query file

    qry_chunk_start = re.compile('\.W')
    cran_qry_rel_data = defaultdict(dict)

    for n in range(0, len(cran_qry_list)):
        line = cran_qry_list[n]
        _, question = re.split(qry_chunk_start, line)
        cran_qry_rel_data[n + 1]['question'] = question

    # process relevance assesments without rating
    cran_rel = defaultdict(list)

    with open(PATH_TO_CRAN_REL, 'r') as f:
        for line in f:
            line = re.split(' ', line)
            if line[1] not in cran_txt_data:
                continue

            # clean unwanted characters
            line[2] = line[2].replace('-', '')  # replace -1 to 1
            line[2] = line[2].replace('\n', '')  # removing \n from the relevant score
            if '' in line: line.remove('')  # removing white space
            cran_rel[int(line[0])].append((line[1], line[2]))

    for id, rels in cran_rel.items():
        try:
            cran_qry_rel_data[id]['relevance_assessments'] = [(int(r[0]), int(r[1])) for r in rels]
        except:
            print('unpacking failed')


    return cran_txt_data,cran_qry_rel_data


main()