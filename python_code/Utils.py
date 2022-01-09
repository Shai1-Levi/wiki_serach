# # Import Module
import os
import pandas as pd


class Utils:
    def __init__(self, path):
        self.titles = pd.read_pickle(path + "titles.pkl")
        self.nf = pd.read_pickle(path + "doc_body_length.pkl")

    def get_page_titles(self, pages_ids):
        ''' Returns the title of the first, fourth, and fifth pages as ranked about
          by PageRank.
            Returns:
            --------
              list of three strings.
          '''
        lst = []
        for page_id in pages_ids:
            try:
                lst.append((page_id, self.titles[page_id]))
            except:
                pass
        return lst

    def get_nf(self):
        return self.nf





# # # Folder Path
# # path = "/content/wikidumps/wikidumps"

# # # Change the directory
# # os.chdir(path)

# # # Read text File

# import nltk
# from nltk.stem.porter import *
# from nltk.corpus import stopwords
# # from time import time
# # from timeit import timeit
# from pathlib import Path
# import pickle
# import pandas as pd
# import numpy as np
# from collections import Counter
# from inverted_index_colab import InvertedIndex
# import hashlib

# nltk.download('stopwords')

# english_stopwords = frozenset(stopwords.words('english'))
# all_stopwords = english_stopwords

# RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
# NUM_BUCKETS = 124


# def token2bucket_id(token):
#     return int(_hash(token), 16) % NUM_BUCKETS


# def tok(text):
#     tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
#     if len(tokens) == 0:
#         return 0
#     tokens_lst = []
#     for token in tokens:
#         if token in all_stopwords:
#             continue
#         else:
#             tokens_lst.append(token)
#     return tokens_lst


# IDX = InvertedIndex()
# # iterate through all file
# for file in os.listdir():
#     # Check whether file is in parquet format or not
#     if file.endswith(".parquet"):
#         file_path = f"{path}/{file}"
#         # call read parquet file function
#         par_df = pd.read_parquet(file_path)
#         for index, row in par_df.iterrows():
#             _tok = tok(row['title'])
#             if _tok == 0:
#                 continue
#             IDX.add_doc(row['id'], _tok)
#             # item = get_doc_length_body(row['title'])
#             # docs_length_body[row['id']] = (docs_length_body.get(row['id'], 0) + item[0], item[1])
#             # docs_length_title[row['id']] = docs_length_title.get(row['id'], 0) + get_doc_length_title(row['title'])
#         print(file)


# def _hash(s):
#     return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

# # # Change the directory
# os.chdir("/content/postings_gcp_title2/")
# # b_w_pl2 = {}
# # for word in IDX.df.keys():
# #   buc = token2bucket_id(word)
# #   if buc in b_w_pl2.keys():
# #     b_w_pl2[buc].append((word, IDX._posting_list[word]))
# #   else:
# #     b_w_pl2[buc] = [(word, IDX._posting_list[word])]
# super_posting_locs =[]
# for item in b_w_pl2.items():
#   super_posting_locs.append(IDX.write_a_posting_list(item))
#   # IDX.write_a_posting_list((buc,[(word, IDX._posting_list[word])]))


# IDXT = InvertedIndex.read_index("/content/postings_gcp_title_new/", "index")
# from contextlib import closing
# from inverted_index_colab import InvertedIndex, MultiFileReader
# TUPLE_SIZE = 6

# from collections import defaultdict
# super_posting_locs2 = defaultdict(list)
# for posting_loc in super_posting_locs:
#   for k, v in posting_loc.items():
#     super_posting_locs2[k].extend(v)

# # Create inverted index instance
# inverted = InvertedIndex()
# # Adding the posting locations dictionary to the inverted index
# inverted.posting_locs = super_posting_locs2
# # Add the token - df dictionary to the inverted index
# inverted.df = IDX.df
# # write the global stats out
# inverted.write_index('.', 'index')


# def read_posting_list(w):
#     with closing(MultiFileReader()) as reader:
#       locs = IDXT.posting_locs[w]
#       # locs = [('postings_gcp_82_015.bin', 1619898)]
#       b = reader.read(locs, IDXT.df[w] * TUPLE_SIZE)
#       posting_list = []
#       for i in range(IDXT.df[w]):
#         doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
#         tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
#         # posting_list.append((doc_id, self.get_page_titles(doc_id)))
#         posting_list.append((doc_id,tf))
#       return posting_list






# # from pathlib import Path
# # # Paths
# # # Using user page views (as opposed to spiders and automated traffic) for the
# # # month of August 2021
# # pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
# # p = Path(pv_path)
# # pv_name = p.name
# # pv_temp = f'{p.stem}-4dedup.txt'
# # pv_clean = f'{p.stem}.pkl'
# # # Download the file (2.3GB)
# # # !wget -N $pv_path
# # # Filter for English pages, and keep just two fields: article Title (2), article ID (3) and monthly
# # # total number of page views (5). Then, remove lines with article id or page
# # # view values that are not a sequence of digits.
# # !bzcat $pv_name | grep "^en\.wikipedia" | cut -d' ' -f2,3,5 > $pv_temp
# #
# # pv_temp = "pageviews-202108-user-4dedup_old.txt"
# # from collections import Counter
# # import pickle
# # wid2pv = Counter()
# # doc_id2title = Counter()
# # with open(pv_temp, 'rt') as f:
# #   for line in f:
# #     # print(line.split(' '))
# #     parts = line.split(' ')
# #     # print(parts)
# #     wid2pv.update({int(parts[1]): int(parts[2])})
# #     doc_id2title[int(parts[1])] = str(parts[0])
# # # write out the counter as binary file (pickle it)
# #
# # S = sorted(wid2pv.items())
# # c = Counter()
# # C = Counter()
# # i = 200000
# # for s_id, s_c in S:
# #   if i - s_id > 0:
# #     c[s_id] = (s_c, doc_id2title[s_id])
# #   else:
# #     C[i] = c
# #     i+=200000
# #     c = Counter()
# #     c[s_id]= (s_c, doc_id2title[s_id])
# # C[i] = c
# #
# # for file_name in range(200000, i, 200000):
# #   with open("wid2pv/{}.pkl".format(file_name), 'wb') as f:
# #     pickle.dump(C[file_name], f)


