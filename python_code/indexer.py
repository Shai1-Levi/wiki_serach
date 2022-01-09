
from inverted_index_colab import InvertedIndex as IIC, MultiFileReader as MRC
from inverted_index_gcp import InvertedIndex as IIG, MultiFileReader as MRG
from inverted_index_anchor import InvertedIndex as IIA, MultiFileReader as MRA
from contextlib import closing
import pandas as pd
from collections import Counter, OrderedDict
import requests
import re
import numpy as np
import pickle
from pprint import pprint
from pathlib import Path
from python_code.utils import Utils
# import pyspark
# from pyspark.sql import *
# from pyspark.sql.functions import *
# from pyspark import SparkContext, SparkConf
# from pyspark.sql import SQLContext
# from pyspark.ml.feature import Tokenizer, RegexTokenizer
# from graphframes import *


# graphframes_jar = 'https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar'
# spark_jars = '/usr/local/lib/python3.7/dist-packages/pyspark/jars'
# conf = SparkConf().set("spark.ui.port", "4050")
# try:
#   sc = pyspark.SparkContext(conf=conf)
# except:
#   sc.addPyFile(str(Path(spark_jars) / Path(graphframes_jar).name))
#   spark = SparkSession.builder.getOrCreate()

class Indexer:

    def __init__(self, index_path, utils, indexr_name="body"):
        self.inv_idx_path = index_path
        self.inv_idx_file = "index"
        if indexr_name == "body":
            self.inv_idx = IIG.read_index(self.inv_idx_path, self.inv_idx_file)
            self.nf = pd.read_pickle(index_path + "doc_body_length.pkl")
        elif indexr_name == "title":
            self.inv_idx = IIC.read_index(self.inv_idx_path, self.inv_idx_file)
        elif indexr_name == "anchor":
            self.inv_idx = IIA.read_index(self.inv_idx_path, self.inv_idx_file)
        self.utils = utils
        self.N = 6348910
        self.TUPLE_SIZE = 6
        self.TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
        # self.sc = sc
        # self.spark = spark
        # self.titles = pd.read_pickle("/content/gDrive/MyDrive/project/titles.pkl")
        self.titles = ""

    def read_posting_list_title(self, w):
        with closing(MRC(self.inv_idx_path)) as reader:
            locs = self.inv_idx.posting_locs[w]
            b = reader.read(locs, self.inv_idx.df[w] * self.TUPLE_SIZE)
            posting_list = []
            for i in range(self.inv_idx.df[w]):
                doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                # tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                # posting_list.append((doc_id, tf))
                posting_list.append(doc_id)
            return posting_list

    def read_posting_list_body(self, w):
        with closing(MRG(self.inv_idx_path)) as reader:
            locs = self.inv_idx.posting_locs[w]
            b = reader.read(locs, self.inv_idx.df[w] * self.TUPLE_SIZE)
            posting_list = []
            for i in range(self.inv_idx.df[w]):
                doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def read_posting_list_anchor(self, w):
        with closing(MRA(self.inv_idx_path)) as reader:
            locs = self.inv_idx.posting_locs[w]
            b = reader.read(locs, self.inv_idx.df[w] * self.TUPLE_SIZE)
            posting_list = []
            for i in range(self.inv_idx.df[w]):
                doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                # tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                # posting_list.append((doc_id, tf))
                posting_list.append(doc_id)
            return posting_list

    def get_binary_match_title(self, query_tokens, with_titles=True):
        relevent_docs = Counter()
        n = 1
        for term in set(query_tokens):
            for doc_id in self.read_posting_list_title(term):
                relevent_docs[doc_id] += 1
                n+=1
        if with_titles == False:
          return relevent_docs.most_common(n)
        sorted_ids, score  = zip(*relevent_docs.most_common(n))
        # print("title", sorted_ids)
        # return self.get_page_titles(sorted_ids)
        return self.utils.get_page_titles(sorted_ids)

    def get_binary_match_anchor(self, query_tokens, with_titles=True):
        relevent_docs = Counter()
        n = 1
        for term in set(query_tokens):
            for doc_id in self.read_posting_list_anchor(term):
                relevent_docs[doc_id] += 1
                n+=1

        if with_titles == False:
          return relevent_docs.most_common(n)
        sorted_ids, score  = zip(*relevent_docs.most_common(n))
        # print("anchor", sorted_ids)
        # return self.get_page_titles(sorted_ids)
        return self.utils.get_page_titles(sorted_ids)

    def tf_idf(self, term, word_freq, doc_id):
        idf = np.log2(self.N / self.inv_idx.df[term])
        try:
            tf = word_freq / self.nf[doc_id][0]
        except:
            tf = 0
            pass
        return tf * idf

    def get_cosine_sim(self, query, N, with_titles=True):
        """
        In this function you need to utilize the cosine_similarity function from sklearn.
        You need to compute the similarity between the queries and the given documents.
        This function will return a DataFrame in the following shape: (# of queries, # of documents).
        Each value in the DataFrame will represent the cosine_similarity between given query and document.

        Parameters:
        -----------
          queries: sparse matrix represent the queries after transformation of tfidfvectorizer.
          documents: sparse matrix represent the documents.

        Returns:
        --------
          DataFrame: This function will return a DataFrame in the following shape: (# of queries, # of documents).
          Each value in the DataFrame will represent the cosine_similarity between given query and document.
        """
        # YOUR CODE HERE
        # query_counter = Counter(query)
        # query_norm = np.linalg.norm(np.array(list(query_counter.values())))
        sim_dict = Counter()  # key: (query,doc_id) , val: norm
        for term in query:
            for doc_id, w2cnt in self.read_posting_list_body(term):
                sim_dict[doc_id] = sim_dict.get(doc_id, 0) + self.tf_idf(term, w2cnt, doc_id)  # W_tfidf_bm25
        # for doc_id, sim_dict_val in sim_dict.items():
        for doc_id in sim_dict.keys():
            # try:
            sim_dict[doc_id] = sim_dict[doc_id] * self.nf[doc_id][0] # * query_norm
            # except:
            #     pass
        if with_titles == False:
          return sim_dict.most_common(N)
        sorted_ids, score = zip(*sim_dict.most_common(N))
        # return  self.get_page_titles(sorted_ids)
        return self.utils.get_page_titles(sorted_ids)

    def get_page_titles(self, pages_ids): #################################################
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
        # files_name = [158361, 434000, 767804, 1165403, 1602318, 2063004, 2548159, 3089250, 3679721, 4351962, 5081862,
        #               5812519, 6598640, 7443732, 8400567, 9413712, 10537076, 11617344, 12468957, 13245200, 14310616,
        #               15321353, 16195550, 17430257, 18508742, 19291087, 20390809, 21386331, 22268217, 23242167, 24074227,
        #               25080812, 26068663, 27111995, 28159117, 29259581, 30499123, 31549598, 32643995, 33656354, 34761591,
        #               35918133, 36815038, 37900157, 38960238, 39974621, 40950632, 41924177, 42980673, 43938195, 45007416,
        #               46575031, 47584475, 48850115, 50136023, 51293440, 52515585, 53647429, 54827383, 56143023, 57230195,
        #               58301426, 59688406, 60952233, 62088946, 63344543, 64478166, 65916205, 67184267, 68380237]
        # pages_to_read = set()
        # mapped_page_id_to_files = {}
        # pages_ids['title'] = np.nan
        # for index, page_id in pages_ids.iterrows():
        #     for i in files_name:
        #         if i > page_id['id']:
        #             pages_to_read.add(i)
        #             if i in mapped_page_id_to_files.keys():
        #                 mapped_page_id_to_files[i].append(page_id['id'])
        #             else:
        #                 mapped_page_id_to_files[i] = [page_id['id']]
        #             break
        #
        # for page_to_read in pages_to_read:
        #     tmp_df = pd.DataFrame({"id": mapped_page_id_to_files[page_to_read]})
        #     with open("/content/gDrive/MyDrive/project/titles/{}.pkl".format(page_to_read), 'rb') as f:
        #         self.titles = pickle.loads(f.read())
        #     try:
        #         relevent_titles = pd.merge(self.titles, tmp_df, on="id")
        #
        #         pages_ids['title'].update(pages_ids['id'].map(relevent_titles.set_index('id')['title']))
        #
        #     except Exception as e:
        #         print(e)
        #
        # return pages_ids.to_dict(orient='split')["data"]

