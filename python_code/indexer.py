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
from python_code.Utils import Utils

class Indexer:

    def __init__(self, index_path, utils, indexr_name="body"):
        self.inv_idx_path = index_path
        self.inv_idx_file = "index"
        self.utils = utils
        if indexr_name == "body":
            self.inv_idx = IIG.read_index(self.inv_idx_path, self.inv_idx_file)
            self.nf = self.utils.get_nf()  # pd.read_pickle(index_path + "doc_body_length.pkl")
        elif indexr_name == "title":
            self.inv_idx = IIC.read_index(self.inv_idx_path, self.inv_idx_file)
        elif indexr_name == "anchor":
            self.inv_idx = IIA.read_index(self.inv_idx_path, self.inv_idx_file)

        self.N = 6348910
        self.TUPLE_SIZE = 6
        self.TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
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
                n += 1
        if with_titles == False:
            return relevent_docs.most_common(n)
        sorted_ids, score = zip(*relevent_docs.most_common(n))
        return self.utils.get_page_titles(sorted_ids)

    def get_binary_match_anchor(self, query_tokens, with_titles=True):
        relevent_docs = Counter()
        n = 1
        for term in set(query_tokens):
            for doc_id in self.read_posting_list_anchor(term):
                relevent_docs[doc_id] += 1
                n += 1

        if with_titles == False:
            return relevent_docs.most_common(n)
        sorted_ids, score = zip(*relevent_docs.most_common(n))
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
        query_counter = Counter(query)
        query_norm = np.linalg.norm(np.array(list(query_counter.values())))
        sim_dict = Counter()  # key: (query,doc_id) , val: norm
        for term in query:
            for doc_id, w2cnt in self.read_posting_list_body(term):
                sim_dict[doc_id] = sim_dict.get(doc_id, 0) + self.tf_idf(term, w2cnt, doc_id)  # W_tfidf_bm25
        for doc_id in sim_dict.keys():
            try:
                sim_dict[doc_id] = sim_dict[doc_id] * np.power(self.nf[doc_id][0], 2) * query_norm
            except:
                pass
        if with_titles == False:
            return sim_dict.most_common(N)
        sorted_ids, score = zip(*sim_dict.most_common(N))
        return self.utils.get_page_titles(sorted_ids)