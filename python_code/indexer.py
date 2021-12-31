from inverted_index_colab import InvertedIndex as IIC, MultiFileReader as MRC
from inverted_index_gcp import InvertedIndex as IIG, MultiFileReader as MRG
from contextlib import closing
import pandas as pd
from collections import Counter
import requests
import re
import numpy as np
import pickle


class Indexer:

    def __init__(self, index_path, indexr_name="body"):
        self.inv_idx_path = index_path
        self.inv_idx_file = "index"
        if indexr_name == "body":
            self.inv_idx = IIG.read_index(self.inv_idx_path, self.inv_idx_file)
            self.nf = pd.read_pickle("/content/gDrive/MyDrive/project/doc_body_length.pkl")
        elif indexr_name == "title":
            self.inv_idx = IIC.read_index(self.inv_idx_path, self.inv_idx_file)
        elif indexr_name == "anchor":
            s = 5
        self.N = 6348910
        self.TUPLE_SIZE = 6
        self.TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

    def read_posting_list_title(self, w):
        with closing(MRC()) as reader:
            locs = self.inv_idx.posting_locs[w]
            # locs = [('postings_gcp_82_015.bin', 1619898)]
            b = reader.read(locs, self.inv_idx.df[w] * self.TUPLE_SIZE)
            posting_list = []
            for i in range(self.inv_idx.df[w]):
                doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                # posting_list.append((doc_id, self.get_page_titles(doc_id)))
                posting_list.append((doc_id, tf))
            return posting_list

    def read_posting_list_body(self, w):
        with closing(MRG()) as reader:
            locs = self.inv_idx.posting_locs[w]
            # locs = [('postings_gcp_82_015.bin', 1619898)]
            b = reader.read(locs, self.inv_idx.df[w] * self.TUPLE_SIZE)
            posting_list = []
            for i in range(self.inv_idx.df[w]):
                doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                # posting_list.append((doc_id, self.get_page_titles(doc_id)))
                posting_list.append((doc_id, tf))
            return posting_list

    def get_binary_match(self, query_tokens):
        relevent_docs = Counter()
        for term in set(query_tokens):
            for doc_id, vals in self.read_posting_list_title(term):
                relevent_docs[doc_id] += 1
        return sorted(relevent_docs.items(), key=lambda x: x[1], reverse=True)

    def tf_idf(self, term, word_freq, doc_id):
        # for term in query:
        idf = np.log2(self.N / self.inv_idx.df[term])
        try:
            tf = word_freq / self.nf[doc_id][0]
        except:
            tf = 0
            pass
        return tf * idf

    def get_cosine_sim(self, query, N):
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
        # query = parser.filter_tokens(parser.tokenize(query), parser.en_stopwords)
        query_counter = Counter(query)
        query_norm = np.linalg.norm(np.array(list(query_counter.values())))

        sim_dict = {}  # key: (query,doc_id) , val: norm
        for term in query:
            # It[term]+= IDX.read_posting_list(term)
            # for doc_id, w2cnt in
            for doc_id, w2cnt in self.read_posting_list_body(term):
                sim_dict[doc_id] = sim_dict.get(doc_id, 0) + 1 * self.tf_idf(term, w2cnt, doc_id)  # W_tfidf_bm25
        # print(sim_dict)
        for doc_id, sim_dict_val in sim_dict.items():
            # nf[doc_id]
            sim_dict[doc_id]
            try:
                sim_dict[doc_id] = sim_dict[doc_id] * query_norm * self.nf[doc_id][0]
            except:
                pass
        # print(sim_dict)
        return sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)[:N]

    def get_page_titles(self, pages_ids):
        ''' Returns the title of the first, fourth, and fifth pages as ranked about
          by PageRank.
            Returns:
            --------
              list of three strings.
          '''
        pages_to_read = set()
        mapped_page_id_to_files = {}
        # print(pages_ids)
        for page_id in pages_ids:
            for i in range(200000, 68800000, 200000):
                if i > page_id:
                    pages_to_read.add(i)
                    if i in mapped_page_id_to_files.keys():
                        mapped_page_id_to_files[i].append(page_id)
                    else:
                        mapped_page_id_to_files[i] = [page_id]
                    break

        for page_to_read in pages_to_read:
            with open("wid2pv/{}.pkl".format(page_to_read), 'rb') as f:
                self.wid2pv = pickle.loads(f.read())
            for page_id in mapped_page_id_to_files[page_to_read]:
                try:
                    pages_ids[pages_ids.index(page_id)] = self.wid2pv[page_id][1]
                except:
                    print("page_id", page_id)
                    print("pages_ids.index(page_id)", pages_ids.index(page_id))
                    print("self.wid2pv[page_id]", self.wid2pv[page_id])
                    print("self.wid2pv[page_id][1]", self.wid2pv[page_id][1])

        return pages_ids