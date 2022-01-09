import csv
from collections import Counter
import pandas as pd


class Ranker:
    def __init__(self, path):
        self.page_rank_df = pd.read_csv(path, compression='gzip', header=None).rename({0: 'doc_id', 1: 'score'}, axis=1)
        self.page_rank_df.reset_index(drop=True, inplace=True)
        self.page_rank_df = self.page_rank_df.to_dict('split')
        self.page_rank_df = dict(self.page_rank_df['data'])
        # self.l = None

    def read_page_rank(self, path):
        self.page_rank_df = pd.read_csv(path)



    def get_page_rank_by_ids(self, page_ids):
        lst = []
        for page_id in page_ids:
            try:
                lst.append(self.page_rank_df[page_id])
            except:
                lst.append(0)
                pass
        return lst


        relevence = self.page_rank_df[self.page_rank_df['doc_id'].isin(page_ids)]
        outpot_lst = [-1] * len(page_ids)
        for index, row in relevence.iterrows():
            outpot_lst[page_ids.index(row['doc_id'])] = row['score']
        for i in range(len(outpot_lst)):
            if outpot_lst[i] == -1:
                outpot_lst[i] = 0
        # self.l = outpot_lst
        return outpot_lst

    def get_binary_match(self, query_tokens):
        relevent_docs = Counter()
        for term in set(query_tokens):
            for doc_id, vals in IDX.read_posting_list(term):
                relevent_docs[doc_id] += 1
        return sorted(relevent_docs.items(), key=lambda x: x[1], reverse=True)
