import csv
from collections import Counter
import pandas as pd


class Ranker:
    def __init__(self, path):
        self.page_rank_df = pd.read_csv(path, compression='gzip', header=None).rename({0: 'doc_id', 1: 'score'}, axis=1)
        self.page_rank_df.reset_index(drop=True, inplace=True)
        self.page_rank_df = self.page_rank_df.to_dict('split')
        self.page_rank_df = dict(self.page_rank_df['data'])

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