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
                lst.append((page_id, "Not found"))
                pass
        return lst

    # return dictionary of normalized len document per document id
    def get_nf(self):
        return self.nf
