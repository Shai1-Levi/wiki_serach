import nltk
from nltk.corpus import lin_thesaurus as thes
from nltk.corpus import wordnet as wn
from collections import Counter
from python_code import Ranker
import numpy as np
import pandas as pd
import pickle

nltk.download('stopwords')
nltk.download('wordnet')


class Searcher:
    # def __init__(self, IDX, IDXT, IDXA, pr, mv, parser):
    # self.IDX = IDX
    # self.IDXT = IDXT
    # self.IDXA = IDXA
    # self.pr = pr
    # self.mv = mv
    # self.parser = parser
    # super(Searcher, self).__init__(IDX,IDXT, IDXA) # *args, **kwargs)
    # self.__dict__ = self
    def __init__(self, utils):
        self.utils = utils

    def search(self, query, IDX, IDXT, IDXA, pr, mv, parser, bm25):
        # query = self.expand_query(query, IDXT)
        # initial_lst_tup_body_cossine = IDX.get_cosine_sim(query, 100, with_titles=False)
        # initial_lst_tup_body_title = IDXT.get_binary_match_title(query, with_titles=False)
        # initial_lst_tup_body_anchor = IDXT.get_binary_match_anchor(query, with_titles=False)
        bm25_body, bm25_title, bm25_anchor = bm25[0], bm25[1], bm25[2]
        bm25_body_dict = bm25_body.search(query, 1000)
        bm25_title_dict = bm25_title.search(query, 1000)
        bm25_anchor_dict = bm25_anchor.search(query, 1000)

        # print(bm25_body_dict)
        # print(bm25_title_dict)

        # msi, _ = zip(*self.merge_score(bm25_title_dict, bm25_body_dict, a=0.6, b=0.4))
        # msi1, _ = zip(*msi)
        # print("initial_lst_tup_body_cossine", initial_lst_tup_body_cossine)
        # print("initial_lst_tup_body_title", initial_lst_tup_body_title)
        # print("initial_lst_tup_body_anchor", initial_lst_tup_body_anchor)
        # msi = self.merge_score(initial_lst_tup_body_cossine, initial_lst_tup_body_title, a=0.6, b=0.4)
        msi = self.merge_score(bm25_title_dict, bm25_body_dict, a=0.9, b=0.1, N = 100)
        msi = self.merge_score(msi, bm25_anchor_dict, a=0.2, b=0.8, N = 100)
        # msi = self.merge_score(msi, initial_lst_tup_body_cossine, a=0.2, b=0.8, N = 100)
        # msi, _ = zip(*bm25_body_dict)
        return msi
        return self.utils.get_page_titles(msi)
        # return self.merge_score(msi, initial_lst_tup_body_anchor, a=0.8, b=0.2)
        # return self.merge_score_pr_or_mv(msi, pr, a=0.8,b=0.2)

    def merge_score(self, lst_tup_body, lst_tup_title, a, b, N):
        counter = Counter()
        for kt, vs in lst_tup_title:
            counter[kt] = vs * a
        for kb, vb in lst_tup_body:
            counter[kb] += vb * b
        if N == None:
          N = len(counter)
        return counter.most_common(N)



    def merge_score_pr_or_mv(self, pr_or_mv, lst_tup, a, b):
        counter = Counter()
        lst = []
        for kt, vs in lst_tup:
            counter[kt] = vs * b
            lst.append(kt)
        print(lst)
        lst2 = Ranker.get_page_rank_by_ids(lst)
        print(lst)
        for index in range(len(lst2)):
            counter[lst_tup[index][0]] += lst2[index] * a
        return counter.most_common(100)

    # def expand_query(self, query):
    #   # counter = 0
    #   neq_query = []
    #   for term in query:
    #       try:
    #           for thes_term in list(thes.synonyms(term, fileid="simN.lsp")):
    #               if thes_term not in self.parser.stop and thes_term in self.IDX.inv_idx.df.keys():
    #                   neq_query.append(thes_term)
    #           neq_query.append(term)
    #       except Exception as e:
    #           neq_query.append(term)
    #   # return self.parser.filter_tokens(tokens=neq_query, tokens2remove=self.parser.stop)
    #   return neq_query

    def expand_query(self, query, IDXT):
      synonyms = set()
      hyponyms = set()
      # antonyms = set()
      s_count = 0
      # a_count = 0
      for term in query:
        for syn in wn.synsets(term):
            if s_count == 2:
                break
            for h in syn.hyponyms():
              if len(h.lemma_names()[0]) >1 and h.lemma_names()[0] in IDXT.inv_idx.df.keys() :
              # print(h.lemma_names())
              # print("h", h)
                hyponyms.add(h.lemma_names()[0])
            for l in syn.lemmas():
                if s_count < 5:
                    # print("l ", l.name().lower())
                    if l.name().lower() in IDXT.inv_idx.df.keys():
                      if len(l.name()[0]) > 1:
                        synonyms.add(l.name()[0])
                        s_count += 1
                # if l.antonyms():
                #     if a_count < 1:
                #         if l.antonyms()[0].name().lower() not in []:
                #             antonyms.add(l.antonyms()[0].name())
                #             a_count += 1
        # print(synonyms.union(hyponyms.union(antonyms)))
      return query + list(synonyms.union(hyponyms))

