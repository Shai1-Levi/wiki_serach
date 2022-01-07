from nltk.corpus import lin_thesaurus as thes
from collections import Counter
from python_code import Ranker
import numpy as np
import pandas as pd
import pickle


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
    def __init__(self):
        self.titles = pd.read_pickle("/content/gDrive/MyDrive/project/titles.pkl")

    def search(self, query, IDX, IDXT, IDXA, pr, mv, parser, bm25):
        # initial_lst_tup_body_cossine = IDX.get_cosine_sim(query, 500, with_titles=False)
        # initial_lst_tup_body_title = IDXT.get_binary_match_title(query, with_titles=False)
        # initial_lst_tup_body_anchor = IDXT.get_binary_match_anchor(query, with_titles=False)
        bm25_body, bm25_title, bm25_anchor = bm25[0], bm25[1], bm25[2]

        bm25_body_dict = bm25_body.search(query, 1000)
        bm25_title_dict = bm25_title.search(query, 1000)
        # bm25_anchor_dict = bm25_anchor.search(query, 500)

        # print(bm25_body_dict)
        # print(bm25_title_dict)

        msi, _ = zip(*self.merge_score(bm25_title_dict, bm25_body_dict, a=0.6, b=0.4))
        # msi1, _ = zip(*msi)
        # print("initial_lst_tup_body_cossine", initial_lst_tup_body_cossine)
        # print("initial_lst_tup_body_title", initial_lst_tup_body_title)
        # print("initial_lst_tup_body_anchor", initial_lst_tup_body_anchor)
        # msi = self.merge_score(initial_lst_tup_body_cossine, initial_lst_tup_body_title, a=0.6, b=0.4)
        # msi = self.merge_results(bm25_title_dict, bm25_body_dict, a=0.4, b=0.6)
        # msi,_ = zip(*self.merge_score(msi, bm25_anchor_dict, a=0.4, b=0.6))

        return self.get_page_titles(msi)
        # return self.merge_score(msi, initial_lst_tup_body_anchor, a=0.8, b=0.2)
        # return self.merge_score_pr_or_mv(msi, pr, a=0.8,b=0.2)

    def merge_score(self, lst_tup_body, lst_tup_title, a, b):
        counter = Counter()
        for kt, vs in lst_tup_title:
            counter[kt] = vs * a
        for kb, vb in lst_tup_body:
            counter[kb] += vb * b
        return counter.most_common(100)

    def get_page_titles(self, pages_ids):
        ''' Returns the title of the first, fourth, and fifth pages as ranked about
          by PageRank.
            Returns:
            --------
              list of three strings.
          '''
        lst = []
        for page_id in pages_ids:
            lst.append((page_id, self.titles[page_id]))
        return lst

        files_name = [158361, 434000, 767804, 1165403, 1602318, 2063004, 2548159, 3089250, 3679721, 4351962, 5081862,
                      5812519, 6598640, 7443732, 8400567, 9413712, 10537076, 11617344, 12468957, 13245200, 14310616,
                      15321353, 16195550, 17430257, 18508742, 19291087, 20390809, 21386331, 22268217, 23242167,
                      24074227,
                      25080812, 26068663, 27111995, 28159117, 29259581, 30499123, 31549598, 32643995, 33656354,
                      34761591,
                      35918133, 36815038, 37900157, 38960238, 39974621, 40950632, 41924177, 42980673, 43938195,
                      45007416,
                      46575031, 47584475, 48850115, 50136023, 51293440, 52515585, 53647429, 54827383, 56143023,
                      57230195,
                      58301426, 59688406, 60952233, 62088946, 63344543, 64478166, 65916205, 67184267, 68380237]
        pages_to_read = set()
        mapped_page_id_to_files = {}
        pages_ids['title'] = np.nan
        for index, page_id in pages_ids.iterrows():
            for i in files_name:
                if i > page_id['id']:
                    pages_to_read.add(i)
                    if i in mapped_page_id_to_files.keys():
                        mapped_page_id_to_files[i].append(page_id['id'])
                    else:
                        mapped_page_id_to_files[i] = [page_id['id']]
                    break

        for page_to_read in pages_to_read:
            tmp_df = pd.DataFrame({"id": mapped_page_id_to_files[page_to_read]})
            self.titles = pd.read_pickle("/content/gDrive/MyDrive/project/titles/{}.pkl".format(page_to_read))
            # with open("/content/gDrive/MyDrive/project/titles/{}.pkl".format(page_to_read), 'rb') as f:
            #     self.titles = pickle.loads(f.read())
            try:
                relevent_titles = pd.merge(self.titles, tmp_df, on="id")

                pages_ids['title'].update(pages_ids['id'].map(relevent_titles.set_index('id')['title']))

            except Exception as e:
                print(e)

        return pages_ids.to_dict(orient='split')["data"]

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

        # lst = []
        # n_inter = len(inter)
        # n_body  = len(diff_body)
        # n_title= len(diff_title)
        # for i in range(max(n_inter, n_body, n_title)):
        #     if i < n_inter:
        #         val = list(inter)[i]
        #         b_i = (1/(lst_body.index(val)+1))*0.7
        #         t_i = (1/(lst_title.index(val)+1))*0.3
        #         lst.append((val, b_i + t_i))
        #     if i < n_body:
        #         val = list(diff_body)[i]
        #         b_i = (1/(lst_body.index(val)+1))*0.7
        #         lst.append((val, b_i))
        #     if i < n_title:
        #         val = list(diff_title)[i]
        #         t_i = (1/(lst_title.index(val)+1))*0.3
        #         lst.append((val, t_i))
        # return sorted(lst, key=lambda k: k[1], reverse=True)

    # def merge_results(title_scores,body_scores,title_weight=0.8,text_weight=0.2,N = 100):
    #     merge_results_dict = {}
    #     for i in title_scores.keys():
    #       dict_scores = dict([(k,v*title_weight) for k,v in title_scores[i]])
    #       for k,v in body_scores[i]:
    #         if k in dict_scores.keys():
    #           dict_scores[k] += v*text_weight
    #         else:
    #           dict_scores[k] = v*text_weight
    #       lst = list(dict_scores.items())
    #       merge_results_dict[i] = [(k,v) for k, v in sorted(lst, key=lambda item: item[1], reverse=True)][:N]
    #     return merge_results_dict

    def expand_query(self, query):
        neq_query = []
        for term in query:
            try:
                thes_term = list(thes.synonyms(term, fileid="simN.lsp"))[0]
                if term in self.IDX.inv_idx.df.keys():
                    neq_query.append(thes_term)
                neq_query.append((term))
            except Exception as e:
                neq_query.append(term)
        return self.parser.filter_tokens(tokens=neq_query, tokens2remove=self.parser.stop)


