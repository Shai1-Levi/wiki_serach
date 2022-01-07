from nltk.corpus import lin_thesaurus as thes
from collections import Counter
from python_code import Ranker


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
        self.s = 5

    def search(self, query, IDX, IDXT, IDXA, pr, mv, parser, bm25):
        initial_lst_tup_body_cossine = IDX.get_cosine_sim(query, 500, with_titles=False)
        initial_lst_tup_body_title = IDXT.get_binary_match_title(query, with_titles=False)
        initial_lst_tup_body_anchor = IDXT.get_binary_match_anchor(query, with_titles=False)
        bm25_body, bm25_title, bm25_anchor = bm25[0], bm25[1], bm25[2]

        bm25_body_dict = bm25_body.search(query, 100)
        bm25_title_dict = bm25_title.search(query, 100)
        bm25_anchor_dict = bm25_anchor.search(query, 100)

        # print(bm25_body_dict)
        # print(bm25_title_dict)

        msi = self.merge_score(bm25_title_dict, bm25_body_dict, a=0.6, b=0.4)
        # print("initial_lst_tup_body_cossine", initial_lst_tup_body_cossine)
        # print("initial_lst_tup_body_title", initial_lst_tup_body_title)
        # print("initial_lst_tup_body_anchor", initial_lst_tup_body_anchor)
        # msi = self.merge_score(initial_lst_tup_body_cossine, initial_lst_tup_body_title, a=0.6, b=0.4)
        # msi = self.merge_results(bm25_title_dict, bm25_body_dict, a=0.4, b=0.6)
        # msi = self.merge_score(msi, bm25_anchor_dict[0], a=0.25, b=0.75)
        return msi
        # return self.merge_score(msi, initial_lst_tup_body_anchor, a=0.8, b=0.2)
        # return self.merge_score_pr_or_mv(msi, pr, a=0.8,b=0.2)

    def merge_score(self, lst_tup_body, lst_tup_title, a, b):
        counter = Counter()
        for kt, vs in lst_tup_title:
            counter[kt] = vs * a
        for kb, vb in lst_tup_body:
            counter[kb] += vb * b
        return counter.most_common(100)

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


