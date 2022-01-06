from nltk.corpus import lin_thesaurus as thes


class Searcher:
    def __init__(self, IDX, IDXT, IDXA, pr, mv, parser):
        self.IDX = IDX
        self.IDXT = IDXT
        self.IDXA = IDXA
        self.pr = pr
        self.mv = mv
        self.parser = parser

    def search(self, query):
        initial_lst_tup_body_cossine = self.IDX.get_cosine_sim(query, 500, with_titles=False)
        initial_lst_tup_body_title = self.IDXT.get_binary_match_title(query, with_titles=False)
        initial_lst_tup_body_anchor = self.IDXT.get_binary_match_anchor(query, with_titles=False)
        print(initial_lst_tup_body_cossine)
        print(initial_lst_tup_body_title)
        print(initial_lst_tup_body_anchor)
        # doc_ids_body = [x[0] for x in initial_lst_tup_body_cossine]
        # doc_ids_title = [x[0] for x in initial_lst_tup_body_title]

        # msi = self.merge_score_indexes(initial_lst_tup_body_cossine, initial_lst_tup_body_title)
        msi = self.merge_results(initial_lst_tup_body_cossine, initial_lst_tup_body_title, text_weight=0.8,
                                 title_weight=0.2, N=50)
        # msi_with_anchor = self.merge_results(msi ,initial_lst_tup_body_anchor, text_weight=0.4 ,title_weight =0.6, N = 50)

        # combine between indexes score to page rank score - take too much time about 8 seconds per query !!!!!!!!!!
        # page_rank_score = self.pr.get_page_rank_by_ids(ids)
        # doc_id_with_page_rank = list(zip(ids, page_rank_score))
        # print("with page rank")
        # combine_pagerank_indexes = [(x[0], a*x[1]+b*y[1]) for x,y in zip(msi, doc_id_with_page_rank)]
        # print(f"msi: {msi}")
        # print(f"doc_id_with_page_rank: {doc_id_with_page_rank}")
        # print(f"combine_pagerank_indexes: {combine_pagerank_indexes}")
        # print("merge_score", msi)

        # combine between indexes score to page view score - take too much time about 8 seconds per query !!!!!!!!!!
        # page_view_score = self.mv.most_viewed(ids)
        # print(f"page_view_score: {page_view_score}")

        # query = self.expand_query(query)
        # initial_files = list(self.IDX.get_cosine_sim(query, 500, with_titles=False))
        # initial_files_that_popular = self.mv.most_viewed(initial_files)
        # initial_files_that_popular_sorted = sorted(initial_files_that_popular, key=lambda x: x)
        # initial_files_that_popular_pr_sorted = sorted(self.pr.get_page_rank_by_ids(initial_files_that_popular_sorted),
        #                                               key=lambda x: x)
        return msi

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

    def merge_score_indexes(self, lst_tup_body, lst_tup_title):
        zip_lst_body = list(zip(*lst_tup_body))
        zip_lst_title = list(zip(*lst_tup_title))
        lst_body = [t[0] for t in zip_lst_body]
        lst_title = [t[0] for t in zip_lst_title]
        a = 0.2
        b = 1 - a
        inter = set(lst_body).intersection(set(lst_title))
        diff_body = set(lst_body).difference(inter)
        diff_title = set(lst_title).difference(inter)
        lst = []
        n_inter = len(inter)
        n_body = len(diff_body)
        n_title = len(diff_title)
        for i in range(max(n_inter, n_body, n_title)):
            if i < n_inter:
                val = list(inter)[i]
                b_i = (1 / (lst_body.index(val) + 1)) * a
                t_i = (1 / (lst_title.index(val) + 1)) * b
                lst.append((val, b_i + t_i))
            if i < n_body:
                val = list(diff_body)[i]
                b_i = (1 / (lst_body.index(val) + 1)) * a
                lst.append((val, b_i))
            if i < n_title:
                val = list(diff_title)[i]
                t_i = (1 / (lst_title.index(val) + 1)) * b
                lst.append((val, t_i))
        return sorted(lst, key=lambda k: k[1], reverse=True)

    def merge_results(body_scores, title_scores, text_weight=0.5, title_weight=0.5, N=50):
        """
        This function merge and sort documents retrieved by its weighte score (e.g., title and body).

        Parameters:
        -----------
        title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)

        body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                                key: query_id
                                                                                value: list of pairs in the following format:(doc_id,score)
        title_weight: float, for weigted average utilizing title and body scores
        text_weight: float, for weigted average utilizing title and body scores
        N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        dictionary of querires and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id,score).
        """
        body_dict = dict(body_scores)
        title_dict = dict(title_scores)
        merge_results_dict = {}
        for i in title_scores.keys():
            dict_scores = dict([(k, v * title_weight) for k, v in title_scores[i]])
            for k, v in body_scores[i]:
                if k in dict_scores.keys():
                    dict_scores[k] += v * text_weight
                else:
                    dict_scores[k] = v * text_weight
            lst = list(dict_scores.items())
            merge_results_dict[i] = [(k, v) for k, v in sorted(lst, key=lambda item: item[1], reverse=True)][:N]

