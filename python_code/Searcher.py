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
        doc_ids_body = [x[0] for x in initial_lst_tup_body_cossine]
        doc_ids_title = [x[0] for x in initial_lst_tup_body_title]

        msi = self.merge_score_indexes(initial_lst_tup_body_cossine, initial_lst_tup_body_title)
        a = 0.7
        b = 1-a
        # combine between indexes score to page view score - take too much time about 8 seconds per query !!!!!!!!!!
        ids, scores = zip(*msi)
        page_rank_score = self.pr.get_page_rank_by_ids(ids)
        doc_id_with_page_view = list(zip(ids, page_rank_score))
        combine_pageview_indexes = [(x[0], a*x[1]+b*y[1]) for x,y in zip(msi, doc_id_with_page_view)]
        print(f"msi: {msi}")
        print(f"doc_id_with_page_view: {doc_id_with_page_view}")
        print(f"combine_pageview_indexes: {combine_pageview_indexes}")
        # print("merge_score", msi)



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

        inter = set(lst_body).intersection(set(lst_title))
        diff_body = set(lst_body).difference(inter)
        diff_title = set(lst_title).difference(inter)
        lst = []
        n_inter = len(inter)
        n_body  = len(diff_body)
        n_title= len(diff_title)
        for i in range(max(n_inter, n_body, n_title)):
            if i < n_inter:
                val = list(inter)[i]
                b_i = (1/(lst_body.index(val)+1))*0.7
                t_i = (1/(lst_title.index(val)+1))*0.3
                lst.append((val, b_i + t_i))
            if i < n_body:
                val = list(diff_body)[i]
                b_i = (1/(lst_body.index(val)+1))*0.7
                lst.append((val, b_i))
            if i < n_title:
                val = list(diff_title)[i]
                t_i = (1/(lst_title.index(val)+1))*0.3
                lst.append((val, t_i))
        return sorted(lst, key=lambda k: k[1], reverse=True)

