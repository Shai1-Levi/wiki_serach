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
        query = self.expand_query(query)
        initial_files = list(self.IDX.get_cosine_sim(query, 500, with_titles=False))
        initial_files_that_popular = self.mv.most_viewed(initial_files)
        initial_files_that_popular_sorted = sorted(initial_files_that_popular, key=lambda x: x)
        initial_files_that_popular_pr_sorted = sorted(self.pr.get_page_rank_by_ids(initial_files_that_popular_sorted),
                                                      key=lambda x: x)
        return initial_files_that_popular_pr_sorted

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
