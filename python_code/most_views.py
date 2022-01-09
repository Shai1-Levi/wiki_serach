from collections import Counter
import pickle


class MostViews:
    def __init__(self, path):
        self.wid2pv = Counter()
        self.page_views_path = path + "page_Views.pkl"

        with open(self.page_views_path, 'rb') as f:
          self.mv_index = pickle.loads(f.read())

    def most_viewed(self, pages_ids):
        # """Rank pages viewed
        # Parameters:
        # -----------
        #   pages: An iterable list of pages ids
        # Returns:
        # --------
        # list of ints:
        #     list of page view numbers from August 2021 that correrspond to the
        #     provided list article IDs.
        # """
        lst = []
        for page_id in pages_ids:
            try:
                lst.append(self.mv_index[page_id])
            except Exception as e:
                lst.append(0)
                print(e)
        return lst
