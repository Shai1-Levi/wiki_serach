from collections import Counter
import pickle


class MostViews:
    def __init__(self, path="gDrive/MyDrive/project/page_views"):
        self.wid2pv = Counter()
        self.page_views_path = path

    def most_viewed(self, pages_ids):
        """Rank pages viewed
      Parameters:
      -----------
        pages: An iterable list of pages ids
      Returns:
      --------
      list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
      """
        pages_to_read = set()
        mapped_page_id_to_files = {}
        for page_id in pages_ids:
            for i in range(200000, 68800000, 200000):
                if i > page_id:
                    pages_to_read.add(i)
                    if i in mapped_page_id_to_files.keys():
                        mapped_page_id_to_files[i].append(page_id)
                    else:
                        mapped_page_id_to_files[i] = [page_id]
                    break

        for page_to_read in pages_to_read:
            with open("{}/{}.pkl".format(self.page_views_path, page_to_read), 'rb') as f:
                self.wid2pv = pickle.loads(f.read())
            for page_id in mapped_page_id_to_files[page_to_read]:
                pages_ids[pages_ids.index(page_id)] = self.wid2pv[page_id]

        return pages_ids