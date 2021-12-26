from python_code.inverted_index_gcp import InvertedIndex, MultiFileReader
from contextlib import closing
import requests
import re
class Indexer:

  def __init__(self, index_path):
    self.inv_idx_path = index_path
    self.inv_idx_file = "index"
    self.inv_idx = InvertedIndex.read_index(self.inv_idx_path, self.inv_idx_file)
    self.TUPLE_SIZE = 6
    self.TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

  def read_posting_list(self, w):
    with closing(MultiFileReader()) as reader:
      locs = self.inv_idx.posting_locs[w]
      # locs = [('postings_gcp_82_015.bin', 1619898)]
      b = reader.read(locs, self.inv_idx.df[w] * self.TUPLE_SIZE)
      posting_list = []
      for i in range(self.inv_idx.df[w]):
        doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
        tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
        # posting_list.append((doc_id, self.get_page_titles(doc_id)))
        posting_list.append((doc_id,tf))
      return posting_list


  def get_page_titles(self, page_id):
    ''' Returns the title of the first, fourth, and fifth pages as ranked about
      by PageRank.
        Returns:
        --------
          list of three strings.
      '''
    # YOUR CODE HERE
    # first = pr.collect()[0].id
    # fourth = pr.collect()[3].id
    # fifth = pr.collect()[4].id

    # pages = [first, fourth, fifth]
    # top_pr_pages = []

    # for page_id in pages:
    url = ' https://en.wikipedia.org/?curid=' + str(page_id)
    f = requests.get(url)
    title = re.findall(r"<title.*?>(.+?)</title>", f.text)[0]
    # top_pr_pages.append(title[0:-12])
    return title[0:-12]