import math
from itertools import chain, count
import time
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from contextlib import closing
from collections import Counter, defaultdict
from python_code.Utils import Utils

TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

# DL = {}  # We're going to update and calculate this after each document. This will be usefull for the calculation of AVGDL (utilized in BM25)
# nf = Utils.get_nf("~/resources/")  #pd.read_pickle("/content/gDrive/MyDrive/project/doc_body_length.pkl")
DL = {}

# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self, path):
        self._open_files = {}
        self.xpath = path

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                # x = "/content/gDrive/MyDrive/project/postings_gcp/"
                self._open_files[f_name] = open(self.xpath + f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            if n_read < -1:
                n_read = -1
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        # DL[(doc_id)] = DL.get(doc_id, 0) + (nf[doc_id][1])
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        max_value = max(w2cnt.items(), key=operator.itemgetter(1))[1]
        # frequencies = {key: value/max_value for key, value in frequencies.items()}
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write(self, base_dir, name):
        """ Write the in-memory index to disk and populate the `posting_locs`
            variables with information about file location and offset of posting
            lists. Results in at least two files:
            (1) posting files `name`XXX.bin containing the posting lists.
            (2) `name`.pkl containing the global term stats (e.g. df).
        """
        #### POSTINGS ####
        self.posting_locs = defaultdict(list)
        with closing(MultiFileWriter(base_dir, name)) as writer:
            # iterate over posting lists in lexicographic order
            for w in sorted(self._posting_list.keys()):
                self._write_a_posting_list(w, writer, sort=True)
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def _write_a_posting_list(self, w, writer, sort=False):
        # sort the posting list by doc_id
        pl = self._posting_list[w]
        if sort:
            pl = sorted(pl, key=itemgetter(0))
        # convert to bytes
        b = b''.join([(int(doc_id) << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                      for doc_id, tf in pl])
        # write to file(s)
        locs = writer.write(b)
        # save file locations to index
        self.posting_locs[w].extend(locs)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                # read a certain number of bytes into variable b
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                # convert the bytes read into `b` to a proper posting list.

                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))

                yield w, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, path, utils, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.nf = utils.get_nf()
        self.N = len(self.nf)
        self.AVGDL = sum(d_n for d_norm, d_n in self.nf.values()) / self.N
        self.path = path
        # self.AVGDL = sum(DL.values()) / self.N
        # self.words, self.pls = zip(*self.index.posting_lists_iter(who_am_i='BM25'))

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, queries, N=3):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                  key: query_id
                                  value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        scores = Counter()
        # queries= queries.split(" ")
        self.idf = self.calc_idf(list(set(queries)))
        # print("self.idf", self.idf)
        # key = query_id, val = list of tuple(candidate_doc_id, bm25value)
        candidate_documents_and_tf = self.get_candidate_documents(queries, self.index)  # , self.words, self.pls)
        for query_index, query in enumerate(queries):
            # candidate_documents_and_tf = self.get_candidate_documents(query, self.index)#, self.words, self.pls)
            # print("candidate_documents", candidate_documents)
            # self.idf = self.calc_idf(query)
            # scores[query_index] = get_top_n(dict([(doc_id, self._score(query, doc_id)) for doc_id in candidate_documents]),N)
            scores_lst = [(doc_id, self._score([query], doc_id, tf)) for doc_id, tf in
                          candidate_documents_and_tf[query]]
            # print("scores_lst", scores_lst)
            for k, bm_v in scores_lst:
                scores[k] += bm_v
            # scores[query_index] = sorted(scores_lst, key=lambda x: x[1], reverse=True)[:N]
        return scores.most_common(N)

    # key term : val [(doc_id, bm25score)]

    def _score(self, query, doc_id, freq):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        if doc_id in self.nf.keys():
            doc_len = self.nf[doc_id][1]
        else:
            doc_len = 0
            # print("doc_len", doc_len)
        # print("query", query)
        # print("self.index.df.keys()", self.index.df.keys())
        for term in query:
            if term in self.index.df.keys():
                # print("HERE")
                # term_frequencies = dict(self.pls[self.words.index(term)])
                # term_frequencies = dict(self.read_posting_list_body(term))
                # print("term_frequencies", term_frequencies)
                # if doc_id in term_frequencies.keys():
                # freq = term_frequencies[doc_id]
                # print("freq", freq)
                # print("self.idf[term]", self.idf[term])
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score += (numerator / denominator)
        return score

    def read_posting_list(self, w):
        with closing(MultiFileReader(self.path)) as reader:
            locs = self.index.posting_locs[w]
            b = reader.read(locs, self.index.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def get_candidate_documents(self, query_to_search, index):  # , words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance
                         (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->
                           ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        list of candidates. In the following format:
                                                    key: pair (doc_id,term)
                                                    value: tfidf score.
        """
        candidates = {}
        for term in np.unique(query_to_search):
            candidates[term] = self.read_posting_list(term)
            # candidates.update(dict(self.read_posting_list_body(term)))
            # if term in words:
            #     current_list = (pls[words.index(term)])
            #     candidates += current_list
        return candidates

        # return np.unique(candidates)
