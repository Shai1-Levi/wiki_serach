from python_code.Parser import Parser
from python_code.indexer import Indexer
part_url = 'C:\\Users\\User\\Documents\\FourthYear\\Semster_A\\Information_Retrieval\\proj_wiki\\enwiki-20211220-pages-articles-multistream1.xml-p1p41242.bz2'


# part_url = 'https://dumps.wikimedia.org/enwiki/20211220/enwiki-20211220-pages-articles-multistream1.xml-p1p41242.bz2'
# wiki_file = Path(part_url).name
# wiki_file_down = wget.download(part_url)

# df = pd.read_pickle(r"C:\Users\User\Documents\FourthYear\Semster_A\Information_Retrieval\wiki_search\postings_gcp_0_posting_locs.pickle")
# print(df)

parser = Parser("")
query = 'Does pasta have preservatives?'

tokens = parser.tokenize(query.lower())
print(tokens)
# indx = InvertedIndex.read_index(base_dir=r"C:\Users\User\Documents\FourthYear\Semster_A\Information_Retrieval\wiki_search\postings_gcp\postings_gcp_index.pkl", name="postings_gcp_index")
indx = Indexer()
print(indx.read_posting_list("python"))


# parser = Parser(part_url)
# iter = parser.page_iter()
# pprint(next(iter)[2])
# print(parser.wiki_file)
# pages = list(islice(parser.page_iter(), None, 25))
# # get_wl = lambda text: parser.get_wikilinks(mwp.parse(text))
# tokens_no_stemming = 0
# tokens_to_stemming = 0
# only_stopwords = 0
# for page in pages:
#     tokens = parser.tok(page[2])
#     # tokens_no_stemming += len(tokens)
#     # tokens_to_stemming += len(parser.filter_tokens(tokens, tokens2remove=parser.en_stopwords, use_stemming=True))
#     only_stopwords += len(parser.filter_tokens(tokens, tokens2remove=parser.en_stopwords, use_stemming=False))
# parser.most_viewed(tokens, wid2pv)
# print(len(tokens))
# print("under" in parser.en_stopwords)
# print(len(parser.filter_tokens(tokens, tokens2remove=parser.en_stopwords, use_stemming=True)))
# print("tokens_no_stemming : 177429")
# print("only_stopwords : 123782")
# print("tokens_to_stemming : 123782")