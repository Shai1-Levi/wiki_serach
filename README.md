# wiki_serach
Hi, 
In this repo you can find an app that retireve articls from wikipedia by august corpus.

files:
  Ranker.py is page rank.
  Most_views.py is most views pages in wikipedia by august 2020.
  Searcher.py is the best search method the search engine could suggest.
  indexer.py is the class that contain the inverted index and add methods. like cossine similarity, binary match.
  Utils.py is class that upload and return the doc titles and return normalize doc for cossine similarty formula.
  BM25.py have a class that impliments the BM25 ranking method.
  search_frontend.py is the "main" of the search engine app. in the app there are 6 routes:     
    1. /search
    2. /search_body
    3. /search_title
    4. /search_anchor
    5. /get_pagerank
    6. /get_pageview

# INCOMING
query qexpansion with thesaurus or wordnet by nltk package
