import mwparserfromhell as mwparserfromhell
import numpy as np
import bz2
from functools import partial
from collections import Counter
import pickle
from itertools import islice
from xml.etree import ElementTree
import codecs
import csv
import time
import os
import re
from pathlib import Path
from nltk.stem.porter import *
from nltk.corpus import stopwords

import wget as wget
import mwparserfromhell as mwp

mwp.definitions.INVISIBLE_TAGS.append('ref')


class Parser:

    def __init__(self, data):
        self.wiki_file = data

        self.RE_TOKENIZE = re.compile(rf"""
                                  (
                                  # parsing html tags
                                   (?P<HTMLTAG>{self.get_html_pattern()})                                  
                                  # dates
                                  |(?P<DATE>{self.get_date_pattern()})
                                  # time
                                  |(?P<TIME>{self.get_time_pattern()})
                                  # Percents
                                  |(?P<PERCENT>{self.get_percent_pattern()})
                                  # Numbers
                                  |(?P<NUMBER>{self.get_number_pattern()})
                                  # Words
                                  |(?P<WORD>{self.get_word_pattern()})
                                  # space
                                  |(?P<SPACE>[\s\t\n]+) 
                                  # everything else
                                  |(?P<OTHER>.)
                                  )
                                  """,
                                      re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)

        self.en_stopwords = frozenset(stopwords.words('english'))

        self.stemmer = PorterStemmer()

    def page_iter(self):  # , wiki_file):
        """ Reads a wiki dump file and create a generator that yields pages.
    Parameters:
    -----------
    wiki_file: str
      A path to wiki dump file.
    Returns:
    --------
    tuple
      containing three elements: article id, title, and body.
    """
        # open compressed bz2 dump file
        with bz2.open(self.wiki_file, 'rt', encoding='utf-8', errors='ignore') as f_in:
            # Create iterator for xml that yields output when tag closes
            elems = (elem for _, elem in ElementTree.iterparse(f_in, events=("end",)))
            # Consume the first element and extract the xml namespace from it.
            # Although the raw xml has the  short tag names without namespace, i.e. it
            # has <page> tags and not <http://wwww.mediawiki.org/xml/export...:page>
            # tags, the parser reads it *with* the namespace. Therefore, it needs the
            # namespace when looking for child elements in the find function as below.
            elem = next(elems)
            m = re.match("^{(http://www\.mediawiki\.org/xml/export-.*?)}", elem.tag)
            if m is None:
                raise ValueError("Malformed MediaWiki dump")
            ns = {"ns": m.group(1)}
            page_tag = ElementTree.QName(ns['ns'], 'page').text
            # iterate over elements
            for elem in elems:
                if elem.tag == page_tag:
                    # Filter out redirect and non-article pages
                    if elem.find('./ns:redirect', ns) is not None or \
                            elem.find('./ns:ns', ns).text != '0':
                        elem.clear()
                        continue
                    # Extract the article wiki id
                    wiki_id = elem.find('./ns:id', ns).text
                    # Extract the article title into a variables called title
                    # YOUR CODE HERE
                    # raise NotImplementedError()
                    title = elem.find('./ns:title', ns).text
                    # extract body
                    body = elem.find('./ns:revision/ns:text', ns).text

                    yield wiki_id, title, body
                    elem.clear()

    def filter_article_links(self, title):
        """ Return false for wikilink titles (str) pointing to non-articles such as images, files, media and more (as described in the documentation).
        Otherwise, returns true.
    """
        # YOUR CODE HERE
        # Filter out redirect and non-article pages
        pattern = r'File|Media|Image|#|Category|Help|Manual|Special|User|Extension|/|{'
        if re.match(pattern, title):
            # print("match")
            return False
        # raise NotImplementedError()
        # print("not match")
        return True

    def get_wikilinks(self, wikicode):
        """ Traverses the parse tree for internal links and filter out non-article
    links.
    Parameters:
    -----------
    wikicode: mwp.wikicode.Wikicode
      Parse tree of some WikiMedia markdown.
    Returns:
    --------
    list of (link: str, anchor_text: str) pair
      A list of outgoing links from the markdown to wikipedia articles.
    """
        links = []
        for wl in wikicode.ifilter_wikilinks():
            # skip links that don't pass our filter
            title = str(wl.title)
            # print(title)
            if not self.filter_article_links(title):
                continue

            # if text is None use title, otherwise strip markdown from the anchor text.
            text = wl.text
            if text is None:
                text = title
            else:
                text = text.strip_code()
            # remove any lingering section/anchor reference in the link
            # YOUR CODE HERE

            pattern = r'#|:'
            title = re.split(pattern, title)[0]
            # raise NotImplementedError()
            links.append((title, text))
        return links

    """# 2. Tokenization
  
  Before tokenizing Wikipedia articles' text we need to remove any remaining MediaWiki markdown from the text. Luckily, our parser knows how to strip all markdown as demonstrated by the following example:
  """

    def remove_markdown(self, text):
        return mwp.parse(text).strip_code()

    # print(remove_markdown("""
    # == Section 2 ==
    # [[File:image1.jpg| '''''beautiful''''' <b>image</b> of [[Wikipedia]]]]
    # """))

    """Great! now we can focus on tokenzing the clean text. Here's the clean text of one article after preprocessing:"""

    """**YOUR TASK (70 POINTS)**: Complete the implementation of the functions in the next cell that return regular expressions (as strings) to capture dates, time, etc. in the text. """

    # YOUR CODE HERE

    def get_html_pattern(self):
        """ Return a string regex pattern for capturing HTML tags. No need to handle
    tags nested inside arrtibutes of other html tags, i.e. you can assume that
    there will never be a tag like <b style="<i>">.
    """
        # YOUR CODE HERE
        # return r'<(\"[^\"]*\"|[^]*|[^\">])*>'
        return r'(<(\w+\s?(\w+\W"\w+\W?\w+")?\/?)>)|(<(\/\w+)>)'

    def get_date_pattern(self):
        """ Return a string regex pattern for capturing dates in the format of
    January 29, 1984, Nov 3, 2020, or 3 Nov 2020. No need to handle negative
    years, other formats, or check .
    """
        # YOUR CODE HERE
        ###raise NotImplementedError()
        date1 = r'([a-zA-Z]+)\s(([0-2]?[0-9])|(3[0-1]))\,\s(\d\d\d\d)'
        date2 = r'((?<!\d)(([0-2]?[0-9])|(3[0-1]))\s(?!.*feb)[a-zA-Z]+\s(\d\d\d\d))'
        feb = r'([0-2][0-8]\s([Ff]eb|[Ff]ebruary)\s\d\d\d\d)'
        return '{0}|{1}|{2}'.format(date1, date2, feb)

    def get_time_pattern(self):
        """ Return a string regex pattern for capturing time in XX.XX AM/PM,
    XX.XX A.M./P.M. (period is optional) or HH:MM:SS format.
    """
        # YOUR CODE HERE
        time1 = '(0[0-9]|1[0-9]|2[0-3])\.[0-5][0-9]([aA]|[pP])[mM]'
        time2 = '(0[0-9]|1[0-9]|2[0-3])(\.)?[0-5][0-9]([aA]|[pP])\.[mM]\.'
        time3 = '((0?[0-9]|1[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9])((?=\s)|(?=$))'
        return '((?<=(\s))|(?<=(^)))({0}|{1}|{2})'.format(time1, time2, time3)

    def get_number_pattern(self):
        """ Return a string regex pattern for capturing positive/negative numbers
    with optional decimal part and may include commas. """
        # YOUR CODE HERE
        ###raise NotImplementedError()
        return r'(?<![\w\+\-,\.])[\+\-]?\d{1,3}((,\d{3})*|\d*)(\.\d+)?(?!\S?[\w\+\-])'

    def get_percent_pattern(self):
        """ Return a string regex pattern for capturing percentages with the same
    number format as before.
    """
        return r"""(?<![\w\+\-,\.])[\+\-]?\d{1,3}((,\d{3})*|\d*)
    (\.\d+)?%(?!\S?[\w\+\-])"""

    def get_word_pattern(self):
        """ Return a string regex pattern for capturing words. It needs to handle
    words with apostrophes and dashes in them, but no word starts with these.
    """
        # YOUR CODE HERE
        ###raise NotImplementedError()
        return r'((?<=(\s))|(?<=(^)))([a-zA-z]+)(\'[a-zA-Z]*)?(\-[a-zA-Z]+)*'

    def tokenize(self, text):
        return [(v, k) for match in self.RE_TOKENIZE.finditer(text)
                for k, v in match.groupdict().items()
                if v is not None and k != 'SPACE']

    # html basic tests (7 points)
    def tok(self, text):
        return self.tokenize(self.remove_markdown(text))

    # tokens = tok(r'<nowiki><b>hello</b></nowiki>')
    # print(tokens)
    # assert ('<b>', 'HTMLTAG') in tokens
    # assert ('</b>', 'HTMLTAG') in tokens
    # tokens = tok(r'<nowiki><b style="color:red">hello</b></nowiki>')
    # assert ('<b style="color:red">', 'HTMLTAG') in tokens
    # tokens = tok(r'<nowiki><br /></nowiki>')
    # assert ('<br />', 'HTMLTAG') in tokens
    #
    # # html advanced tests (6 points)
    # tokens = tok(r'<nowiki><b><i>hello</i></b></nowiki>')
    # assert 4 == sum([1 for _, t in tokens if t == 'HTMLTAG'])
    #
    # # dates basic tests (7 points)
    # tokens = tok(r'dates in the format of January 29, 1984, Nov 3, 2020, or 3 Nov 2020.')
    # assert ('January 29, 1984', 'DATE') in tokens
    # assert ('Nov 3, 2020', 'DATE') in tokens
    # assert ('3 Nov 2020', 'DATE') in tokens
    #
    # # dates advanced tests (6 points)
    # tokens = tok(r'Sep 29, 1984, Apr 33, 2020, or 30 feb 2020.')
    # print(tokens)
    # assert ('Sep 29, 1984', 'DATE') in tokens
    # assert ('Apr 33, 2020', 'DATE') not in tokens
    # assert ('30 feb 2020', 'DATE') not in tokens
    #
    # # time basic tests (7 points)
    # tokens = tok(r'12.12PM 1202a.m. 6:12:12')
    # print(tokens)
    # assert ('12.12PM', 'TIME') in tokens
    # assert ('1202a.m.', 'TIME') in tokens
    # assert ('6:12:12', 'TIME') in tokens
    #
    # # time advanced tests (6 points)
    # tokens = tok(r'36.12PM 1272a.m. 1202a.m 12:12:12am 56:12:12 6:72:12')
    # print(tokens)
    # assert 0 == sum([1 for _, t in tokens if t == 'TIME'])
    #
    # # number basic tests (7 points)
    # tokens = tok(r"""12 +12 -12.0 -12,345.5466 +12,345,678,678 0.154""")
    # print(tokens)
    # assert ('12', 'NUMBER') in tokens
    # assert ('+12', 'NUMBER') in tokens
    # assert ('-12.0', 'NUMBER') in tokens
    # assert ('-12,345.5466', 'NUMBER') in tokens
    # assert ('+12,345,678,678', 'NUMBER') in tokens
    # assert ('0.154', 'NUMBER') in tokens
    #
    # # number advanced tests (6 points)
    # print(tok('the pound (500 in value)...'))
    # assert ('500', 'NUMBER') in tok('the pound (500 in value)...')
    # assert ('500', 'NUMBER') in tok('the price is 500.')
    # assert ('500', 'NUMBER') in tok('the price is 500, but it is negotiable.')
    # assert ('500', 'NUMBER') in tok('the price is 500: no less!')
    # assert ('500', 'NUMBER') not in tok('the price rose 500%')
    # tokens = tok(r"""12.A W12 +-12 -.12.0 -12,34.5466 +12,345,6+78,678 0.15,4""")
    # print(tokens)
    # assert 0 == sum([1 for _, t in tokens if t == 'NUMBER'])
    #
    #
    #
    # # word tests (13 points)
    # tokens = tok(r"""Hello Bob! It's Mary, your mother-in-law,
    #   the mistake is your parents'! --Mom""")
    # print(tokens)
    # assert ('Hello', 'WORD') in tokens
    # assert ('Bob', 'WORD') in tokens
    # assert ("It's", 'WORD') in tokens
    # assert ('Mary', 'WORD') in tokens
    # assert ('your', 'WORD') in tokens
    # assert ('mother-in-law', 'WORD') in tokens
    # assert ("parents'", 'WORD') in tokens
    # assert ("Mom", 'WORD') not in tokens
    # assert ("-Mom", 'WORD') not in tokens
    # assert ("--Mom", 'WORD') not in tokens
    #
    # # comprehensiveness test (5 points)
    # _, t = zip(*tok(pages[9][2]))
    # # assert 5 == len(set(t))

    """# 3. Collect and merge page views
  
    Data about page views on Wikipedia is available at https://dumps.wikimedia.org and there is documentation about the [definition of a page view](https://meta.wikimedia.org/wiki/Research:Page_view) and the [format of lines](https://dumps.wikimedia.org/other/pagecounts-ez/) in the file. In the class project, you will need to use page view data that we'll provide for ALL of English Wikipedia from the month of August 2021, which is more than 10.7 million viewed articles. The commented out code below shows how we generate that data, no need to run it yourself, this is just for your information. 
    """

    """In order to keep things simple, in this assignment we provide you with a small sample of articles and their page view counts.
    """
    'upgrade the ranker module - NEED TO REMOVE TO RANKER.py'

    def most_viewed(self, pages, wid2pv):
        """Rank pages from most viewed to least viewed using the above `wid2pv`
       counter.
    Parameters:
    -----------
      pages: An iterable list of pages as returned from `page_iter` where each
             item is an article with (id, title, body)
    Returns:
    --------
    A list of tuples
      Sorted list of articles from most viewed to least viewed article with
      article title and page views. For example:
      [('Langnes, Troms': 16), ('Langenes': 10), ('Langenes, Finnmark': 4), ...]
    """
        # YOUR CODE HERE
        view_list = []
        for page in pages:
            if wid2pv.get(page[0]):
                view_list.append((page[1], wid2pv.get(page[0])))
            else:
                view_list.append((page[1], 0))
        view_list.sort(key=lambda x: x[1], reverse=True)
        return view_list

    def filter_tokens(self):
      pass

    # Getting tokens from the text while removing punctuations.
    def filter_tokens(self, tokens, tokens2remove=None, use_stemming=False):
        ''' The function takes a list of tokens, filters out `tokens2remove` and
          stem the tokens using `stemmer`.
          Parameters:
          -----------
          tokens: list of str.
            Input tokens.
          tokens2remove: frozenset.
            Tokens to remove (before stemming).
          use_stemming: bool.
            If true, apply stemmer.stem on tokens.
          Returns:
          --------
          list of tokens from the text.
        '''

        tmp_tokens = []
        # not change the token
        if (not tokens2remove) and (not use_stemming):
            return tokens
        for x in tokens:
            # we dont take the word
            if x in tokens2remove:
                continue
            # use stem on the word
            if use_stemming:
                tmp_tokens.append(self.stemmer.stem(x))
            # take the word without using stem on it
            else:
                tmp_tokens.append(x)
        tokens = tmp_tokens

        return tokens
