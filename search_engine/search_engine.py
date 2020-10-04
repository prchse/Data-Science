import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import *
import collections
from collections import defaultdict
import string
import math
import argparse
import unittest

'''
Preprocessing of a single word: including stemming and converting into lowercase

'''

def preprocessing(word):
    stemmer = PorterStemmer()
    word= word.lower()
    word = stemmer.stem(word)
    return word

'''

Database to store documents woth their name which is unique

'''

class Database:
    def __init__(self):
        self.db = dict()
    '''
    getting a document in the database using the given name/id
    '''

    def get_doc(self, doc_name):
        return self.db.get(doc_name, None)

    '''
     Adding a document in the database
     '''
    def add_doc(self, doc):
        return self.db.update({doc['name']: doc['text']})


'''

InvertedIndexDB Class to convert a document into inverted index and save it into the Database

'''


class InvertedIndexDB:
    def __init__(self, db):
        self.index = defaultdict(list)
        self.db = db

    def indexing(self, doc):
        '''
        :param doc: a single document to do indexing and save it into a database.
                    Required Format:{'name': id of the doc, 'text':'content of the doc'}(dictionary)
        :return: None
        '''
        # removing stop words and punctuation
        stop_words = list(stopwords.words('english'))
        punc = list(string.punctuation)
        stop = stop_words + punc

        value = doc['text'].splitlines()
        words = [preprocessing(w) for l in value for w in l.strip().split() if not w in stop]

        word_freq = collections.Counter(words)  # count frequency of each word in doc

        for key_c, value_c in word_freq.items():  # add (doc,frequency) in index
            if key_c not in self.index:
                self.index[key_c] = [(doc['name'], value_c)]
            else:
                self.index[key_c].append((doc['name'], value_c))

        self.db.add_doc(doc)
'''
Search Engine which takes indexed data and database and gives relevant document
given a query using the search_doc function and sorted by Tf-idf scor eimplemented in  
This result is sorted by
'''

class SearchEngine:
    def __init__(self, indexed_data, db):
        self.indexed_data = indexed_data
        self.db = db

    def search_doc(self, query):
        '''

        :param query: single word to be searched (string)
        :return: list of document_id sorted by tf_idf (list)
        '''
        searched_word = preprocessing(query)
        relvant_doc=[]
        if searched_word in self.indexed_data:
            relvant_doc = self.indexed_data[searched_word]

        tf_idf_socred = dict()

        for doc_dict in relvant_doc:
            doc = self.db[doc_dict[0]]
            tf_idf_score = self.tf_idf(doc, doc_dict[1], len(self.db.keys()), len(relvant_doc))
            tf_idf_socred[doc_dict[0]] = tf_idf_score

        tf_idf_sorted = sorted(tf_idf_socred, key=tf_idf_socred.get, reverse=True)

        return tf_idf_sorted

    @staticmethod
    def tf_idf(doc, w_freq, total_doc, term_docs):
        '''

        :param doc: document to calculate tf (string)
        :param w_freq: frequency of a given word in 'doc'(int)
        :param total_doc: total number of docs(int)
        :param term_docs: total number of docs having given word(int)
        :return: TF-IDF score (float)
        '''

        totalterm = len(doc.strip().split())
        tf_doc = w_freq / totalterm
        idf = 0

        if term_docs > 0:
            idf = math.log(total_doc / term_docs)

        tfidf = tf_doc * idf

        return tfidf


def read_file_n_indexing(dir_path,index):
    '''

    :param dir_path: path to a directory having all text files in .txt format. for
                     For example if all documents are in reuters-topics/grain. Expected path: 'reuters-topics/grain/'
    :param index: object of InvertedIndexDB
    :return: object of InvertedIndexDB
    '''
    all_file_grain = os.listdir(dir_path)

    for idx, doc in enumerate(all_file_grain):
        with open(dir_path + doc, encoding='utf-8') as f:
            final_doc = {'name': doc, 'text': f.read()}
            index.indexing(final_doc)
    return index






if __name__ == "__main__":

    # parsing   command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('filesdir',
                        type=str,
                        help='The root directory containing all text files. example : path_to_directory/directory_conating_files/')
    parser.add_argument('query',
                        type=str,
                        help='single word : search query')

    args = parser.parse_args()
    dir_path= args.filesdir
    query = args.query

    # reading the files from directory and search for given single term

    db = Database()
    index= InvertedIndexDB(db)
    indexed = read_file_n_indexing(dir_path, index)
    se = SearchEngine(index.index, db.db)
    resulted_doc = se.search_doc(query)

    for doc in resulted_doc:  #listing the relevant documents having first line of it
        print('****************************')
        print ("document name ---- ", doc)
        print ("document's first line----")
        print(db.get_doc(doc).splitlines()[0])

    if len(resulted_doc) == 0 :

        print('No relevant document is found')

    '''
    run the program in command line as 
    'python search_engine.py  [path of the file for example: /Users/pragya/Downloads/NLP/a2/reuters-topics/grain/]  [query to be searched]'
    '''

