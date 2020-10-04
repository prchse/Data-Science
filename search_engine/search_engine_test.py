from search_engine import *
import unittest

class TestInvertedIndexDB(unittest.TestCase):

    '''
    Test inverted index of  a given list of documents
    '''

    def test_indexing(self):
        test_docs = dict({'a1': 'A fox is an animal. fox lives in the forest',
                     'a2': 'cat is saying mew',
                     'a3': 'fox is clever',
                     'a4': 'this is fox'})
        db = Database()

        invtd = InvertedIndexDB(db)
        for key,value in test_docs.items():
            final_doc = {'name': key, 'text': value}
            invtd.indexing(final_doc)
        expected_index = defaultdict(list,
            {'a': [('a1', 1)],
             'fox': [('a1', 2), ('a3', 1), ('a4', 1)],
             'animal.': [('a1', 1)],
             'live': [('a1', 1)],
             'forest': [('a1', 1)],
             'cat': [('a2', 1)],
             'say': [('a2', 1)],
             'mew': [('a2', 1)],
             'clever': [('a3', 1)]})
        self.assertEqual(invtd.index, expected_index)


class TestSearchEngine(unittest.TestCase):
    '''
    Test TF-IDf score calculated by tf idf function
    '''

    def test_tf_idf(self): #
        test_docs = [['The', 'man', 'we', 'saw', 'saw', 'a', 'saw'],
                ['The', 'key', 'key', 'is', 'this', 'one'],
                ['I', 'saw', 'him', 'running', 'away'],
                ['This', 'can', 'be', 'done', 'with', 'almost', 'any', 'noun'],
                ['She', 'saw', 'him', 'at', 'the', 'station'],
                ['The', 'sound', 'sounds', 'sound']]
        query = 'saw'
        doc = 'The man we saw saw a saw'

        db = Database()
        invtd = InvertedIndexDB(db)
        se = SearchEngine(invtd.index,db.db)
        term_docs = 0
        word_freq = 0
        for docu in test_docs:
            if query in docu:
                term_docs += 1

        for w in doc.split(' '):
            if w==query:
                word_freq+=1

        tf_idf = se.tf_idf(doc,word_freq,len(test_docs),term_docs)
        expected_score = 0.29706307738283366
        self.assertEqual(tf_idf, expected_score)

    '''
    Test list of relvant documents returned by the search_doc function of SearchEngine Class.
    '''
    def testsearch_doc(self):

        test_docs = dict({'doc1': 'the brown fox jumped over the brown dog',
                     'doc2': 'the lazy brown dog sat in the corner',
                     'doc3': 'the red fox bit the lazy dog'})

        db = Database()
        invtd = InvertedIndexDB(db)
        for key,value in test_docs.items():
            final_doc = {'name': key, 'text': value}
            invtd.indexing(final_doc)
        se = SearchEngine(invtd.index,db.db)

        query = 'brown'
        resulted_doc = se.search_doc(query)
        expected_result = ['doc1', 'doc2']
        self.assertEqual(resulted_doc, expected_result)


        query = 'fox'
        resulted_doc = se.search_doc(query)
        expected_result = ['doc3', 'doc1']
        self.assertEqual(resulted_doc, expected_result)





if __name__ == "__main__":
    unittest.main()