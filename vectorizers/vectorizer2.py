import nltk
import numpy as np

from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin

from nltk.corpus import sentiwordnet as swn

class Vectorizer2(BaseEstimator, VectorizerMixin):
    """Convert a collection of text documents to a matrix of features
    This implementation produces a sparse representation of the counts using
    scipy.sparse.coo_matrix.
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.vocabulary = vocabulary
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = nltk.word_tokenize
        self.preprocessor = self.build_preprocessor()
        self.part_of_speech = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']

    def _remove_stop_words(self, text):
        tokens = self.build_tokenizer()(text)
        stops = self.get_stop_words()
        result = [x for x in tokens if x not in stops]
        return " ".join(result)

    def _get_sentiment(self, word):
        synsets = list(swn.senti_synsets(word.lower()))
        pos_score = sum(ss.pos_score() for ss in synsets)/float(len(synsets))
        neg_score = sum(ss.neg_score() for ss in synsets)/float(len(synsets))
        obj_score = sum(ss.obj_score() for ss in synsets)/float(len(synsets))
        return {'pos_score': pos_score,
                'neg_score': neg_score,
                'obj_score': obj_score}

    def _get_avg_sentiment(self, text):
        words = self.build_tokenizer()(text)
        sentiments = [self._get_sentiment(word) for word in words]
        pos_score = sum(s['pos_score'] for s in sentiments)/float(len(sentiments))
        neg_score = sum(s['neg_score'] for s in sentiments)/float(len(sentiments))
        obj_score = sum(s['obj_score'] for s in sentiments)/float(len(sentiments))
        return {'pos_score': pos_score,
                'neg_score': neg_score,
                'obj_score': obj_score}

    def _count_pos(self, text):
        tokens = self.build_tokenizer()(text)
        text = nltk.Text(tokens)
        tagged = nltk.pos_tag(text, tagset='universal')
        counts = Counter(tag for word,tag in tagged)
        total = sum(counts.values())
        res = dict((word, float(count)/total) for word,count in counts.items())
        for pos in self.part_of_speech:
            if pos not in res:
                res[pos] = 0
        return res

    def fit(self, raw_documents, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return features matrix.
        This is equivalent to fit followed by transform.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        matrix : array, [n_samples, n_features]
            Document-features matrix.
        """

        matrix = np.zeros((len(raw_documents), len(self.part_of_speech)+3))
        cont = 0
        for doc in raw_documents:
            doc = self._remove_stop_words(self.preprocessor(doc))
            print(doc)

            sentiment = self._get_avg_sentiment(doc)
            pos = self._count_pos(doc)
            matrix[cont, 0] = sentiment['pos_score']
            matrix[cont, 1] = sentiment['neg_score']
            matrix[cont, 2] = sentiment['obj_score']
            matrix[cont, 3] = pos['ADJ']
            matrix[cont, 4] = pos['ADP']
            matrix[cont, 5] = pos['ADV']
            matrix[cont, 6] = pos['CONJ']
            matrix[cont, 7] = pos['DET']
            matrix[cont, 8] = pos['NOUN']
            matrix[cont, 9] = pos['NUM']
            matrix[cont, 10] = pos['PRT']
            matrix[cont, 11] = pos['PRON']
            matrix[cont, 12] = pos['VERB']
            matrix[cont, 13] = pos['.']
            matrix[cont, 14] = pos['X']
            cont += 1

        self.matrix = matrix
        return self.matrix

    def transform(self, raw_documents):
        return self.fit_transform(raw_documents)

    def get_feature_names(self):
        return ['pos_score', 'neg_score', 'obj_score', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']