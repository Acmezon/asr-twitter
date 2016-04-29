import nltk
import numpy as np

from collections import Counter
from scipy.sparse import csc_matrix
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin
from nltk.corpus import sentiwordnet as swn

class NLTKVectorizer(BaseEstimator, VectorizerMixin):
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
        self.part_of_speech = self.get_feature_names()[3:]
        self.sentiments = {}

    def _remove_stop_words(self, text):
        """
        Deletes all the stop_words previously defined from a text
        """
        if self.stop_words is not None:
            tokens = self.build_tokenizer()(text)
            stops = self.get_stop_words()
            result = [x for x in tokens if x not in stops]
            result = " ".join(result)
        else:
            result = text
        
        return result

    def _get_sentiment(self, word):
        """
        Retrieve the sentiments of a word via SentiWordNet lexicon.
        If not previously learned, calculate itself.
        It is calculated as the average sentiment of all its synsets.
        """
        if word in self.sentiments:
            return self.sentiments[word]
        else:
            synsets = list(swn.senti_synsets(word.lower()))
            if len(synsets)>0:
                pos_score = sum(ss.pos_score() for ss in synsets)/float(len(synsets))
                neg_score = sum(ss.neg_score() for ss in synsets)/float(len(synsets))
                obj_score = sum(ss.obj_score() for ss in synsets)/float(len(synsets))
                sentiment = {'pos_score': pos_score,
                        'neg_score': neg_score,
                        'obj_score': obj_score}
            else:
                sentiment = {'pos_score': 0.0,
                        'neg_score': 0.0,
                        'obj_score': 1.0}
            self.sentiments[word] = sentiment
        return sentiment

    def _get_avg_sentiment(self, text):
        """
        Calculate the sentiments of a text as the avergage sentiments of its words.
        The sentiment values are retrieved via SentiWordNet lexicon.
        """
        words = self.build_tokenizer()(text)
        sentiments = [self._get_sentiment(word) for word in words]
        pos_score = sum(s['pos_score'] for s in sentiments)/float(len(sentiments))
        neg_score = sum(s['neg_score'] for s in sentiments)/float(len(sentiments))
        obj_score = sum(s['obj_score'] for s in sentiments)/float(len(sentiments))
        return {'pos_score': pos_score,
                'neg_score': neg_score,
                'obj_score': obj_score}

    def _count_pos(self, text):
        """
        Calculate proportion of each part of speech (POS) of the text.
        Returns
        ----------
            dictionary
                keys: POS-type of word (string)
                values: proportion (float)
        """
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
        """Learn a sentiment vocabulary dictionary of all tokens in the raw documents.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        for doc in raw_documents:
            doc = self._remove_stop_words(self.build_preprocessor()(doc))
            self._get_avg_sentiment(doc)

        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn the sentiment vocabulary dictionary and return features matrix.
        This is equivalent to fit followed by transform.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        matrix : csc sparse matrix array, [n_samples, n_features]
            Document-features matrix.
        """

        cont = 0
        rows = []
        cols = []
        data = []
        for doc in raw_documents:
            doc = self._remove_stop_words(self.build_preprocessor()(doc))

            sentiment = self._get_avg_sentiment(doc)
            pos = self._count_pos(doc)

            if sentiment['pos_score']!=0:
                rows.append(cont)
                cols.append(0)
                data.append(sentiment['pos_score'])
            if sentiment['neg_score']!=0:
                rows.append(cont)
                cols.append(1)
                data.append(sentiment['neg_score'])
            if sentiment['obj_score']!=0:
                rows.append(cont)
                cols.append(2)
                data.append(sentiment['obj_score'])
            if pos['ADJ']!=0:
                rows.append(cont)
                cols.append(3)
                data.append(pos['ADJ'])
            if pos['ADP']!=0:
                rows.append(cont)
                cols.append(4)
                data.append(pos['ADP'])
            if pos['ADV']!=0:
                rows.append(cont)
                cols.append(5)
                data.append(pos['ADV'])
            if pos['CONJ']!=0:
                rows.append(cont)
                cols.append(6)
                data.append(pos['CONJ'])
            if pos['DET']!=0:
                rows.append(cont)
                cols.append(7)
                data.append(pos['DET'])
            if pos['NOUN']!=0:
                rows.append(cont)
                cols.append(8)
                data.append(pos['NOUN'])
            if pos['NUM']!=0:
                rows.append(cont)
                cols.append(9)
                data.append(pos['NUM'])
            if pos['PRT']!=0:
                rows.append(cont)
                cols.append(10)
                data.append(pos['PRT'])
            if pos['PRON']!=0:
                rows.append(cont)
                cols.append(11)
                data.append(pos['PRON'])
            if pos['VERB']!=0:
                rows.append(cont)
                cols.append(12)
                data.append(pos['VERB'])
            if pos['.']!=0:
                rows.append(cont)
                cols.append(13)
                data.append(pos['.'])
            if pos['X']!=0:
                rows.append(cont)
                cols.append(14)
                data.append(pos['X'])

            cont += 1

        rows = np.array(rows)
        cols = np.array(cols)
        data = np.array(data)
        shape = (len(raw_documents), len(self.part_of_speech)+3)

        self.matrix = csc_matrix((data, (rows, cols)), shape=shape)
        return self.matrix

    def transform(self, raw_documents):
        return self.fit_transform(raw_documents)

    def get_feature_names(self):
        return ['pos_score', 'neg_score', 'obj_score', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']