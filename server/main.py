# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from pymongo import MongoClient
import vectorizer.tf_idf_vectorizer as tf_idf
import preprocessing.tweet as tweet

client = MongoClient()
db = client['AS-Twitter']

to_time = datetime.utcnow()
from_time = to_time - timedelta(hours=1)

cursor = db.tweets.find({'date': {
                            '$gte': from_time,
                            '$lt': to_time
                        }                                                    
                    })

sent_nosent = tf_idf.build_sentiment_no_sentiment()
pos_neg = tf_idf.build_positive_negative()

neut = 0
pos = 0
neg = 0
for document in cursor:
    doc = tweet.sanitize(document['text'])
    s_ns = sent_nosent.predict([doc])
    
    if s_ns == 's':
        p_n = pos_neg.predict([doc])
        if p_n == 'positive':
            pos += 1
        else:
            neg += 1
    else:
        neut += 1

proportion = {
    'from_time': from_time,
    'to_time': to_time,
    'positive': pos,
    'negative': neg,
    'neutral': neut
}

db.proportions.insert_one(proportion)

db.tweets.delete_many({'date': {
                            '$gte': from_time,
                            '$lt': to_time
                        }
                    })