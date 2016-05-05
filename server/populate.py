# -*- coding: utf-8 -*-
import random
import math
from datetime import datetime, timedelta
from pymongo import MongoClient

client = MongoClient()
db = client['AS-Twitter']

for i in range(23):
    from_time = datetime.now()
    from_time = from_time.replace(hour=i, minute=0, second=0, microsecond=0)
    
    to_time = datetime.now()
    to_time = to_time.replace(hour=i+1, minute=0, second=0, microsecond=0)
    
    tweets_prop = [random.random() for _ in range(3)]
    tweets_prop = [math.floor((i / sum(tweets_prop)) * 100) for i in tweets_prop]
    
    proportion = {
        'from_time': from_time,
        'to_time': to_time,
        'positive': tweets_prop[0],
        'negative': tweets_prop[1],
        'neutral': tweets_prop[2]
    }
    
    db.proportions.insert_one(proportion)
    
    from_time = from_time + timedelta(1)
    to_time = to_time + timedelta(1)
    
    proportion = {
        'from_time': from_time,
        'to_time': to_time,
        'positive': tweets_prop[0],
        'negative': tweets_prop[1],
        'neutral': tweets_prop[2]
    }
    
    
    db.proportions.insert_one(proportion)