# -*- coding: utf-8 -*-
import tweepy
from pprint import pprint

#override tweepy.StreamListener to add logic to on_status
class TwitterListener(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api

    def on_status(self, status):
        pprint(status.text)
        print("----------------------------")
        
    def on_error(self, status_code):
        print(status_code)
        if status_code == 420:
            raise TimeoutError("Rate limit exceed")