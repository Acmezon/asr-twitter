# -*- coding: utf-8 -*-
import tweepy
import time
from StreamListener import TwitterListener

consumer_key = "d6gTepyCjNfsPRmHiyGMQT1ZU"
consumer_secret = "FP9pSHT3uLDEEyruVzJByl5GoNQqO5Ll8in7Ockhq9u4JlTSVw"
access_token = "388301947-xdsg5uVrSKlHD5iutyTzijNlrlwOjKfJp7hWoQN7"
access_token_secret = "4PN4K1d4byBq4Mra7xmz6X4M803rOoJwIpqi3sB1pY8Jr"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

while True:
    try:
        print("Waking up!")
        myStreamListener = TwitterListener(api)
        myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
        
        myStream.filter(languages=['en'], track=['Goodell'])
    except TimeoutError:
        print("Going to sleep...")
        time.sleep(15 * 60)