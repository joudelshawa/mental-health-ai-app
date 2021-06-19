# import pandas as pd
# import searchtweets
# import GetOldTweets3 as got
#
# tweetCriteria = got.manager.TweetCriteria().setQuerySearch('anxiety')\
#                             .setSince("2021-05-30")\
#                            .setMaxTweets(10000)
# tweet_object = got.manager.TweetManager.getTweets(tweetCriteria)
#
#
# def tweet_to_df(tweet_object):
#     tweet_dict = list(map(lambda x: {'text': x.text, 'user': x.username,
#                                      'date': x.date, 'retweet': x.retweets,
#                                      'mention': x.mentions, 'hashtags': x.hashtags,
#                                      'location': x.geo}, tweet_object))
#
#     tweet_df = pd.DataFrame(tweet_dict)
#
#     return tweet_df
#
#
# df = tweet_to_df(tweet_object)
#
#
# # import pandas as pd
# # import tweepy
# # import csv
# #
# # consumer_key = 'vTJwEoiaNx0RpHHlfRjSfKMwI'
# # consumer_secret = 'ovChqGW35Hdken31HoKIFr6DPRLmL9aaJnEl9O9irXLUB9tQFM'
# # access_token = '1014489193-9D7oZPGOvmkTfmcmb6MbIhYts2ES1cCjTtdZFXl'
# # access_token_secret = 'LK2e7qU4gI6hin7XU1kVyhtBY96ARRyUQdLp8Gs7iX9YT'
# #
# # auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# # auth.set_access_token(access_token, access_token_secret)
# # api = tweepy.API(auth)
# #
# # tweet_url = pd.read_csv("Your_Text_File.txt", index_col= None,
# # header = None, names = ["links"])
# #
# # af = lambda x: x["links"].split("/")[-1]
# # tweet_url['id'] = tweet_url.apply(af, axis=1)
# # tweet_url.head()
# #
# # ids = tweet_url['id'].tolist()
# # total_count = len(ids)
# # chunks = (total_count - 1) // 50 + 1
# #
# # def fetch_tw(ids):
# #     list_of_tw_status = api.statuses_lookup(ids, tweet_mode= "extended")
# #     empty_data = pd.DataFrame()
# #     for status in list_of_tw_status:
# #             tweet_elem = {"date": status.created_at,
# #                      "tweet_id":status.id,
# #                      "tweet":status.full_text,
# #                      "User location":status.user.location,
# #                      "Retweet count":status.retweet_count,
# #                      "Like count":status.favorite_count,
# #                      "Source":status.source}
# #             empty_data = empty_data.append(tweet_elem, ignore_index = True)
# #     empty_data.to_csv("new_tweets.csv", mode="a")
# #
# # for i in range(chunks):
# #         batch = ids[i*50:(i+1)*50]
# #         result = fetch_tw(batch)

import requests

r = requests.post('https://stevesie.com/cloud/api/v1/endpoints/7b2e5903-3549-4ba1-bcb6-053e13488690/executions',\
    headers={
        'Token': '3f05be20-2b59-437c-a2c7-a695bbcb43ba',
    },
    json={ "proxy": { "type": "shared", "location": "nyc" }, "format": "json", "inputs": { "query": "-is:retweet", "access_token": "AAAAAAAAAAAAAAAAAAAAAL7bQgEAAAAAmIhcI3pkoomgcZLJ2aAqIxoLkPA%3DWZieh6A343sslcgaYg43IOHmpzSnkTwzRgkL7uY1KBSlXflRCN" } })

response_json = r.json()
print(response_json)