"""puts tweets in CSV file"""

import sys
import tweepy
import csv

NUM_TWEETS = 200  # change number in brackets for more tweets


def twitter_auth():
    """ authentication function """
    try:
        consumer_key = '*'
        consumer_secret = '*'
        access_token = '*'
        access_secret = '*'
    except KeyError:
        sys.stderr.write("TWITTER_* environment variable not set\n")
        sys.exit(1)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


def get_twitter_client():
    """gets twitter client using given authentication"""
    auth = twitter_auth()
    new_client = tweepy.API(auth, wait_on_rate_limit=True)
    return new_client


if __name__ == '__main__':
    # user = input("Enter username: ")
    user = "joudelshawa"
    client = get_twitter_client()
    with open('real_time_tweets.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Tweet'])
        for status in tweepy.Cursor(client.home_timeline, screen_name=user).items(NUM_TWEETS):
            print(status.text, '\n')
            writer.writerow([status.text])  # puts data in a csv file
    print("done")