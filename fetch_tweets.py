"""puts tweets in CSV file"""

import sys
import tweepy
import csv

NUM_TWEETS = 200  # change number in brackets for more tweets


def twitter_auth():
    """ authentication function """
    try:
        # application tokens needed to use twitter API
        consumer_key = 'vTJwEoiaNx0RpHHlfRjSfKMwI'
        consumer_secret = 'ovChqGW35Hdken31HoKIFr6DPRLmL9aaJnEl9O9irXLUB9tQFM'
        # tokens that allow API requests on your account's behalf
        access_token = '1014489193-9D7oZPGOvmkTfmcmb6MbIhYts2ES1cCjTtdZFXl'
        access_secret = 'LK2e7qU4gI6hin7XU1kVyhtBY96ARRyUQdLp8Gs7iX9YT'
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
    with open('data_new.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Tweet'])
        for status in tweepy.Cursor(client.home_timeline, screen_name=user).items(NUM_TWEETS):
            print(status.text, '\n')
            writer.writerow([status.text])  # puts data in a csv file
