import sys, tweepy
import csv
'''' authentication function '''
def twitter_auth():
    try:
        consumer_key = 'vTJwEoiaNx0RpHHlfRjSfKMwI'
        consumer_secret = 'ovChqGW35Hdken31HoKIFr6DPRLmL9aaJnEl9O9irXLUB9tQFM'
        access_token = '1014489193-9D7oZPGOvmkTfmcmb6MbIhYts2ES1cCjTtdZFXl'
        access_secret = 'LK2e7qU4gI6hin7XU1kVyhtBY96ARRyUQdLp8Gs7iX9YT'
    except KeyError:
        sys.stderr.write("TWITTER_* environment variable not set\n")
        sys.exit(1)
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token,access_secret)
    return auth

def get_twitter_client():
    auth = twitter_auth()
    client = tweepy.API(auth,wait_on_rate_limit=True)
    return client

if __name__ == '__main__':
    # user = input("Enter username: ")
    user = "joudelshawa"
    client = get_twitter_client()
    with open('data_new.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Tweet'])
        for status in tweepy.Cursor(client.home_timeline, screen_name=user).items(200): # change number in brackets for more tweets
            print(status.text, '\n')
            writer.writerow([status.text]) # puts data in a csv file