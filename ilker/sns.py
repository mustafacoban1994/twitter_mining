import snscrape.modules.twitter as twitter
import csv

maxTweets = 100
def tweetSearch(keyword=''):
    for i, tweet in enumerate(twitter.TwitterSearchScraper(keyword + ' since:2020-11-01 until:2021-01-01 lang:"en" ').get_items()):
        if i > maxTweets:
            break
        print(i)
        print(tweet.username)
        print(tweet.content)
        print(tweet.date)
        print(tweet.url)
        print('\n')

tweetSearch('trump')