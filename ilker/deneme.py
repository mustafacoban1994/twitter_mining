########Kütüphanelerin içeri aktarımı##########
###############################################
import snscrape.modules.twitter as twitter
import numpy as np
import pandas as pd
from warnings import filterwarnings
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
import os
import datetime
from datetime import datetime,timedelta
import openpyxl
import nltk
import zeyrek
pd.set_option('display.max_colwidth', None) #öncelikle URl kolonunun hepsinin gösterilmesi adına dataframe column expand yapıyorum.
day_ = 1
now = datetime.now() #şimdiki zaman
date_ = now.strftime("%Y-%m-%d") #format çekilmiş şimdiki zaman str yapılmış
ago_ = datetime.now() - timedelta(days=day_) #şimdiki zamandan geriye gidiyoruz gün parametreli
ago_ = ago_.strftime("%Y-%m-%d") #geçmiş zamana format çekilmiş hali str yapılmış
zaman__ = 'since:'+ago_+' until:'+date_+' lang:tr'
zaman__


liste = []  # boş bir liste oluşturdum.
maxTweets = 50000  # maximum kaç tweet çekilecekse onu verdik.



####bir fonksiyon tanımlıyoruz#######
def anyOfWords(keyword=''):
    for i, tweet in enumerate(
            twitter.TwitterSearchScraper(keyword + zaman__).get_items()):
        if i > maxTweets:
            break
        print(i)
        print(tweet.date)
        print(tweet.username)
        print(tweet.content)
        print(tweet.url)
        print(tweet.likeCount)
        print(tweet)
        print((tweet.place))
        print('\n')

        liste.append(
            [tweet.date, tweet.username, tweet.content, tweet.url, tweet.hashtags, tweet.likeCount, tweet.retweetCount])


####yazdığımız fonksiyonun içereceği kelimeleri tanımlıyoruz#####
anyOfWords(
    '(göçmen OR göç OR suriyeli OR mülteci OR afganli OR afganlı OR düzensizgöç OR '
    ' uygur OR doguturkistan OR pakistanlı OR pakistanli OR sınırdışı OR hudut OR yabancı OR uyruklu OR turist OR doğutürkistan OR #göç OR #göçmen OR #suriyeli OR sığınmacı)')
df = pd.DataFrame(liste,
                  columns=['Tarih/Zaman', 'KullanıcıAdı', 'İçerik', 'Url', 'Hashtag', 'BeğeniSayısı', 'PaylaşımSayısı'])