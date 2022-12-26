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



day_ = 1
now = datetime.now() #şimdiki zaman
date_ = now.strftime("%Y-%m-%d") #format çekilmiş şimdiki zaman str yapılmış
ago_ = datetime.now() - timedelta(days=day_) #şimdiki zamandan geriye gidiyoruz gün parametreli
ago_ = ago_.strftime("%Y-%m-%d") #geçmiş zamana format çekilmiş hali str yapılmış
zaman__ = 'since:'+ago_+' until:'+date_+' lang:tr'
zaman__


liste = []  # boş bir liste oluşturdum.
maxTweets = 2000  # maximum kaç tweet çekilecekse onu verdik.


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
        print('\n')
        liste.append(
            [tweet.date, tweet.username, tweet.content, tweet.url, tweet.hashtags, tweet.likeCount, tweet.retweetCount])


####yazdığımız fonksiyonun içereceği kelimeleri tanımlıyoruz#####
anyOfWords(
    '(#göçmen OR #göç OR #suriyeli OR #mülteci OR #afganli OR #afganlı OR #düzensizgöç OR #ümitözdağ OR #umitozdag OR '
    '#ümitozdag OR #zaferpartisi OR #uygur OR #doguturkistan OR #pakistanlı OR #pakistanli)')
df = pd.DataFrame(liste,
                  columns=['Tarih/Zaman', 'KullanıcıAdı', 'İçerik', 'Url', 'Hashtag', 'BeğeniSayısı', 'PaylaşımSayısı'])

#bir tane daha kopyasını oluşturalım ki bu dataframe in ki yaptığımız yanlışlarda tekrar tekrar çekmek zorunda kalmayalım#####
df_=df.copy()
df_

####şimdi daplikasyona düşen varsa yani yinelenen varsa onları kaldırsın
