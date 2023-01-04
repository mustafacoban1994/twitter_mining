####Kütüphanelerin Yüklenmesi####

import snscrape.modules.twitter as twitter
import numpy as np
import pandas as pd
from warnings import filterwarnings
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from openpyxl.chart import BubbleChart
from textblob import Word, TextBlob
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
import os
import datetime
from datetime import datetime, timedelta
import openpyxl
import nltk
import zeyrek
import schedule
import time

pd.set_option('display.max_colwidth',
              None)  # öncelikle URl kolonunun hepsinin gösterilmesi adına dataframe column expand yapıyorum.
day_ = 1
now = datetime.now()  # şimdiki zaman
date_ = now.strftime("%Y-%m-%d")  # format çekilmiş şimdiki zaman str yapılmış
ago_ = datetime.now() - timedelta(days=day_)  # şimdiki zamandan geriye gidiyoruz gün parametreli
ago_ = ago_.strftime("%Y-%m-%d")  # geçmiş zamana format çekilmiş hali str yapılmış
zaman__ = 'since:' + ago_ + ' until:' + date_ + ' lang:tr'

liste = []  # boş bir liste oluşturdum.
maxTweets = 50000


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
anyOfWords(
        '(göçmen OR göç OR suriyeli OR mülteci OR afganli OR afganlı OR düzensizgöç OR '
        ' uygur OR doguturkistan OR pakistanlı OR pakistanli OR sınırdışı OR hudut OR yabancı OR uyruklu OR turist OR doğutürkistan OR #göç OR #göçmen OR #suriyeli OR sığınmacı OR göçidaresi)')

df = pd.DataFrame(liste,
                      columns=['Tarih/Zaman', 'KullanıcıAdı', 'İçerik', 'Url', 'Hashtag', 'BeğeniSayısı', 'PaylaşımSayısı'])

# bir tane daha kopyasını oluşturalım ki bu dataframe in ki yaptığımız yanlışlarda tekrar tekrar çekmek zorunda
# kalmayalım#####
df_ = df.copy()

filterwarnings('ignore')  # bazı hataları ignore ediyorum.
pd.set_option('display.max_columns', None)  # max kolon göstersin.
pd.set_option('display.width', 300000)  # genişliği ayarladım.
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # virgülden sonraki iki basamağı alsın.

# Normalizing Case Folding (büyük küçük harfleri standartlaştırırız.)
df['İçerik'] = df['İçerik'].str.lower()
df_['İçerik'] = df_['İçerik'].str.lower()
# Punctuations (noktalama işareti gördüğünde boşluk ile değiştirelim)
df['İçerik'] = df['İçerik'].str.replace('[^\w\s]', '')
df_['İçerik'] = df_['İçerik'].str.replace('[^\w\s]', '')
# regular expression
# Numbers (sayıları uçurdum)
df['İçerik'] = df['İçerik'].str.replace('\d', '')  # sayıları \d ile buluruz.
df_['İçerik'] = df_['İçerik'].str.replace('\d', '')  # sayıları \d ile buluruz.

####stopwords nltk
import nltk
nltk.download('stopwords')
sw=stopwords.words('turkish')
##stopwords
listes=pd.read_excel("ilker/turkish-stopwords.xlsx") #kendi oluşturduğum stopwordsu okutuyorum.
listem=listes.values.tolist() #listeye çeviriyorum.
for i in range(len(listem)):
    print(i)
 #liste içinde ki listenin parçalanması
sum(listem)


joinedlist=[*sw,*listem]#nltk dan gelen kelimelerle birleştirdim. Yeni bir liste kurdum.
joinedlist.append('hav')
df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in joinedlist)) #listenin içindeki kelimelerden varsa çıkarıyorum. Edat bağlaç gibi gereksiz kelimeler.
df_['İçerik'] = df_['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in joinedlist))


####Rare Words#####Nadir Kelimeleri bulalım######
#geçici bir dataframe oluşturdum
temp_df = pd.Series(' '.join(df['İçerik']).split()).value_counts() #bu df'nin içerisine wordlerin countlarını aldım.
len(temp_df)
sum(temp_df)
drops = temp_df[temp_df <= 2] #frekans değeri 2 ve altında olanları yeni bir listeye attım.
df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops)) #sonrada bunları uçurdum.

#####Görselleştirme######
###BARPLOT####
frekanslar=df['İçerik'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
frekanslar.columns=['Kelime', 'Sayı']
frekanslar.sort_values('Sayı', ascending=False)
sirali_frekanslar=frekanslar[frekanslar['Sayı']>=200]
sirali_frekanslar.sort_values('Sayı',ascending=False)
sirali_frekanslar.plot.bar(x='Kelime', y='Sayı')
plt.show()

df['Tarih/Zaman'].sort_values(ascending=True)

import plotly.express as px
import pandas as pd

df.tail()