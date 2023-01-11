# #VERİ ÇEKELİM GALDAŞ
# import snscrape.modules.twitter as sntwitter
# import pandas as pd
#https://pypi.org/project/tweet-capture/
#
# # Creating list to append tweet data to
# tweets_list = []
#
# # Using TwitterSearchScraper to scrape data and append tweets to list
# for i, tweet in enumerate(
#         sntwitter.TwitterSearchScraper('Suriyeli since:2022-11-01 until:2022-11-27 lang:tr').get_items()):
#     if i > 500:
#         break
#     tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.user.username,tweet.hashtags,tweet.likeCount,tweet.retweetCount])
#
# # Creating a dataframe from the tweets list above
# tweets_df = pd.DataFrame(tweets_list, columns=['Tarih', 'Id', 'İçerik', 'Hashtag','Kullanıcı','Beğeni','Paylaşma'])
# tweets_df.count()
# tweets_df
#
# # tweets_df[tweets_df.columns[2]].to_excel("C:\\Users\\ilker\\OneDrive\\Desktop\\pytweet\\deneme4.xlsx")


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
import re

liste = []
maxTweets = 2000


def anyOfWords(keyword=''):
    for i, tweet in enumerate(
            twitter.TwitterSearchScraper(keyword + ' since:2022-12-18 until:2022-12-19 lang:tr ').get_items()):
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


anyOfWords(
    '(#göçmen OR #göç OR #suriyeli OR #mülteci OR #afganli OR #afganlı OR #düzensizgöç OR #ümitözdağ OR #umitozdag OR #ümitozdag OR #zaferpartisi OR #uygur OR #doguturkistan OR #pakistanlı OR #pakistanli)')
df = pd.DataFrame(liste,
                  columns=['Tarih/Zaman', 'KullanıcıAdı', 'İçerik', 'Url', 'Hashtag', 'BeğeniSayısı', 'PaylaşımSayısı'])



df_=df.copy()

df["İçerik"].drop_duplicates(inplace=True)
df_

# NELER YAPACAĞIZ GALDAŞ?

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment


# Sentiment Analysis and Sentiment Modeling for Twitter (keyword=göç,etc.)
# 1. Text Preprocessing
filterwarnings('ignore')  # bazı hataları ignore ediyorum.
pd.set_option('display.max_columns', None)  # max kolon göstersin.
pd.set_option('display.width', 1000)  # genişliği ayarladım.
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # virgülden sonraki iki basamağı alsın.

df.head()
df.info()

# Normalizing Case Folding (büyük küçük harfleri standartlaştırırız.)
df['İçerik']
df['İçerik'] = df['İçerik'].str.lower()

# Punctuations (noktalama işareti gördüğünde boşluk ile değiştirelim)
df['İçerik'] = df['İçerik'].str.replace('[^\w\s]', '')

# regular expression
# Numbers (sayıları uçurdum)
df['İçerik'] = df['İçerik'].str.replace('\d', '')  # sayıları \d ile buluruz.

# Stopwords (bağlaçlar falan anlamsız cümleleri kaldırdım)
import nltk
nltk.download('stopwords')
sw = stopwords.words('turkish')
sw  # bütün anlamsız kelimeleri görebiliriz.
abc='bir'

# Şimdi gelelim bunları çıkarmak. Pythonic yolu döngü yazmak, pandasla apply kullansakta aynı şeyi yaparız galdaş:)
# satırları boşluklarına göre split edip hepsinin içinde gezinirim. eğer varsa yoksa diye bakacağım.
# ne yapacağım biliyor musun? bütün satırları gezicem.apply gezme imkanı verdiya her birisinde list comp.
# lambda ile split etsin. Stop wordsde olmayanları seçelim olanlara dokunmayalım sonra joinleyelim.

df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))  # müthiş bir şey oldu
df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in abc))
# Rarewords yani nadir olanları çıkarmak isteyebilirim.
df['İçerik']
# geçici bir dataframe oluştururum.
temp_df = pd.Series(' '.join(
    df['İçerik']).split()).value_counts()  # yine split ile value counts'larına bakalım hangi kelimeden kaç kere geçmiş.
# silinecekleri ayarlayalım.
drops = temp_df[temp_df <= 1]

df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(
    x for x in str(x).split() if x not in drops))  # gez ve drops içinde olmayanları birleştir. Sonra da çıkar.
# dataframe olarak kalmayı istediğim için böyle yaptım.
# Tokenization
# cümleleri tokenlarına ayıracağım. Parçalayacağım. İstatistiksel anlamda birimleştirmek.
nltk.download("punkt")

df["İçerik"].apply(lambda x: TextBlob(x).words).head()  # ayırdım hepsini tek tek yine Allah razı olsun apply Lambdadan

# Lemmatization
# kelimeleri köklerine ayırma işlemi yapacağım.
temp_df

import zeyrek
analyzer = zeyrek.MorphAnalyzer()
print(analyzer.lemmatize('unsurlarınca'))
df['İçerik']

for i in df['İçerik']:
    print(analyzer.lemmatize(i))
df['İçerik'] = df['İçerik'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# 2. Text Visualization

# Terim Frekanslarının Hesaplanması

tf = df["İçerik"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["Kelime", "tf"]
tf.sort_values("tf", ascending=False)

df.sort_values(by='BeğeniSayısı',ascending=False)
df
# Barplot

tf[tf["tf"] > 20].plot.bar(x="Kelime", y="tf")
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
plt.show()



# Wordcloud
print(plt.rcParams)
print(plt.rcParams.get('figure.figsize'))
text = " ".join(i for i in df['İçerik'])

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.rcParams['figure.figsize'] = [20, 5]
plt.show()

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.rcParams['figure.figsize'] = [20, 10]
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")



#3-Sentiment Analysis

df["İçerik"].head()
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")

df.sort_values(by='Hashtag',ascending=False)


df_


