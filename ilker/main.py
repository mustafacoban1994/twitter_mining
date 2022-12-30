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

async def bokye:
    ####yazdığımız fonksiyonun içereceği kelimeleri tanımlıyoruz#####
    anyOfWords(
        '(göçmen OR göç OR suriyeli OR mülteci OR afganli OR afganlı OR düzensizgöç OR '
        ' uygur OR doguturkistan OR pakistanlı OR pakistanli OR sınırdışı OR hudut OR yabancı OR uyruklu OR turist OR doğutürkistan OR #göç OR #göçmen OR #suriyeli OR sığınmacı)')
    df = pd.DataFrame(liste,
                      columns=['Tarih/Zaman', 'KullanıcıAdı', 'İçerik', 'Url', 'Hashtag', 'BeğeniSayısı', 'PaylaşımSayısı'])
    df['İçerik']


    df['Url']
    df.head()
    #bir tane daha kopyasını oluşturalım ki bu dataframe in ki yaptığımız yanlışlarda tekrar tekrar çekmek zorunda kalmayalım#####
    df_=df.copy()
    df_

    ####şimdi daplikasyona düşen varsa yani yinelenen varsa onları kaldırsın####
    #df["İçerik"]=df["İçerik"].drop_duplicates(inplace=True)

    #EN SON KONUŞTUKLARIMIZIN ÜZERİNE 3 ADET CASE YAPACAĞIZ.
    df
    #1-En Çok Beğeniye Göre Tweet Screens

    filterwarnings('ignore')  # bazı hataları ignore ediyorum.
    pd.set_option('display.max_columns', None)  # max kolon göstersin.
    pd.set_option('display.width', 20000)  # genişliği ayarladım.
    pd.set_option('display.float_format', lambda x: '%.2f' % x)  # virgülden sonraki iki basamağı alsın.

    df.head() #bi bakalım neler var.
    df.info() #birde veri tiplerine bakalım.

    # Normalizing Case Folding (büyük küçük harfleri standartlaştırırız.)
    df['İçerik']
    df['İçerik'] = df['İçerik'].str.lower()

    # Punctuations (noktalama işareti gördüğünde boşluk ile değiştirelim)
    df['İçerik'] = df['İçerik'].str.replace('[^\w\s]', '')


    # regular expression
    # Numbers (sayıları uçurdum)
    df['İçerik'] = df['İçerik'].str.replace('\d', '')  # sayıları \d ile buluruz.


    # Stopwords (bağlaçlar falan anlamsız cümleleri kaldırdım)
    df['İçerik']
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    #sw=stopwords.words('turkish')
    #sw
    listes=pd.read_excel("ilker/turkish-stopwords.xlsx") #kendi oluşturduğum stopwordsu okutuyorum.
    listem=listes.values.tolist() #listeye çeviriyorum.
    listem=sum(listem,[]) #liste içinde ki listenin parçalanması
    df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in listem)) #listenin içindeki kelimelerden varsa çıkarıyorum. Edat bağlaç gibi gereksiz kelimeler.


    # geçici bir dataframe oluştururum.
    temp_df = pd.Series(' '.join(
        df['İçerik']).split()).value_counts()
    len(temp_df)
    sum(temp_df)
    kelimeler_=temp_df.nsmallest(len(temp_df)-50)
    print(kelimeler_)
    drops_=kelimeler_
    # silinecekleri ayarlayalım.
    drops = temp_df[temp_df <= 300]

    df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(
        x for x in str(x).split() if x not in drops_))  # gez ve drops içinde olmayanları birleştir. Sonra da çıkar.


    # dataframe olarak kalmayı istediğim için böyle yaptım.
    # Tokenization
    # cümleleri tokenlarına ayıracağım. Parçalayacağım. İstatistiksel anlamda birimleştirmek.

    nltk.download("punkt")
    df["İçerik"].apply(lambda x: TextBlob(x).words).head()
    analyzer = zeyrek.MorphAnalyzer()

    for i in df['İçerik']:
        print(analyzer.lemmatize(i))
        df['İçerik'] = df['İçerik'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # Terim Frekanslarının Hesaplanması

    tf = df["İçerik"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["Kelime", "tf"]
    tf.sort_values(by='tf',ascending=False)


    df.sort_values(by='BeğeniSayısı',ascending=False)
    df['Url'].to_excel("C:\\Users\\i.onur.isik\\Desktop\\tsl\\a.xlsx")

    # Barplot

    tf[tf["tf"] > 200].plot.bar(x="Kelime", y="tf")
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 2
    plt.show()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print (df)

    df['Url']
    df

bokye();
