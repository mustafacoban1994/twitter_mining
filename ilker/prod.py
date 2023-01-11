####Kütüphanelerin Yüklenmesi####
import imageio as imageio
import snscrape.modules.twitter as twitter
import numpy as np
import pandas as pd
from warnings import filterwarnings

import tweetcapture
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
import os
import datetime
from datetime import datetime, timedelta
import openpyxl
import nltk
import zeyrek
import schedule
import time
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from PIL import Image
import numpy as np

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
        print(tweet.hashtags)
        print(tweet.likeCount)
        print(tweet.retweetCount)
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
nltk.download('stopwords')
sw=stopwords.words('turkish')

##stopwords
listes=pd.read_excel("D:\\twitter\\twitter_mining\\ilker\\turkish-stopwords.xlsx") #kendi oluşturduğum stopwordsu okutuyorum.
listem=listes.values.tolist() #listeye çeviriyorum.
listem=sum(listem,[])

joinedlist=[*sw,*listem]#nltk dan gelen kelimelerle birleştirdim. Yeni bir liste kurdum.
df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in joinedlist)) #listenin içindeki kelimelerden varsa çıkarıyorum. Edat bağlaç gibi gereksiz kelimeler.
df_['İçerik'] = df_['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in joinedlist))

####Rare Words#####Nadir Kelimeleri bulalım######
#geçici bir dataframe oluşturdum
temp_df = pd.Series(' '.join(df['İçerik']).split()).value_counts() #bu df'nin içerisine wordlerin countlarını aldım.
#len(temp_df)
#sum(temp_df)
drops = temp_df[temp_df <= 2] #frekans değeri 2 ve altında olanları yeni bir listeye attım.
df['İçerik'] = df['İçerik'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops)) #sonrada bunları uçurdum.
df['İçerik']=df['İçerik'].str.lower()



#####Görselleştirme######
###BARPLOT####

frekanslar=df['İçerik'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
frekanslar.columns=['Kelime', 'Sayı']
frekanslar.sort_values('Sayı', ascending=False)
sirali_frekanslar = frekanslar[frekanslar['Sayı']>=100]
sirali_frekanslar.sort_values('Sayı',ascending=False)
sirali_frekanslar.plot.bar(x='Kelime', y='Sayı')
plt.subplots_adjust(bottom=0.4, top=0.99)
plt.yticks(color='orange')
plt.savefig("D:\\twitter\\twitter_mining\\ilker\\Barplot")
plt.show()

###Screen Shot

df_screenshot=df[df['KullanıcıAdı']!='behzatuygur'].sort_values('BeğeniSayısı',ascending=False)
df_screenshot=df_screenshot[df_screenshot['BeğeniSayısı']>1000].sort_values('BeğeniSayısı',ascending=False)
df_screenshot['Url']


directory = "ss"
parent_dir = "D:\\twitter\\twitter_mining\\ilker"
path = os.path.join(parent_dir, directory)
os.mkdir(path)
print("Directory '% s' created" % directory)

for i in df_screenshot['Url']:
    a = str('cmd /c "tweetcapture {0}''"').format((i))
    print(a)
    os.system(a)
    print("bitti")
print("Hepsi bitti")

####WordCloud

text = " ".join(i for i in sirali_frekanslar['Kelime'])

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.rcParams['figure.figsize'] = [20, 10]
plt.savefig("D:\\twitter\\twitter_mining\\ilker\Wordcloud\\wordcloud.png")
plt.show()


wordcloud = WordCloud(
                      max_font_size=50,
                      max_words=100,
                      background_color="red").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.rcParams['figure.figsize'] = [20, 10]
plt.axis("off")
plt.savefig("D:\\twitter\\twitter_mining\\ilker\Wordcloud\\wordcloudwhite.png")
plt.show()


wordcloud = WordCloud(
                      max_font_size=50,
                      max_words=100,
                      background_color="red").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.rcParams['figure.figsize'] = [20, 10]
plt.axis("off")
plt.savefig("D:\\twitter\\twitter_mining\\ilker\Wordcloud\\wordcloudred.png")
plt.show()




