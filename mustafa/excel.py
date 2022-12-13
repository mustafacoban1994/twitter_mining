import pandas as pd

df = pd.read_csv("tweetler.csv")
df.to_excel("suriyeli_tweets.xlsx")