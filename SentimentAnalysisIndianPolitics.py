import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import streamlit as st
from PIL import Image
st.title('Sentiment Analysis Over Indian Politics')
df=pd.read_csv('Narendra Modi_data.csv')
df1=pd.read_csv('Rahul Gandhi_data.csv')
df2=pd.read_csv('Arvind Kejriwal_data.csv')
df3=pd.read_csv('2022.csv')
df=pd.DataFrame(df)
df1=pd.DataFrame(df1)
df2=pd.DataFrame(df2)
df3=pd.DataFrame(df3)

# Function for removing @user name ,http urls ,# like some symbols
def remove_usernames_links(tweet):
    s2 = re.sub('http://\S+|https://\S+', '', tweet)
    s1=re.sub(r"#[a-zA-Z0-9\\n@_\s]+","",s2)
    return s1 

# Function for removing emoji

def remove_emoji(txt):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', txt)

#using stopwords
stp=stopwords.words('english')
print(stp)

#Functions for cleaning tweets and for polarity , subjectivity & Segmentation calculation

def TweetCleaning(tweet):
    link_removal=remove_usernames_links(tweet)
    emoji_removal=remove_emoji(link_removal)
    after_stopword_removal=' '.join(word for word in emoji_removal.split()if word not in stp)
    return after_stopword_removal

def calcPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity


def calcSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

def segmentation(tweet):
    if tweet > 0:
        return 'positive'
    elif tweet == 0 :
        return 'neutral'
    else:
        return 'negative'

# Creating New Columns

df["CleanedTweet"]=df["Tweet"].apply(TweetCleaning)
df['tPolarity']=df['CleanedTweet'].apply(calcPolarity)
df['tSubjectivity']=df['CleanedTweet'].apply(calcSubjectivity)
df['segmentation']=df['tPolarity'].apply(segmentation)

st.subheader("Narendra Modi")
image = Image.open('Modi.jpeg')
st.image(image)
st.subheader("Word Cloud of Narendra Modi")
fig=plt.figure(figsize=(30,10))
consolidated=' '.join(word for word in df ['CleanedTweet'])
wordCloud=WordCloud(width=400,height=200,random_state=20,max_font_size=119).generate(consolidated)
plt.imshow(wordCloud,interpolation='bilinear')
plt.axis('off')
st.pyplot(fig)