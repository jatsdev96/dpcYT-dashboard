import streamlit as st
import pandas as pd
import math
from pathlib import Path
import csv
from itertools import islice
from youtube_comment_downloader import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from textblob import TextBlob
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='DPCYT dashboard',
    page_icon=':computer:', # This is an emoji shortcode. Could be a URL too.
)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :computer: DPCYT dashboard


The YouTube Comment Analyzer is a web application 
designed to receive a YouTube video link and analyze the 
comments associated with the video. 
'''

#Input for Youtube video url
url = st.text_input("URL video",)

if url:
    st.write("no vacio")
    # downloader = YoutubeCommentDownloader()
    # comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)

    # # Abre un archivo CSV en modo escritura
    # with open('comentarios.csv', 'w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["cid", "text", "time", "author", "channel", "votes", "replies", "photo", "heart", "reply", "time_parsed"])
    #     for comment in islice(comments, 4307): #4307
    #         writer.writerow([comment['cid'], comment['text'], comment['time'], comment['author'], comment['channel'], comment['votes'], comment['replies'], comment['photo'], comment['heart'], comment['reply'], comment['time_parsed']])

else:
    st.write("")

df = pd.read_csv('comentarios.csv')

df['new_column'] = 1
df.loc[df['votes'].str.contains('K'), 'new_column'] = 1000
df['votes'] = df['votes'].str.replace('K', '')
df['votes'] = df['votes'].astype(float)
df['votes'] = df['votes'] * df['new_column']
df.drop(columns=['new_column'], inplace=True)
df['votes'] = df['votes'].replace(0, np.nan)
df['votes'] = df['votes'].replace(' ', np.nan)

df['polaridad']=df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subobj']=df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity) 
st.write(df['polaridad'].describe())
st.write(df['subobj'].describe())

plt.figure(figsize=(15, 6))

sns.histplot(df['polaridad'], color='skyblue', label='Polaridad')
sns.histplot(df['subobj'], color='orange', label='Subjetividad')

plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.yscale('log')
plt.title('Distribución de Polaridad y Subjetividad de los comentarios')
plt.legend()
plt.ylim(-1, 10000)
st.pyplot(plt.show())

