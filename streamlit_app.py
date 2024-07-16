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

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
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
    st.write("vacio")

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

