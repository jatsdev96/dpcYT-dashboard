import streamlit as st
import pandas as pd
#import regex as re
import re
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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='DPCYT dashboard',
    page_icon=':computer:', # This is an emoji shortcode. Could be a URL too.
)
#st.set_option('deprecation.showPyplotGlobalUse', False)

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
    # st.video("url")
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

st.write("Análisis de sentimiento:")

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

# plt.figure(figsize=(15, 6))

# sns.histplot(df['polaridad'], color='skyblue', label='Polaridad')
# sns.histplot(df['subobj'], color='orange', label='Subjetividad')

# plt.xlabel('Valores')
# plt.ylabel('Frecuencia')
# plt.yscale('log')
# plt.title('Distribución de Polaridad y Subjetividad de los comentarios')
# plt.legend()
# plt.ylim(-1, 10000)
# plt.show()

#st.write(df.head())

fig, ax = plt.subplots(figsize=(15, 6))

# Crear los histogramas
sns.histplot(df['polaridad'], color='skyblue', label='Polaridad', ax=ax)
sns.histplot(df['subobj'], color='orange', label='Subjetividad', ax=ax)

# Configurar etiquetas y título
ax.set_xlabel('Valores')
ax.set_ylabel('Frecuencia')
ax.set_yscale('log')
ax.set_title('Distribución de Polaridad y Subjetividad de los comentarios')
ax.legend()
ax.set_ylim(-1, 10000)

# Mostrar la figura en Streamlit
st.pyplot(fig)
#st.bar_chart(fig)


# Configura el tema de Seaborn
sns.set_theme(style="ticks")

# Crea la figura del jointplot
g = sns.jointplot(x='votes', y='polaridad', data=df, kind='scatter', color='red', height=10)
g.set_axis_labels('Votos', 'polaridad')
g.fig.suptitle('Distribución Conjunta de Votos y polaridad de los comentarios', fontsize=14)

# Ajusta la escala del eje x
g.ax_joint.set_xscale('log')

# Ajusta la figura para que el título no se sobreponga con los ejes
plt.subplots_adjust(top=0.95)

# Muestra la gráfica en Streamlit
st.pyplot(g.fig)

# Configura el tema de Seaborn
sns.set_theme(style="ticks")

# Crea la figura del jointplot
g = sns.jointplot(x='votes', y='subobj', data=df, kind='scatter', color='blue', height=10)
g.set_axis_labels('Votos', 'objetividad')
g.fig.suptitle('Distribución Conjunta de Votos y objetividad', fontsize=14)

# Ajusta la escala del eje x
g.ax_joint.set_xscale('log')

# Ajusta la figura para que el título no se sobreponga con los ejes
plt.subplots_adjust(top=0.95)

# Muestra la gráfica en Streamlit
st.pyplot(g.fig)

st.write("Primeras diez filas del DataFrame:")
st.write(df[0:10])

st.write("Nuevo procesado:")

# Elimina emojis
df_cleaned = df.copy()
df_cleaned['text'] = df_cleaned['text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)

# Carga las stopwords en español
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Función para eliminar stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(filtered_tokens)

# Aplica el preprocesamiento a la columna 'text' en el nuevo DataFrame
df_cleaned['processed_text'] = df_cleaned['text'].apply(preprocess_text)

# Tokeniza cada comentario y guarda los tokens en una nueva columna 'tokens'
df_cleaned['tokens'] = df_cleaned['processed_text'].apply(word_tokenize)

# Ahora puedes acceder a la columna 'tokens' en el DataFrame 'df_cleaned'

df_cleaned = df_cleaned.drop(columns=['channel','cid','time', 'photo','heart', 'reply',	'time_parsed','author','processed_text' ])
st.write("Primeras diez filas del DataFrame:")
st.write(df_cleaned[0:10])

df_cleaned['NumCom'] = range(1, len(df_cleaned) + 1)
df_comentarios = df_cleaned
df_cleaned = df_cleaned.drop(columns=['text','votes','replies', 'polaridad', 'subobj'])
st.write("Primeras diez filas del DataFrame:")
st.write(df_cleaned[0:10])
#df_cleaned[0:5]

df_tokens_cap = df_cleaned.explode(column='tokens')
df_tokens_cap