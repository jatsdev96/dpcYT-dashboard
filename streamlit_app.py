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
from nltk.stem.snowball import SnowballStemmer
from wordcloud import WordCloud
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='DPCYT dashboard',
    page_icon=':computer:', # This is an emoji shortcode. Could be a URL too.
)
#st.set_option('deprecation.showPyplotGlobalUse', False)
def csv_sin_informacion(ruta_archivo):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(ruta_archivo)

        # Verificar si el DataFrame está vacío (sin filas ni columnas)
        if df.empty:
            return True

        # Verificar si todas las celdas están vacías o contienen solo espacios en blanco
        if df.applymap(lambda x: str(x).strip()).replace('', pd.NA).isna().all().all():
            return True

        return False
    except pd.errors.EmptyDataError:
        # Capturar el error en caso de que el archivo esté completamente vacío
        return True

def eliminar_contenido_archivo(ruta_archivo):
    # Abrir el archivo en modo de escritura y escribir una cadena vacía
    with open(ruta_archivo, 'w') as archivo:
        archivo.write('')
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :computer: DPCYT dashboard


The YouTube Comment Analyzer is a web application 
designed to receive a YouTube video link and analyze the 
comments associated with the video. 
'''
eliminar_contenido_archivo('comentarios.csv')
#Input for Youtube video url
#https://www.youtube.com/watch?v=kZaucITWv00&t=10s
url = st.text_input("URL video",)

if url:
    #st.write("no vacio")
    st.video(url)
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)

    # Abre un archivo CSV en modo escritura
    with open('comentarios.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["cid", "text", "time", "author", "channel", "votes", "replies", "photo", "heart", "reply", "time_parsed"])
        for comment in islice(comments, 4307): #4307
            writer.writerow([comment['cid'], comment['text'], comment['time'], comment['author'], comment['channel'], comment['votes'], comment['replies'], comment['photo'], comment['heart'], comment['reply'], comment['time_parsed']])

    if not csv_sin_informacion('comentarios.csv'):
        #st.write("HAY INFOOOO")

        #Preparación de datos
        df = pd.read_csv('comentarios.csv')


        df['new_column'] = 1
        df.loc[df['votes'].str.contains('K'), 'new_column'] = 1000
        df['votes'] = df['votes'].str.replace('K', '')
        df['votes'] = df['votes'].astype(float)
        df['votes'] = df['votes'] * df['new_column']
        df.drop(columns=['new_column'], inplace=True)
        df['votes'] = df['votes'].replace(0, np.nan)
        df['votes'] = df['votes'].replace(' ', np.nan)

        st.write("Descripción de datos:")
        st.write("Estadísticos básicos:")
        st.write(df.describe())
        st.write("Estadísticos de autores:")
        st.write(df['author'].describe())
        st.write("===============================")
        st.write(df['author'].value_counts())

        st.write("Top 10 autores:")
        top_10_autores = df['author'].value_counts().nlargest(10)
        # Crea la figura de la gráfica de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top_10_autores.index, top_10_autores.values)
        ax.set_xlabel('Autores')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Top 10 Autores')
        ax.set_xticks(range(len(top_10_autores.index)))
        ax.set_xticklabels(top_10_autores.index, rotation=45, ha='right')
        plt.tight_layout()

        # Muestra la gráfica en Streamlit
        st.pyplot(fig)

        st.write("Top 10 autores con más votos:")
        top_autores_votos = df.groupby('author')['votes'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top_autores_votos.index, top_autores_votos.values)
        ax.set_xlabel('Autores')
        ax.set_ylabel('Total de Votos')
        ax.set_title('Top 10 Autores con Más Votos')
        ax.set_xticks(range(len(top_autores_votos.index)))
        ax.set_xticklabels(top_autores_votos.index, rotation=45, ha='right')
        plt.tight_layout()

        # Muestra la gráfica en Streamlit
        st.pyplot(fig)

        st.write("Top 10 autores con más réplicas:")
        top_autores_replicas = df.groupby('author')['replies'].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top_autores_replicas.index, top_autores_replicas.values)
        ax.set_xlabel('Autores')
        ax.set_ylabel('Total de Réplicas')
        ax.set_title('Top 10 Autores con Más Réplicas')
        ax.set_xticks(range(len(top_autores_replicas.index)))
        ax.set_xticklabels(top_autores_replicas.index, rotation=45, ha='right')
        plt.tight_layout()

        # Muestra la gráfica en Streamlit
        st.pyplot(fig)

        st.write("Estadísticos de votos:")
        st.write(df['votes'].describe())
        st.write(df['votes'].value_counts())
        st.write("Estadísticos de replicas:")
        st.write(df['replies'].describe())
        st.write(df['replies'].value_counts())
        st.write("Distribución conjunta de Votos y Réplicas:")
        sns.set_theme(style="ticks")

        # Crea la figura del jointplot
        g = sns.jointplot(x='votes', y='replies', data=df, kind='scatter', palette="rocket", height=10)
        g.set_axis_labels('Votos', 'Réplicas')
        g.fig.suptitle('Distribución Conjunta de Votos y Réplicas', fontsize=14)

        # Ajusta la escala del eje x
        g.ax_joint.set_xscale('log')

        # Ajusta la figura para que el título no se sobreponga con los ejes
        plt.subplots_adjust(top=0.95)

        # Muestra la gráfica en Streamlit
        st.pyplot(g.fig)

        st.write("Identificación y procesado del procesado de texto:")
        df['text'] = df['text'].astype(str)
        ComYT = df['text']
        ComYT[0:10]



        # Supongamos que df['text'] es tu Serie de pandas con los párrafos
        ComYT = df['text'].astype(str)

        # Inicializa una cadena vacía para almacenar todo el texto
        ComYTcom = ''

        for parrafo in ComYT:
            ComYTcom += parrafo + '\r'

        ComYT_sp = ComYTcom.split("\n")
        #ComYT_sp[:20]

        ComYT_filtrado1 = list(filter(None, ComYT_sp))
        #ComYT_filtrado1[:20]

        #Aqui inicia el procesado de datos

        nltk.download('stopwords')
        nltk.download('punkt')

        def eliminar_emojis(texto):
            emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticones
                                u"\U0001F300-\U0001F5FF"  # símbolos & pictogramas
                                u"\U0001F680-\U0001F6FF"  # transporte & símbolos de mapas
                                u"\U0001F1E0-\U0001F1FF"  # banderas (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
            return emoji_pattern.sub(r'', texto)

        def tokens(txt):
            ComYT_limpio = eliminar_emojis(txt)
            ComYT_limpio = re.sub(r'[^\w\s]', '', txt.lower())

            stop_words = set(stopwords.words('spanish'))## filtramos stordswords
            tokens = word_tokenize(ComYT_limpio)

            stemmer = SnowballStemmer('spanish') # filtrado de Stemmer /desidencia
            ComYT_filtrado = [stemmer.stem(word) for word in tokens if word not in stop_words]
            ComYT_filtrado = [word for word in tokens if word not in stop_words]
            return ComYT_filtrado

        map(tokens, ComYT_filtrado1)
        #map

        map_ComYT = list(map(tokens, ComYT_filtrado1))

        tokens = []
        for token in map_ComYT:
            tokens.extend(token)

        df = pd.DataFrame(tokens)
        conteo_frecuencias = df.value_counts()
        conteo_frecuencias

        df_frecuencias = conteo_frecuencias.to_frame()
        df_frecuencias

        df_frecuencias.reset_index(inplace = True)
        df_frecuencias.columns = ['token', 'conteo']
        df_frecuencias

        df_frecuencias.head(30)

        st.write("Top 20 Palabras más Frecuentes:")
        st.write(df_frecuencias.head(20))

        # Crea la figura de la gráfica de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df_frecuencias.iloc[:20]['token'], df_frecuencias.iloc[:20]['conteo'])

        # Personaliza las etiquetas del eje X
        plt.xticks(rotation=45, ha='right')

        ax.set_xlabel('Palabras')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Top 20 Palabras más Frecuentes')
        plt.tight_layout()

        # Muestra la gráfica en Streamlit
        st.pyplot(fig)

        # Genera la nube de palabras
        wordcloud = WordCloud().generate(' '.join(tokens))

        # Configura la visualización de la nube de palabras
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title("Nube de Palabras")
        plt.tight_layout()

        # Muestra la imagen en Streamlit
        st.pyplot(fig)

        st.write("Análisis de sentimientos:")

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


        # Asegúrate de que los datos necesarios están descargados
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        nltk.download('stopwords')
        stop_words = set(stopwords.words('spanish'))


        # Elimina emojis
        df_cleaned = df.copy()
        df_cleaned['text'] = df_cleaned['text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)

        # # Carga las stopwords en español
        # nltk.download('stopwords')
        # stop_words = set(stopwords.words('spanish'))

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


    else:
        st.write("No hay")    
else:
    st.write("")

#st.write("Análisis de sentimiento:")
#eliminar_contenido_archivo('comentarios.csv')
#st.write(csv_sin_informacion('comentarios.csv'))

