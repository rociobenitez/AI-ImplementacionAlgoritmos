import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter, defaultdict
from itertools import chain

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

import nltk
from nltk import ngrams, bigrams, pos_tag, word_tokenize
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

nltk_download('stopwords')
nltk_download('punkt')
nltk_download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))


def check_missing_values(data):
    print("Valores faltantes en cada columna:")
    print(data.isnull().sum())


def set_plot_style():
    """
    Establece el estilo minimalista para todas las gráficas.
    """
    sns.set(style="whitegrid", palette="muted", color_codes=True)
    sns.set_context("talk")
    plt.rcParams['axes.edgecolor'] = 'gray'
    plt.rcParams['axes.linewidth'] = 0.7
    plt.rcParams['xtick.color'] = 'gray'
    plt.rcParams['ytick.color'] = 'gray'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_category_distribution(data, column_name):
    """
    Visualiza la distribución de categorías en una columna específica.

    Args:
    - data (DataFrame): El DataFrame que contiene los datos.
    - column_name (str): El nombre de la columna categórica a visualizar.
    """
    set_plot_style()  # Establece el estilo de la gráfica
    plt.figure(figsize=(10, 4))
    sns.countplot(y=column_name, data=data, color='royalblue')
    plt.title(f'Distribución de {column_name}', fontsize=14)
    plt.xlabel('Frecuencia', fontsize=12)
    plt.ylabel(column_name.capitalize(), fontsize=12)
    sns.despine()  # Quita los ejes superior y derecho
    plt.show()


def calculate_text_length(df, text_column='text'):
    """
    Añade una nueva columna al DataFrame con la longitud de cada texto.
    """
    df['text_length'] = df[text_column].apply(len)


def plot_text_length_distribution(df, text_length_column='text_length'):
    """
    Visualiza la distribución de las longitudes de texto usando un boxplot.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 3))
    sns.boxplot(x=df[text_length_column], color='royalblue')
    plt.title('Distribución de la Longitud de los Textos')
    plt.xlabel('Longitud del Texto')
    sns.despine()  # Quita los ejes superior y derecho para un estilo más limpio
    plt.show()


def get_most_common_words(df, text_column='text', num_words=20):
    """
    Obtiene y devuelve las palabras más comunes en la columna de texto.
    """
    all_words = ' '.join(df[text_column]).split()
    filtered_words = [
        word for word in all_words if word not in ENGLISH_STOP_WORDS]
    common_words = Counter(filtered_words).most_common(num_words)
    return common_words


def plot_most_common_words(common_words):
    """
    Visualiza las palabras más comunes.
    """
    sns.barplot(x=[count for word, count in common_words], y=[
                word for word, count in common_words], color='royalblue')
    plt.title('Palabras Más Comunes')
    plt.xlabel('Frecuencia')
    plt.ylabel('Palabras')
    sns.despine()
    plt.show()


def generate_wordcloud(text, max_words=100):
    """
    Genera y visualiza una nube de palabras.
    """
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', max_words=max_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nube de Palabras para Texto del Dataset')
    sns.despine()
    plt.show()


def vocabulary_diversity(texts):
    """Calcula y visualiza la diversidad del vocabulario."""
    words = [word for text in texts for word in word_tokenize(
        text.lower()) if word.isalpha()]
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words) if words else 0
    print(
        f"Vocabulario total: {len(words)}, Vocabulario único: {len(unique_words)}, Diversidad de vocabulario: {diversity_score}")


def plot_word_length_distribution(texts):
    """Plotea la distribución de la longitud de las palabras."""
    word_lengths = [len(word) for text in texts for word in word_tokenize(
        text) if word.isalpha() and word not in stop_words]
    plt.figure(figsize=(10, 6))
    sns.histplot(word_lengths, bins=30, color='royalblue')
    plt.title('Distribución de la Longitud de las Palabras')
    plt.xlabel('Longitud de la Palabra')
    plt.ylabel('Frecuencia')
    sns.despine()
    plt.show()


def common_ngrams(texts, n=2, num_ngrams=20):
    """Identifica y visualiza los n-gramas más comunes."""
    n_grams = ngrams(' '.join(texts).split(), n)
    common_ngrams = Counter(n_grams).most_common(num_ngrams)
    for gram_count in common_ngrams:
        print(f"{gram_count[0]}: {gram_count[1]}")


def plot_top_ngrams(corpus, n=None, ngrams=(1,1)):
    set_plot_style()
    vec = CountVectorizer(ngram_range=ngrams).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    top_words_freq = words_freq[:n]
    top_words, top_freqs = zip(*top_words_freq)
    set_plot_style()
    plt.figure(figsize=(10,4))
    sns.barplot(x=list(top_freqs), y=list(top_words), color='royalblue')
    plt.title(f'Top {n} {ngrams}-grams')
    sns.despine()
    plt.show()


def sentiment_analysis(texts):
    """Realiza un análisis de sentimientos básico y visualiza los resultados."""
    polarity = [TextBlob(text).sentiment.polarity for text in texts]
    plt.figure(figsize=(10, 6))
    sns.histplot(polarity, bins=30, color='royalblue')
    plt.title('Distribución de la Polaridad de Sentimientos')
    plt.xlabel('Polaridad')
    plt.ylabel('Frecuencia')
    sns.despine()
    plt.show()


def sentiment_analysis_sub(texts):
    """Realiza un análisis de sentimientos básico y visualiza los resultados."""
    polarity = [TextBlob(text).sentiment.subjectivity for text in texts]
    plt.figure(figsize=(10, 6))
    sns.histplot(polarity, bins=30, color='royalblue')
    plt.title('Distribución de la Subjetividad de Sentimientos')
    plt.xlabel('Subjetividad')
    plt.ylabel('Frecuencia')
    sns.despine()
    plt.show()


def pos_tagging(texts):
    """Realiza el etiquetado de partes de la oración (POS) y visualiza las frecuencias."""
    pos_tags = [pos for text in texts for word,
                pos in pos_tag(word_tokenize(text))]
    pos_counts = Counter(pos_tags)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(pos_counts.keys()), y=list(
        pos_counts.values()), color='royalblue')
    plt.title('Frecuencia de Partes de la Oración (POS)')
    plt.xlabel('POS')
    plt.ylabel('Frecuencia')
    sns.despine()
    plt.show()


# Para la frecuencia de términos específicos


def term_frequency(texts, term):
    count = sum(text.lower().count(term) for text in texts)
    print(f"El término '{term}' aparece {count} veces.")


def plot_word_co_occurrence(texts, target_word, num_words=25):
    """
    Calcula y visualiza las palabras que más frecuentemente co-ocurren con la palabra objetivo.

    Args:
    - texts (list of str): Una lista de cadenas de texto en las que buscar co-ocurrencias.
    - target_word (str): La palabra objetivo para la cual buscar co-ocurrentes.
    - num_words (int): Número de palabras co-ocurrentes más frecuentes para mostrar.

    Muestra:
    - Un gráfico de barras que representa las palabras que más frecuentemente co-ocurren con la palabra objetivo.
    """
    # La lista de stopwords debe estar descargada
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    co_occurrences = Counter()  # Inicializa un contador para almacenar las co-ocurrencias

    for text in texts:
        # Tokeniza el texto
        tokens = word_tokenize(text.lower())

        # Encuentra bigramas que incluyan la palabra objetivo y que no sean stopwords
        for bigram in bigrams(tokens):
            if target_word in bigram:
                co_word = bigram[1] if bigram[0] == target_word else bigram[0]
                if co_word not in stop_words:
                    co_occurrences[co_word] += 1

    most_common_co_occurrences = co_occurrences.most_common(num_words)
    words, frequencies = zip(*most_common_co_occurrences)

    set_plot_style()
    plt.figure(figsize=(12, 9))
    plt.barh(words, frequencies, color='royalblue')
    plt.xlabel('Frecuencia')
    plt.ylabel('Co-ocurrencias')
    plt.title(
        f'Palabras que más frecuentemente co-ocurren con "{target_word}"')
    plt.gca().invert_yaxis()  # Invierte el eje y para tener la palabra más frecuente arriba
    sns.despine()
    plt.show()


# Para Topic Modeling con LDA
def lda_topic_modeling(texts, n_components=5):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_components)
    lda.fit(X)
    for i, topic in enumerate(lda.components_):
        terms = [vectorizer.get_feature_names_out()[index]
                 for index in topic.argsort()[-10:]]
        print(f"Top términos en el tema #{i}: {terms}")


def find_collocations(texts, n_best=20):
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(word_tokenize(' '.join(texts)))
    collocations = finder.nbest(bigram_measures.pmi, n_best)
    print(f"Top {n_best} collocations: {collocations}")
