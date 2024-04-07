from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

import nltk
from nltk import word_tokenize
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

nltk_download('stopwords')
nltk_download('punkt')
nltk_download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))


def set_plot_style():
    """
    Configura un estilo visual minimalista para todas las gráficas generadas con seaborn y matplotlib,
    eliminando los bordes innecesarios y ajustando colores y tamaños para una presentación limpia y profesional.
    """
    sns.set(style="white", palette="muted", color_codes=True)
    sns.set_context("talk")
    plt.rcParams['axes.edgecolor'] = 'gray'
    plt.rcParams['axes.linewidth'] = 0.7
    plt.rcParams['xtick.color'] = 'gray'
    plt.rcParams['ytick.color'] = 'gray'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.figsize'] = (10, 6)
    # Eliminar los ejes superior y derecho
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def plot_class_distribution(df, class_column):
    """
    Genera una gráfica de barras que muestra la distribución de las clases dentro del DataFrame proporcionado.
    """
    set_plot_style()
    sns.countplot(x=class_column, data=df)
    plt.title('Distribución de Clases')
    plt.xlabel('Clase')
    plt.ylabel('Frecuencia')
    plt.show()


def plot_message_length(df, text_column):
    """
    Crea un histograma para explorar la longitud de los textos dentro del DataFrame.
    """
    set_plot_style()
    df['text_length'] = df[text_column].apply(len)
    sns.histplot(df['text_length'], bins=50, color='royalblue')
    plt.title('Distribución de la Longitud de los Mensajes')
    plt.xlabel('Longitud del Mensaje')
    plt.ylabel('Frecuencia')
    plt.show()


def plot_word_frequency(df, text_column, n_most_common=20):
    """
    Identifica y visualiza las palabras más frecuentes en el conjunto de datos.
    Permite identificar posibles stop words o ruido.
    """
    set_plot_style()
    words = ' '.join(df[text_column]).split()
    counter = Counter(words)
    most_common_words = counter.most_common(n_most_common)
    words_df = pd.DataFrame(most_common_words, columns=['word', 'frequency'])
    sns.barplot(x='frequency', y='word', data=words_df, color="royalblue")
    plt.title('Palabras Más Comunes')
    plt.xlabel('Frecuencia')
    plt.ylabel('Palabra')
    plt.show()


def plot_ngrams(df, text_column, n=2, n_most_common=20):
    """
    Calcula y muestra los n-gramas más comunes en el conjunto de datos.
    """
    set_plot_style()
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    ngrams = vectorizer.fit_transform(df[text_column])
    sum_ngrams = ngrams.sum(axis=0)
    words_freq = [(word, sum_ngrams[0, idx])
                  for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    ngrams_df = pd.DataFrame(words_freq[:n_most_common], columns=[
                             'ngram', 'frequency'])
    sns.barplot(x='frequency', y='ngram', data=ngrams_df, color="royalblue")
    plt.title(f'Top {n}-gramas Más Comunes')
    plt.xlabel('Frecuencia')
    plt.ylabel('N-grama')
    plt.show()


def plot_sentiment_distribution(df, text_column):
    """
    Grafica la distribución de la polaridad de los sentimientos de los textos contenidos en el DataFrame.
    Utiliza TextBlob para calcular la polaridad y luego genera un histograma con seaborn.
    """
    set_plot_style()
    polarity = df[text_column].map(
        lambda text: TextBlob(text).sentiment.polarity)
    sns.histplot(polarity, bins=50, color='royalblue')
    plt.title('Distribución de la Polaridad de Sentimientos')
    plt.xlabel('Polaridad')
    plt.ylabel('Frecuencia')
    plt.show()


def plot_subjectivity_distribution(df, column='subjectivity', bins=30, color='royalblue', figsize=(8, 6)):
    """
    Grafica la distribución de la subjetividad de los textos en el DataFrame.
    La subjetividad es una medida de TextBlob que indica cuán subjetivo u objetivo es un texto.
    """
    plt.figure(figsize=figsize)
    set_plot_style()
    plt.hist(df[column], bins=bins, color=color)
    plt.title('Distribución de la Subjetividad')
    plt.xlabel('Subjetividad')
    plt.ylabel('Frecuencia')
    plt.show()


def plot_boxplot_by_class(df, column='subjectivity', class_column='label', color='royalblue', figsize=(8, 6)):
    """
    Grafica un diagrama de cajas que muestra la distribución de la subjetividad separada por clases.
    Permite comparar la subjetividad de los mensajes clasificados como trolls versus los que no lo son.
    """
    plt.figure(figsize=figsize)
    set_plot_style()
    sns.boxplot(x=class_column, y=column, data=df, color=color)
    plt.title('Subjetividad en Mensajes Troll vs No Troll')
    plt.xlabel('Clase')
    plt.ylabel('Subjetividad')
    plt.show()


def plot_common_words_by_class(df, class_column='label', text_column='content', n_words=20):
    """
    Grafica las palabras más comunes encontradas en mensajes clasificados como trolls y no trolls.
    Utiliza CountVectorizer para extraer estas palabras y luego las visualiza en barras.
    """
    troll_df = df[df[class_column] == 1][text_column]
    not_troll_df = df[df[class_column] == 0][text_column]

    vectorizer = CountVectorizer(stop_words='english')

    # Calcula las palabras más comunes para trolls
    troll_words = vectorizer.fit_transform(troll_df)
    troll_word_counts = pd.DataFrame(
        troll_words.toarray(), columns=vectorizer.get_feature_names_out())
    troll_common_words = troll_word_counts.sum(
    ).sort_values(ascending=False).head(n_words)

    # Calcula las palabras más comunes para no trolls
    not_troll_words = vectorizer.fit_transform(not_troll_df)
    not_troll_word_counts = pd.DataFrame(
        not_troll_words.toarray(), columns=vectorizer.get_feature_names_out())
    not_troll_common_words = not_troll_word_counts.sum(
    ).sort_values(ascending=False).head(n_words)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    with sns.axes_style("white", rc={"axes.edgecolor": 'gray', "axes.linewidth": 0.7}):
        sns.despine(left=True)

        sns.barplot(ax=axes[0], x=troll_common_words.values,
                    y=troll_common_words.index, palette="Reds_d", hue=troll_common_words.index)
        axes[0].set_title('Palabras más comunes en mensajes trolls')
        axes[0].set_xlabel('Frecuencia')
        axes[0].set_ylabel('Palabra')

        sns.barplot(ax=axes[1], x=not_troll_common_words.values,
                    y=not_troll_common_words.index, palette="Blues_d", hue=not_troll_common_words.index)
        axes[1].set_title('Palabras más comunes en mensajes no trolls')
        axes[1].set_xlabel('Frecuencia')
        axes[1].set_ylabel('Palabra')

    plt.tight_layout()
    plt.show()


def vocabulary_diversity(texts):
    """
    Calcula y muestra la diversidad del vocabulario de un conjunto de textos.
    """
    words = [word for text in texts for word in word_tokenize(
        text.lower()) if word.isalpha()]
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words) if words else 0
    print(
        f"Vocabulario total: {len(words)}, Vocabulario único: {len(unique_words)}, Diversidad de vocabulario: {diversity_score}")


def calculate_subjectivity(text):
    """
    Calcula la subjetividad de un texto utilizando TextBlob.
    """
    return TextBlob(text).sentiment.subjectivity


def plot_collocations(texts, n_best=20):
    """
    Identifica y grafica las collocations más frecuentes en el conjunto de textos.
    Las collocations son pares de palabras que ocurren juntas más frecuentemente de lo normal.
    """
    # Encuentra las collocations en los textos
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(word_tokenize(' '.join(texts)))

    # Filtra las collocations y obtiene las mejores n
    collocations = finder.nbest(bigram_measures.pmi, n_best)

    # Prepara los datos para el DataFrame
    colloc_strings = [' '.join(colloc) for colloc in collocations]
    colloc_freqs = [finder.ngram_fd[colloc] for colloc in collocations]

    # Crea un DataFrame para la visualización
    colloc_df = pd.DataFrame(
        {'collocation': colloc_strings, 'frequency': colloc_freqs})
    colloc_df.sort_values('frequency', ascending=False, inplace=True)

    # Visualiza las collocations
    plt.figure(figsize=(10, 6))
    set_plot_style()
    sns.barplot(x='frequency', y='collocation', data=colloc_df,
                palette='Blues_d', hue='collocation')
    plt.xlabel('Frecuencia')
    plt.ylabel('Collocation')
    plt.title(f'Top {n_best} Collocations')
    plt.show()
