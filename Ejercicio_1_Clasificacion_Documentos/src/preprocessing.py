import pandas as pd
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    """
    Preprocesa el texto aplicando una serie de transformaciones:
    - Convertir a minúsculas
    - Eliminar caracteres no alfabéticos
    - Eliminar stopwords
    - Aplicar lematización
    """
    # Verificar si el texto es una instancia de str
    if not isinstance(text, str):
        return ""

    # Convertir a minúsculas
    text = text.lower()
    # Eliminar símbolos y todo lo que no sea una letra
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenización
    words = text.split()
    # Eliminar stopwords y lematización
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(
        word) for word in words if word not in stopwords.words('english')]

    return ' '.join(words)
