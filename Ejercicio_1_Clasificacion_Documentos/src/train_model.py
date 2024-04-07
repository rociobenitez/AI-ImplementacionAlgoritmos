from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def train_and_tune(x_train, y_train, n_jobs=-1, cv=5, random_state=42):
    """
    Entrena y optimiza un modelo RandomForest utilizando GridSearchCV.

    Parámetros:
    - x_train: Características de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - n_jobs: Número de trabajos para correr en paralelo.
    - cv: Número de pliegues de validación cruzada.
    - random_state: Semilla para la reproducibilidad de los resultados.

    Retorna:
    - El modelo entrenado con la mejor combinación de parámetros.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(random_state=random_state)),
    ])

    parameters = {
        'clf__n_estimators': [100, 200],
        'clf__min_samples_leaf': [1, 2],
        'clf__class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=n_jobs, cv=cv)
    grid_search.fit(x_train, y_train)

    print("Mejores parámetros encontrados:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_


def train_and_evaluate(x_train, y_train, x_val, y_val, min_samples_leaf=2, n_estimators=200, random_state=42):
    """
    Entrena un modelo RandomForest y lo evalúa en un conjunto de validación.

    Parámetros:
    - x_train, y_train: Datos de entrenamiento.
    - x_val, y_val: Datos de validación.
    - vectorizer_params: Parámetros para el TfidfVectorizer.
    - classifier_params: Parámetros para RandomForestClassifier.

    Retorna:
    - Un diccionario con el modelo y las métricas de evaluación.
    """
    if not isinstance(x_train, pd.Series) or not isinstance(x_val, pd.Series):
        raise ValueError("x_train y x_val deben ser Series de pandas.")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(
            class_weight='balanced',
            min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators,
            random_state=random_state))
    ])

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_val)

    # Métricas de evaluación
    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, average='weighted')
    print(classification_report(y_val, predictions))

    # Matriz de confusión
    cm = confusion_matrix(y_val, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return {'model': pipeline, 'accuracy': accuracy, 'f1_score': f1}
