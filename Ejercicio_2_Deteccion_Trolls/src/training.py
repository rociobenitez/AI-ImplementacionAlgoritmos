import pandas as pd
import time
import mlflow
import subprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient


def train_and_log_model(data_path, n_stimators_list):

    data = pd.read_json(data_path, lines=True)
    data = data[['content', 'label']]

    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        data['content'], data['label'], test_size=0.2, random_state=42, stratify=data['label'])

    # Vectorización de los textos
    vectorizer = TfidfVectorizer(max_features=1000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # Launch MLflow UI asynchronously
    mlflow_ui_process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

    # Delay to allow MLflow UI to start
    time.sleep(5)

    # Start MLflow experiment
    mlflow.set_experiment('Pruebas en async')

    for i in n_stimators_list:
        with mlflow.start_run():
            print(f"Running for n_estimators = {i}")
            clf = RandomForestClassifier(n_estimators=i, min_samples_leaf=2,
                                         class_weight='balanced', random_state=42)
            preprocessor = Pipeline(steps=[('scaler', StandardScaler())])
            model = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('classifier', clf)])
            model.fit(x_train_vec, y_train)
            y_pred = model.predict(x_test_vec)
            accuracy = accuracy_score(y_test, y_pred)

            # Registro de métricas y parámetros
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_param('n_estimators', i)
            mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    data_path = 'balanced_dataset.json'
    n_stimators_list = [2, 10, 20, 30, 50, 80, 130]
    train_and_log_model(data_path, n_stimators_list)
