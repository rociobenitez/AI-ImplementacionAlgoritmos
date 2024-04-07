import re
import json
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, StandardOptions
from textblob import Word

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')


class PreprocessTextFn(beam.DoFn):
    def process(self, element):
        element['content'] = preprocess_text(element['content'])
        yield element


def preprocess_text(text):
    # Conversión a minúsculas y eliminación de caracteres no alfanuméricos
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenización y limpieza
    words = text.split()

    # Eliminación de stopwords y lematización
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word)
             for word in words if word not in stop_words]

    return ' '.join(words)


def run(argv=None, runner='DirectRunner', project='deteccion-trolls', temp_location='gs://cyber-trolls', region='europe-west9'):
    options = PipelineOptions(
        flags=argv,
        runner=runner,
        project=project,
        temp_location=temp_location,
        region=region,
    )

    if runner == 'DataflowRunner':
        options.view_as(SetupOptions).save_main_session = True
        options.view_as(StandardOptions).streaming = False

    with beam.Pipeline(options=options) as p:
        # Leer datos del archivo JSON en el entorno de Colab
        raw_data = (
            p
            | 'Leer Datos' >> beam.io.ReadFromText('cleaned_data.json')
            | 'Parsear JSON' >> beam.Map(json.loads)
        )

        processed_data = (
            raw_data
            | 'Preprocesar Texto' >> beam.ParDo(PreprocessTextFn())
        )

        # Convertir cada elemento procesado a una cadena JSON antes de escribir
        processed_data | 'Convertir a JSON' >> beam.Map(lambda x: json.dumps(
            x)) | 'Escribir Resultados' >> beam.io.WriteToText('preprocessed_data', file_name_suffix='.json')


if __name__ == '__main__':
    # Analizador de argumentos para permitir especificaciones de línea de comandos para el runner y las opciones de GCP
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', default='DirectRunner',
                        help='Runner to execute the pipeline.')
    parser.add_argument('--project', default='deteccion-trolls',
                        help='The GCP project to which the pipeline is associated.')
    parser.add_argument('--temp_location', default='gs://cyber-trolls',
                        help='Temporary location for Beam jobs.')
    parser.add_argument('--region', default='europe-west9',
                        help='Region in which the pipeline should run.')
    args, pipeline_args = parser.parse_known_args()

    run(
        argv=pipeline_args,
        runner=args.runner,
        project=args.project,
        temp_location=args.temp_location,
        region=args.region
    )
