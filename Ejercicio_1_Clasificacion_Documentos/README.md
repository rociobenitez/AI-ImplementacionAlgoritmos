# Proyecto de Clasificación de Documentos

Este proyecto está enfocado en la clasificación automática de documentos utilizando técnicas de procesamiento de lenguaje natural y aprendizaje automático.


## Estructura del Proyecto 🗂️

Proyecto_Clasificacion_Documentos/
- data/                     # Carpeta para almacenar los conjuntos de datos
    - raw/                   # Datos brutos, sin procesar
    - processed/             # Datos preprocesados listos para el modelado
- notebooks/                # Jupyter notebooks para exploración y análisis
    - EDA.ipynb             # Análisis Exploratorio de Datos (Exploratory Data Analysis)
    - main_script.ipynb     # Notebook principal para el entrenamiento y modelado
- src/                      # Código fuente del proyecto
    - preprocessing.py      # Script para preprocesar los datos
    - train_model.py        # Script para entrenar y evaluar el modelo
    - utils.py              # Script con funciones de utilidad
- models/                   # Modelos entrenados
- mlruns/                   # Directorio para los registros de MLflow
- README.md                 # Documentación y guíadel proyecto


## Instalación 👩🏼‍💻

Para ejecutar este proyecto se necesita Python 3.8+ y se recomienda usar un entorno virtual para gestionar las dependencias.

```bash
# Clonar el repositorio del proyecto
git clone https://github.com/tu-usuario/Proyecto_Clasificacion_Documentos.git
cd Proyecto_Clasificacion_Documentos

# Instalar las dependencias
pip install -r requirements_ejercicio1.txt
```