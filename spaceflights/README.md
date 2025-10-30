COVID-19 Chile: Análisis y Modelado Predictivo con Machine Learning
Descripción del Proyecto
Proyecto de análisis exhaustivo y modelado predictivo de datos COVID-19 en Chile (2020-2022) utilizando metodologías de Machine Learning y el framework Kedro. Implementa las primeras 3 fases de la metodología CRISP-DM para generar insights epidemiológicos y capacidades predictivas.
Estudiantes: [Nombre Estudiante 1] - [Nombre Estudiante 2]
Curso: MLY0100 - Machine Learning
Institución: [Tu Universidad]
Fecha: Septiembre 2025
Objetivos
Objetivos de Negocio

Analizar la evolución temporal de la pandemia COVID-19 en Chile
Identificar patrones geográficos y diferencias regionales
Caracterizar olas pandémicas y períodos críticos
Evaluar indicadores epidemiológicos clave
Generar insights para optimización de políticas sanitarias

Objetivos de Machine Learning

Desarrollar modelos predictivos para casos futuros a corto plazo (7-14 días)
Crear sistema de clasificación de períodos de alta/baja transmisión
Implementar detección automática de tendencias epidemiológicas
Modelar volatilidad y riesgo de saturación hospitalaria
Optimizar feature engineering para máximo poder predictivo

Datasets
El proyecto utiliza 3 datasets principales de COVID-19 Chile:
DatasetPeríodoRegistrosDescripciónchile_completo_covid_2020.csv202033,253Datos COVID-19 Chile año 2020chile_completo_covid_2021.csv202136,330Datos COVID-19 Chile año 2021chile_completo_covid_2022.csv202229,610Datos COVID-19 Chile año 2022
Total: 99,193 registros de 363 ubicaciones únicas cubriendo el período 2020-2022
Arquitectura del Proyecto
Metodología CRISP-DM
Fase 1: Business Understanding - Completada

Definición del problema de negocio
Objetivos específicos y criterios de éxito
Evaluación de recursos y riesgos
Plan detallado del proyecto

Fase 2: Data Understanding - Completada

EDA exhaustivo (univariado, bivariado, multivariado)
Análisis de calidad de datos
Identificación de patrones temporales y geográficos
Validación de integridad de datos

Fase 3: Data Preparation - Completada

Limpieza diferenciada por dataset
Feature engineering avanzado
Integración de múltiples fuentes
Preparación de targets para ML

Pipelines Kedro
PipelineDescripciónNodosFuncionalidaddata_engineeringValidación, limpieza e integración9Procesamiento inicial de datosdata_processingFeature engineering avanzado6Creación de variables predictivasdata_scienceEntrenamiento y evaluación7Modelado de Machine LearningreportingVisualización y reportes4Generación de insights
Instalación y Configuración
Prerrequisitos

Python 3.8 o superior
Git para control de versiones
*   **Docker Desktop (con WSL 2):** Esencial para el despliegue con Airflow. Asegúrate de que esté configurado para usar el backend de WSL 2.
*   **Configuración de Memoria de WSL 2:** Si usas WSL 2, es crucial aumentar la memoria asignada para Docker. Edita el archivo `.wslconfig` en `C:\Users\<TuNombreDeUsuario>\` y configura `memory` a `8GB` o más (e.g., `memory=8GB`). Después de guardar, ejecuta `wsl --shutdown` en PowerShell (admin) y reinicia Docker Desktop.
8GB RAM recomendado para procesamiento
2GB espacio libre en disco

Instalación

Clonar el repositorio:

bashgit clone https://github.com/ClauMurua/data_covid_ML.git
cd data_covid_ML

Crear ambiente virtual:

bashpython -m venv kedro-env
source kedro-env/bin/activate  # Linux/Mac
kedro-env\Scripts\activate     # Windows

Instalar dependencias:

bashpip install -r requirements.txt

Configurar estructura de datos:

bashmkdir -p data/01_raw
# Colocar archivos CSV en data/01_raw/
Ejecución del Proyecto
Pipelines Completos
bash# Ejecutar todos los pipelines
kedro run

# Ejecutar pipeline específico
kedro run --pipeline=data_engineering
kedro run --pipeline=data_processing
kedro run --pipeline=data_science
kedro run --pipeline=reporting
Análisis Interactivo
bash# Jupyter con contexto Kedro
kedro jupyter notebook

# Visualización de pipelines
kedro viz

## Despliegue con Airflow (MLOps)

Para orquestar y programar los pipelines de Kedro, este proyecto utiliza Apache Airflow, ejecutado con Docker Compose. Esto permite un despliegue y monitoreo robusto de todo el flujo de trabajo de ML.

### Prerrequisitos Específicos para Airflow/Docker

*   **Docker Desktop (con WSL 2):** Asegúrate de tener Docker Desktop instalado y configurado para usar el backend de WSL 2 en Windows.
*   **Configuración de Memoria de WSL 2:** Es crucial que asignes suficiente memoria a WSL 2 para evitar errores de "Memoria Agotada" durante la ejecución de los pipelines. Edita (o crea) el archivo `.wslconfig` en `C:\Users\<TuNombreDeUsuario>\` y añade:

```ini
[wsl2]
memory=8GB   ; Asigna al menos 8GB de RAM a WSL 2
processors=4 ; Ajusta al número de núcleos de CPU que desees
```

    Después de guardar el archivo, apaga WSL con `wsl --shutdown` en PowerShell (como administrador) y reinicia Docker Desktop.

### Levantar el Entorno de Airflow

Desde el directorio `spaceflights` del proyecto, puedes levantar todos los servicios de Airflow (webserver, scheduler, postgres) usando Docker Compose:

```bash
cd spaceflights
docker-compose -f docker-compose.airflow.yml up --build -d
```

*   El flag `--build` es importante para reconstruir las imágenes de Docker y asegurar que todas las dependencias (`kedro-datasets[pandas-csvdataset]`, `fastapi`, etc.) estén instaladas dentro de los contenedores de Airflow.
*   El flag `-d` ejecuta los servicios en segundo plano.

Una vez que los servicios estén levantados, podrás acceder a la interfaz de usuario de Airflow en `http://localhost:8080` (usuario: `admin`, contraseña: `admin`).

### Ejecutar el Pipeline de Kedro en Airflow

El DAG de Airflow (`covid_ml_dag.py`) está configurado para ejecutar los pipelines de Kedro en secuencia. Para ejecutar el DAG:

1.  Abre la interfaz de usuario de Airflow en `http://localhost:8080`.
2.  Busca el DAG llamado `covid_ml_pipeline`.
3.  Activa el DAG (si no está activo) y luego, desde la vista de lista o la vista gráfica, puedes disparar una ejecución manual.

Los pipelines de Kedro se ejecutarán con los nombres `dp` (data processing), `ds` (data science) y `rp` (reporting) tal como están definidos en `pipeline_registry.py`.

## Notebooks Disponibles

01_business_understanding.ipynb - Comprensión del negocio
02_data_understanding.ipynb - Análisis exploratorio
03_data_preparation.ipynb - Preparación de datos

Modelos de Machine Learning
Targets Identificados
Problemas de Regresión:

target_cases_next_7d: Predicción de casos confirmados en próximos 7 días
target_growth_rate_14d: Tasa de crecimiento de casos en 14 días

Problemas de Clasificación:

target_high_transmission: Clasificación de períodos de alta transmisión
target_risk_level: Nivel de riesgo epidemiológico (Bajo/Medio/Alto)
target_trend_direction: Dirección de tendencia epidemiológica

Algoritmos Implementados
Modelos de Regresión:

Linear Regression
Ridge Regression
Random Forest Regressor
Gradient Boosting Regressor

Modelos de Clasificación:

Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier
Support Vector Machine

Feature Engineering
Variables Creadas (80+ features)
Features Temporales:

Variables lag (1, 3, 7, 14, 21 días)
Rolling statistics (medias móviles, desviaciones)
Variables estacionales (componentes sin/cos)
Tendencias y aceleración

Features Epidemiológicas:

Tasas de letalidad diaria y acumulada
Ratios de crecimiento temporal
Índices de volatilidad
Comparaciones regionales

Transformaciones:

StandardScaler para variables numéricas
MinMaxScaler para variables cíclicas
Label encoding para variables categóricas

Estructura del Proyecto
data_covid_ML/
├── airflow/
├── dags/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── parameters.yml
│   │   └── logging.yml
│   └── local/
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   └── 03_data_preparation.ipynb
├── src/spaceflights/
│   └── pipelines/
│       ├── data_engineering/
│       ├── data_processing/
│       ├── data_science/
│       └── reporting/
├── requirements.txt
├── README.md
└── .gitignore
Resultados Principales
Calidad de Datos

Score de calidad: 92.0/100 (Excelente)
Completitud: Mayor al 95% en variables clave
Cobertura temporal: 727 días consecutivos
Ubicaciones procesadas: 363 regiones/comunas

Performance de Modelos

Modelos de regresión: R² superior a 0.85
Modelos de clasificación: F1-Score superior a 0.80
Validación temporal implementada
Feature importance analizada

Dependencias del Sistema
Librerías Principales
kedro>=0.18.0
kedro-datasets>=3.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
fastapi>=0.100.0
jupyter>=1.0.0
Consideraciones Técnicas
Reproducibilidad

Control de versiones completo con Git
Parametrización en archivos de configuración
Seeds fijos para resultados reproducibles
Documentación exhaustiva de decisiones

Buenas Prácticas

Pipelines modulares y reutilizables
Separación clara entre datos raw y procesados
Logging detallado de todos los procesos
Validación automática de calidad de datos

Contribuciones
Metodología de Desarrollo

Trabajo colaborativo con control de versiones
Code review antes de integración
Documentación de decisiones técnicas
Testing de funcionalidades críticas

Equipo

Estudiante 1: Data Engineering, Business Understanding
Estudiante 2: Data Science, Feature Engineering, Modelado

Licencia
Este proyecto está desarrollado bajo licencia MIT para fines académicos.
Contacto

Repositorio: https://github.com/ClauMurua/data_covid_ML
Documentación: Ver notebooks en directorio /notebooks/
Issues: Reportar en el repositorio de GitHub