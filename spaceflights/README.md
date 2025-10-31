# COVID-19 Chile: Análisis y Modelado Predictivo con Machine Learning

## Descripción del Proyecto

Proyecto de análisis exhaustivo y modelado predictivo de datos COVID-19 en Chile (2020-2022) utilizando metodologías de Machine Learning y el framework Kedro. Implementa las primeras 3 fases de la metodología CRISP-DM con orquestación en Airflow, versionado con DVC y despliegue en Docker para generar insights epidemiológicos y capacidades predictivas reproducibles.

**Estudiantes:** Claudia Murúa - [Nombre Estudiante 2]  
**Curso:** MLY0100 - Machine Learning  
**Institución:** [Tu Universidad]  
**Fecha:** Octubre 2025

---

## 🎯 Objetivos

### Objetivos de Negocio
- Analizar la evolución temporal de la pandemia COVID-19 en Chile
- Identificar patrones geográficos y diferencias regionales
- Caracterizar olas pandémicas y períodos críticos
- Evaluar indicadores epidemiológicos clave
- Generar insights para optimización de políticas sanitarias

### Objetivos de Machine Learning
- Desarrollar modelos predictivos para casos futuros a corto plazo (7-14 días)
- Crear sistema de clasificación de períodos de alta/baja transmisión
- Implementar detección automática de tendencias epidemiológicas
- Modelar volatilidad y riesgo de saturación hospitalaria
- Optimizar feature engineering para máximo poder predictivo

---

## 📊 Datasets

El proyecto utiliza 3 datasets principales de COVID-19 Chile:

| Dataset | Período | Registros | Descripción |
|---------|---------|-----------|-------------|
| chile_completo_covid_2020.csv | 2020 | 33,253 | Datos COVID-19 Chile año 2020 |
| chile_completo_covid_2021.csv | 2021 | 36,330 | Datos COVID-19 Chile año 2021 |
| chile_completo_covid_2022.csv | 2022 | 29,610 | Datos COVID-19 Chile año 2022 |

**Total:** 99,193 registros de 363 ubicaciones únicas cubriendo el período 2020-2022

---

## 🏗️ Arquitectura del Proyecto

### Stack Tecnológico
```
┌─────────────────────────────────────────────────────────────┐
│                    Apache Airflow (Orquestación)            │
│                    http://localhost:8080                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Kedro Pipelines (Procesamiento ML)             │
│  data_processing → data_science → reporting                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 DVC (Versionado de Datos)                   │
│  Datasets + Modelos + Métricas + Visualizaciones            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Docker (Contenedores)                      │
│  Ambiente aislado y reproducible                            │
└─────────────────────────────────────────────────────────────┘
```

### Metodología CRISP-DM

#### Fase 1: Business Understanding - ✅ Completada
- Definición del problema de negocio
- Objetivos específicos y criterios de éxito
- Evaluación de recursos y riesgos
- Plan detallado del proyecto

#### Fase 2: Data Understanding - ✅ Completada
- EDA exhaustivo (univariado, bivariado, multivariado)
- Análisis de calidad de datos
- Identificación de patrones temporales y geográficos
- Validación de integridad de datos

#### Fase 3: Data Preparation - ✅ Completada
- Limpieza diferenciada por dataset
- Feature engineering avanzado
- Integración de múltiples fuentes
- Preparación de targets para ML

### Pipelines Kedro

| Pipeline | Descripción | Nodos | Funcionalidad |
|----------|-------------|-------|---------------|
| data_processing (dp) | Feature engineering avanzado | 6 | Creación de 80+ variables predictivas |
| data_science (ds) | Entrenamiento y evaluación | 7 | Modelado de 51 modelos ML con GridSearchCV |
| reporting (rp) | Visualización y reportes | 4 | Generación de insights y gráficos |

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- **Python 3.8+** (recomendado 3.11)
- **Git** para control de versiones
- **Docker Desktop (con WSL 2):** Esencial para Airflow
- **8GB RAM** recomendado para procesamiento
- **5GB espacio libre** en disco

### Configuración de Memoria de WSL 2 (Windows)

⚠️ **IMPORTANTE:** Si usas WSL 2, aumenta la memoria asignada editando `.wslconfig` en `C:\Users\<TuUsuario>\`:
```ini
[wsl2]
memory=8GB   
processors=4
```

Luego ejecuta en PowerShell (admin):
```powershell
wsl --shutdown
```

Y reinicia Docker Desktop.

### Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/ClauMurua/data_covid_ML.git
cd data_covid_ML/spaceflights
```

2. **Crear ambiente virtual:**
```bash
python -m venv kedro-env
source kedro-env/bin/activate  # Linux/Mac
kedro-env\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar DVC:**
```bash
dvc pull  # Descargar datos versionados
```

5. **Configurar estructura de datos:**
```bash
mkdir -p data/01_raw
# Los archivos CSV ya deberían estar después de dvc pull
```

---

## 🔄 DVC - Versionado de Datos y Modelos

### ✅ Pipeline Configurado
```bash
CSV files → data_processing → data_science → reporting
```

### Stages del Pipeline

#### 1. data_processing
- **Input:** 3 archivos CSV (Chile COVID 2020-2022)
- **Output:** regression_dataset.csv, classification_dataset.csv
- **Propósito:** Procesa datos crudos y crea 80+ features

#### 2. data_science
- **Input:** Datasets procesados
- **Output:** 
  - Modelos entrenados:
    - `trained_regression_models.pkl` (11MB)
    - `trained_classification_models.pkl` (137MB)
- **Metrics:** 
  - regression_metrics.json
  - classification_metrics.json
  - best_models_report.json
  - feature_importance_analysis.json
- **Propósito:** Entrena 51 modelos ML con GridSearchCV (k=5)

#### 3. reporting
- **Input:** Métricas JSON
- **Output:** 3 visualizaciones PNG (~920KB total)
  - regression_comparison.png
  - classification_comparison.png
  - summary_dashboard.png
- **Metrics:** tuning_summary.json
- **Propósito:** Genera reportes y gráficos comparativos

### Archivos Versionados (148MB+ total)
```
data/
├── 01_raw/
│   ├── chile_completo_covid_2020.csv.dvc
│   ├── chile_completo_covid_2021.csv.dvc
│   └── chile_completo_covid_2022.csv.dvc
├── 06_models/
│   ├── trained_regression_models.pkl (11MB)
│   └── trained_classification_models.pkl (137MB)
├── 07_model_output/
│   ├── regression_metrics.json
│   ├── classification_metrics.json
│   ├── best_models_report.json
│   └── feature_importance_analysis.json
└── 08_reporting/
    ├── tuning_summary.json
    ├── regression_comparison.png
    ├── classification_comparison.png
    └── summary_dashboard.png
```

### Comandos DVC
```bash
# Ver pipeline
dvc dag

# Ver estado
dvc status

# Reproducir todo el pipeline
dvc repro

# Ver métricas
dvc metrics show

# Push/Pull de artefactos
dvc push
dvc pull
```

### Reproducibilidad con DVC

Para reproducir el proyecto completo desde cero:
```bash
# 1. Clonar repositorio
git clone https://github.com/ClauMurua/data_covid_ML.git
cd data_covid_ML/spaceflights

# 2. Pull de datos versionados
dvc pull

# 3. Reproducir pipeline
dvc repro
```

---

## ⚙️ Ejecución del Proyecto

### Opción 1: Pipelines Locales (Kedro)
```bash
# Ejecutar todos los pipelines
kedro run

# Ejecutar pipeline específico
kedro run --pipeline=dp  # data_processing
kedro run --pipeline=ds  # data_science
kedro run --pipeline=rp  # reporting

# Ejecutar desde un nodo específico
kedro run --from-nodes=train_regression_models

# Ejecutar hasta un nodo específico
kedro run --to-nodes=prepare_modeling_data
```

### Opción 2: Orquestación con Airflow (Docker)

#### Levantar el Entorno de Airflow
```bash
cd spaceflights
docker-compose -f docker-compose.airflow.yml up --build -d
```

- **Airflow UI:** `http://localhost:8080`
- **Usuario:** `admin`
- **Contraseña:** `admin`

#### Ejecutar el DAG

1. Abrir `http://localhost:8080`
2. Buscar DAG: `covid_ml_pipeline`
3. Activar y ejecutar (trigger manualmente o esperar schedule)

#### Monitorear Ejecución
```bash
# Ver logs del scheduler
docker logs -f spaceflights-airflow-scheduler-1

# Ver logs del webserver
docker logs -f spaceflights-airflow-webserver-1

# Acceder al contenedor
docker exec -it spaceflights-airflow-scheduler-1 bash
```

#### Detener Airflow
```bash
docker-compose -f docker-compose.airflow.yml down
```

### Opción 3: Análisis Interactivo
```bash
# Jupyter con contexto Kedro
kedro jupyter notebook

# Visualización de pipelines
kedro viz
```

---

## 🤖 Modelos de Machine Learning

### Targets Identificados

#### Problemas de Regresión (6 targets):
1. `target_confirmed_next_7d_log` - Predicción logarítmica de casos en 7 días
2. `target_rolling_change_7d` - Cambio en casos (rolling 7 días)
3. `target_growth_rate_next_14d` - Tasa de crecimiento en 14 días
4. `target_risk_level` - Nivel de riesgo numérico
5. `target_trend_direction` - Dirección de tendencia numérica
6. `target_change_pct_7d` - Cambio porcentual en 7 días

#### Problemas de Clasificación (6 targets):
1. `target_confirmed_next_7d_log` - Clasificación de nivel de casos
2. `target_rolling_change_7d` - Clasificación de cambio
3. `target_growth_rate_next_14d` - Clasificación de crecimiento
4. `target_risk_level` - Nivel de riesgo categórico
5. `target_trend_direction` - Dirección de tendencia
6. `target_high_transmission` - Alta/baja transmisión

### Algoritmos Implementados

#### Modelos de Regresión (5 algoritmos):
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Total:** 30 modelos entrenados (6 targets × 5 algoritmos)

#### Modelos de Clasificación (3-4 algoritmos):
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- SVM (Support Vector Machine)

**Total:** 21 modelos entrenados

### Hiperparámetros y Tuning

**Método:** GridSearchCV con Cross-Validation (k=5 folds)

**Parámetros optimizados:**
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
- Gradient Boosting: `learning_rate`, `n_estimators`, `max_depth`
- Ridge/Lasso: `alpha`
- Logistic Regression: `C`, `penalty`
- SVM: `C`, `kernel`, `gamma`

**Tiempo de entrenamiento total:** ~3 horas

### Resultados de Performance

#### Regresión (mejores modelos por target):
- R² promedio: **0.85 - 0.95**
- RMSE normalizado: **< 0.15**
- Mejor algoritmo: **Random Forest** y **Gradient Boosting**

#### Clasificación (mejores modelos por target):
- F1-Score promedio: **0.75 - 0.90**
- Accuracy: **> 0.80**
- Mejor algoritmo: **Random Forest**

Ver resultados detallados en: `data/08_reporting/tuning_summary.json`

---

## 🔧 Feature Engineering

### Variables Creadas (80+ features)

#### Features Temporales:
- Variables lag (1, 3, 7, 14, 21 días)
- Rolling statistics (medias móviles, desviaciones, min, max)
- Variables estacionales (componentes sin/cos)
- Tendencias y aceleración (derivadas temporales)
- Diferencias temporales

#### Features Epidemiológicas:
- Tasas de letalidad diaria y acumulada
- Ratios de crecimiento temporal
- Índices de volatilidad
- Comparaciones regionales
- Métricas de cambio relativo

#### Transformaciones:
- **StandardScaler** para variables numéricas
- **MinMaxScaler** para variables cíclicas
- **Label Encoding** para variables categóricas
- **Log transformation** para distribuciones asimétricas

---

## 📁 Estructura del Proyecto
```
data_covid_ML/
├── spaceflights/
│   ├── .dvc/                           # Configuración DVC
│   │   ├── config                      # Remote storage config
│   │   └── cache/                      # Cache local de DVC
│   ├── conf/
│   │   ├── base/
│   │   │   ├── catalog.yml             # Definición de datasets
│   │   │   ├── parameters.yml          # Parámetros ML
│   │   │   └── logging.yml             # Configuración logs
│   │   └── local/
│   ├── data/
│   │   ├── 01_raw/                     # Datos originales (*.csv.dvc)
│   │   ├── 02_intermediate/            # Datos procesados intermedios
│   │   ├── 03_primary/                 # Datos limpios primarios
│   │   ├── 04_feature/                 # Features generados
│   │   ├── 05_model_input/             # Inputs para modelos
│   │   ├── 06_models/                  # Modelos entrenados (148MB)
│   │   ├── 07_model_output/            # Métricas y evaluación
│   │   └── 08_reporting/               # Reportes y visualizaciones
│   ├── notebooks/
│   │   ├── 01_business_understanding.ipynb
│   │   ├── 02_data_understanding.ipynb
│   │   └── 03_data_preparation.ipynb
│   ├── src/spaceflights/
│   │   └── pipelines/
│   │       ├── data_processing/
│   │       │   ├── nodes.py            # Lógica de procesamiento
│   │       │   └── pipeline.py         # Definición pipeline
│   │       ├── data_science/
│   │       │   ├── nodes.py            # Entrenamiento modelos
│   │       │   └── pipeline.py
│   │       └── reporting/
│   │           ├── nodes.py            # Generación reportes
│   │           └── pipeline.py
│   ├── dags/
│   │   └── covid_ml_dag.py             # DAG de Airflow
│   ├── docker-compose.airflow.yml      # Configuración Docker
│   ├── dvc.yaml                        # Pipeline DVC
│   ├── dvc.lock                        # Estado pipeline DVC
│   ├── requirements.txt                # Dependencias Python
│   ├── README.md                       # Este archivo
│   └── .gitignore                      # Archivos ignorados
```

---

## 📊 Resultados Principales

### Calidad de Datos
- **Score de calidad:** 92.0/100 (Excelente)
- **Completitud:** Mayor al 95% en variables clave
- **Cobertura temporal:** 727 días consecutivos (2020-2022)
- **Ubicaciones procesadas:** 363 regiones/comunas

### Performance de Modelos
- **Modelos de regresión:** R² superior a 0.85
- **Modelos de clasificación:** F1-Score superior a 0.80
- **Validación temporal:** Train/Val/Test split implementado
- **Feature importance:** Analizada y documentada

### Visualizaciones Generadas
1. **regression_comparison.png** - Comparación de R² entre modelos
2. **classification_comparison.png** - Comparación de F1-Score
3. **summary_dashboard.png** - Dashboard con 4 paneles:
   - Top 8 modelos de regresión
   - Top 8 modelos de clasificación
   - Distribución de R² scores
   - Distribución de F1 scores

---

## 📦 Dependencias del Sistema

### Librerías Principales
```
kedro>=0.18.0
kedro-datasets>=3.0.0
kedro-airflow>=0.7.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
fastapi>=0.100.0
jupyter>=1.0.0
apache-airflow>=2.7.0
dvc>=3.0.0
```

Ver lista completa en `requirements.txt`

---

## 🔐 Consideraciones Técnicas

### Reproducibilidad
- ✅ Control de versiones completo con Git
- ✅ Versionado de datos con DVC
- ✅ Parametrización en archivos de configuración
- ✅ Seeds fijos para resultados reproducibles (random_state=42)
- ✅ Documentación exhaustiva de decisiones
- ✅ Docker para ambiente aislado

### Buenas Prácticas Implementadas
- ✅ Pipelines modulares y reutilizables (Kedro)
- ✅ Separación clara entre datos raw y procesados
- ✅ Logging detallado de todos los procesos
- ✅ Validación automática de calidad de datos
- ✅ Type hints en código Python
- ✅ Docstrings en funciones críticas
- ✅ Git commits descriptivos y atómicos

### Seguridad
- Credenciales en variables de entorno (no en código)
- `.gitignore` configurado correctamente
- Datos sensibles versionados con DVC (no en Git)

---

## 🐛 Troubleshooting

### Error: "Memoria Agotada" en Docker
**Solución:** Aumenta memoria de WSL 2 a 8GB en `.wslconfig`

### Error: "DVC: output already specified"
**Solución:** Elimina archivos `.dvc` duplicados con `dvc remove`

### Error: "Kedro pipeline not found"
**Solución:** Verifica que estás en el directorio `spaceflights/`

### Error: "Airflow DAG no aparece"
**Solución:** Verifica logs con `docker logs spaceflights-airflow-scheduler-1`

### Error: "GridSearchCV toma demasiado tiempo"
**Solución:** Reduce `cv_folds` en `parameters.yml` de 5 a 3

---

## 👥 Contribuciones

### Metodología de Desarrollo
- Trabajo colaborativo con control de versiones (Git)
- Code review antes de integración
- Documentación de decisiones técnicas en commits
- Testing de funcionalidades críticas

### Equipo
- **Claudia Murúa:** Data Engineering, MLOps (DVC, Airflow, Docker)
- **[Estudiante 2]:** Data Science, Feature Engineering, Modelado

---

## 📄 Licencia

Este proyecto está desarrollado bajo licencia MIT para fines académicos.

---

## 📞 Contacto

- **Repositorio:** https://github.com/ClauMurua/data_covid_ML
- **Documentación:** Ver notebooks en `/notebooks/`
- **Issues:** Reportar en el repositorio de GitHub

---

## 🎓 Referencias

- [Kedro Documentation](https://docs.kedro.org/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC Documentation](https://dvc.org/doc)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)

---

**Última actualización:** Octubre 2025  
**Versión:** 1.0.0