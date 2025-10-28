# Configuración del Proyecto COVID-19 Chile

Este directorio contiene toda la configuración del proyecto de análisis y modelado predictivo de COVID-19 en Chile.

## Estructura de Configuración
```
conf/
├── base/                          # Configuración compartida del proyecto
│   ├── catalog.yml               # Definición de todos los datasets
│   ├── parameters.yml            # Parámetros generales del proyecto
│   ├── parameters_data_processing.yml  # Parámetros de feature engineering
│   ├── parameters_data_science.yml     # Parámetros de modelado ML
│   ├── parameters_reporting.yml        # Parámetros de visualización
│   └── logging.yml               # Configuración de logging
└── local/                        # Configuración local (no en Git)
    └── credentials.yml           # Credenciales y configuración sensible
```

## Archivos de Configuración

### catalog.yml
Define todos los datasets del proyecto organizados por capas de datos:
- **01_raw**: Datos originales COVID-19 (2020, 2021, 2022)
- **02_intermediate**: Datos procesados parcialmente
- **03_primary**: Datos limpios e integrados
- **04_feature**: Features para Machine Learning
- **05_model_input**: Datos listos para entrenamiento
- **06_models**: Modelos entrenados guardados
- **07_model_output**: Resultados y métricas de modelos
- **08_reporting**: Reportes y visualizaciones generadas

### parameters.yml
Parámetros generales del proyecto:
- Configuración de limpieza de datos
- Umbrales de calidad
- Métodos de detección de outliers
- Targets de ML principales

### parameters_data_processing.yml
Parámetros específicos para feature engineering:
- Variables lag y rolling windows
- Transformaciones (log, scaling)
- Población por región para cálculos per cápita
- Configuración de feature engineering temporal

### parameters_data_science.yml
Parámetros para modelado de Machine Learning:
- Configuración de modelos (Random Forest, Gradient Boosting, etc.)
- Hiperparámetros de algoritmos
- División train/val/test
- Criterios de éxito del modelo
- Métricas de evaluación

### parameters_reporting.yml
Parámetros para generación de reportes:
- Configuración de visualizaciones
- Eventos clave COVID-19 a marcar en gráficos
- Colores y estilos de gráficos
- KPIs principales del dashboard
- Períodos de comparación

## Uso de Parámetros

Los parámetros se cargan automáticamente en los pipelines usando la sintaxis:
```python
# En pipeline.py
node(
    func=mi_funcion,
    inputs=["dataset", "params:feature_engineering"],
    outputs="output",
    name="mi_nodo"
)
```

## Modificación de Parámetros

### Para Desarrollo
Modifica los archivos en `conf/base/` directamente. Estos cambios aplican a todo el equipo.

### Para Configuración Local
Crea archivos en `conf/local/` con el mismo nombre. Los valores en `local/` sobrescriben los de `base/`.

Ejemplo:
```yaml
# conf/local/parameters_data_science.yml
modeling:
  test_size: 0.25  # Sobrescribe el valor de base
```

## Configuración Sensible

**IMPORTANTE**: NUNCA subas credenciales o configuración sensible a Git.

Usa `conf/local/credentials.yml` para:
- API keys
- Contraseñas de bases de datos
- Tokens de acceso
- Rutas locales específicas

Este archivo está en `.gitignore` y no se versiona.

## Versionado de Configuración

Los cambios en `conf/base/` deben:
1. Documentarse en commits descriptivos
2. Ser revisados por el equipo
3. Actualizarse en conjunto con código relacionado
4. Mantener compatibilidad con pipelines existentes

## Variables de Entorno

Variables de entorno disponibles:
- `KEDRO_ENV`: Ambiente actual (local, dev, prod)
- `KEDRO_DISABLE_TELEMETRY`: Deshabilitar telemetría

## Criterios de Éxito (de la Rúbrica)

Los parámetros incluyen los criterios de éxito definidos en la rúbrica:
- **Regresión**: RMSE < 500, R² > 0.85
- **Clasificación**: F1-Score > 0.80, Precision > 0.85

Estos se usan en la evaluación automática de modelos.

## Troubleshooting

### Error: "No module named 'kedro.io.data_catalog'"
- Actualiza kedro: `pip install --upgrade kedro`

### Error: "Parameters not found"
- Verifica que el archivo exists en `conf/base/`
- Revisa el nombre en `pipeline.py`: debe ser `params:nombre_seccion`

### Parámetros no se aplican
- Verifica que la función recibe el parámetro
- Revisa logs: `kedro run --log-level=DEBUG`

---
**Proyecto COVID-19 Chile - Machine Learning**  
*Framework: Kedro | Metodología: CRISP-DM*