from kedro.pipeline import Pipeline, node, pipeline

def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline de ingeniería de datos - Fase 2 y 3 CRISP-DM
    Comprensión y Preparación de los Datos
    """
    return pipeline([
        # Fase 2: Comprensión de los Datos
        node(
            func=validate_raw_data,
            inputs=["raw_covid_2020", "raw_covid_2021", "raw_covid_2022"],
            outputs="data_validation_report",
            name="validate_raw_datasets",
            tags=["data_understanding", "validation"]
        ),
        
        # Limpieza individual de cada dataset
        node(
            func=clean_covid_2020,
            inputs=["raw_covid_2020", "params:data_cleaning"],
            outputs="intermediate_covid_2020",
            name="clean_covid_2020_data",
            tags=["data_preparation", "cleaning"]
        ),
        
        node(
            func=clean_covid_2021,
            inputs=["raw_covid_2021", "params:data_cleaning"],
            outputs="intermediate_covid_2021", 
            name="clean_covid_2021_data",
            tags=["data_preparation", "cleaning"]
        ),
        
        node(
            func=clean_covid_2022,
            inputs=["raw_covid_2022", "params:data_cleaning"],
            outputs="intermediate_covid_2022",
            name="clean_covid_2022_data", 
            tags=["data_preparation", "cleaning"]
        ),
        
        # Fase 3: Preparación de los Datos
        node(
            func=integrate_datasets,
            inputs=["intermediate_covid_2020", "intermediate_covid_2021", "intermediate_covid_2022"],
            outputs="primary_covid_complete",
            name="integrate_covid_datasets",
            tags=["data_preparation", "integration"]
        ),
        
        node(
            func=create_national_aggregation,
            inputs="primary_covid_complete",
            outputs="primary_covid_national",
            name="create_national_daily_data",
            tags=["data_preparation", "aggregation"]
        ),
        
        node(
            func=quality_assessment,
            inputs=["primary_covid_complete", "primary_covid_national"],
            outputs="data_quality_report",
            name="assess_data_quality",
            tags=["data_understanding", "quality_assessment"]
        ),
        
        # Feature Engineering Básico
        node(
            func=create_basic_features,
            inputs=["primary_covid_national", "params:feature_engineering"],
            outputs="featured_covid_data",
            name="create_basic_features",
            tags=["data_preparation", "feature_engineering"]
        ),
        
        # Identificación de Targets ML
        node(
            func=identify_preliminary_targets,
            inputs=["featured_covid_data", "params:targets"],
            outputs="preliminary_targets",
            name="identify_preliminary_targets", 
            tags=["data_preparation", "target_identification"]
        )
    ])

# Importar las funciones desde nodes.py
from .nodes import (
    validate_raw_data,
    clean_covid_2020,
    clean_covid_2021,
    clean_covid_2022,
    integrate_datasets,
    create_national_aggregation,
    quality_assessment,
    create_basic_features,
    identify_preliminary_targets
)