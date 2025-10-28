from kedro.pipeline import Pipeline, node

# Importar funciones desde nodes.py
from .nodes import (
    combine_all_years,
    create_temporal_features,
    create_epidemiological_features,
    create_regional_features,
    apply_transformations,
    identify_ml_targets,
    prepare_ml_datasets
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline de procesamiento de datos COVID-19
    Feature Engineering y preparación para ML
    
    MEJORADO: Usa TODOS los 99,193 registros (363 ubicaciones)
    """
    return Pipeline([
        # NUEVO: Combinar todos los años primero
        node(
            func=combine_all_years,
            inputs=["raw_covid_2020", "raw_covid_2021", "raw_covid_2022"],
            outputs="combined_covid_regional",  # ← CAMBIO AQUÍ
            name="combine_all_years",
            tags=["data_preparation"]
        ),
        
        # Feature Engineering Epidemiológico
        node(
            func=create_epidemiological_features,
            inputs=["combined_covid_regional", "params:feature_engineering"],  # ← CAMBIO AQUÍ
            outputs="feature_epidemiological_data",
            name="create_epidemiological_features",
            tags=["feature_engineering", "epidemiological"]
        ),
        
        # Features Regionales
        node(
            func=create_regional_features,
            inputs=["combined_covid_regional", "params:feature_engineering"],  # ← CAMBIO AQUÍ
            outputs="feature_regional_data",
            name="create_regional_features",
            tags=["feature_engineering", "regional"]
        ),
        
        # Feature Engineering Temporal POR UBICACIÓN
        node(
            func=create_temporal_features,
            inputs=["combined_covid_regional", "params:feature_engineering"],  # ← CAMBIO AQUÍ
            outputs="feature_temporal_data",
            name="create_temporal_features",
            tags=["feature_engineering", "temporal"]
        ),
        
        # Transformaciones de datos
        node(
            func=apply_transformations,
            inputs=["feature_temporal_data", "feature_epidemiological_data", "params:transformations"],
            outputs="transformed_features",
            name="apply_transformations",
            tags=["feature_engineering", "transformations"]
        ),
        
        # Identificación de targets ML
        node(
            func=identify_ml_targets,
            inputs=["transformed_features", "params:ml_targets"],
            outputs="ml_targets_data",
            name="identify_ml_targets",
            tags=["target_identification", "ml_preparation"]
        ),
        
        # Preparación datasets finales para ML
        node(
            func=prepare_ml_datasets,
            inputs=["transformed_features", "ml_targets_data", "params:ml_preparation"],
            outputs=["regression_dataset", "classification_dataset"],
            name="prepare_ml_datasets", 
            tags=["ml_preparation", "final_datasets"]
        )
    ])