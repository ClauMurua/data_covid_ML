from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline de Data Science para modelado COVID-19
    Entrenamiento y evaluación de múltiples modelos ML
    """
    return Pipeline([
        # Preparación de datos para modelado
        node(
            func=prepare_modeling_data,
            inputs=["regression_dataset", "classification_dataset", "params:modeling"],
            outputs=["regression_splits", "classification_splits"],
            name="prepare_modeling_data",
            tags=["data_preparation", "modeling"]
        ),
        
        # Modelos de Regresión
        node(
            func=train_regression_models,
            inputs=["regression_splits", "params:regression_models"],
            outputs="trained_regression_models",
            name="train_regression_models",
            tags=["modeling", "regression"]
        ),
        
        node(
            func=evaluate_regression_models,
            inputs=["trained_regression_models", "regression_splits"],
            outputs="regression_metrics",
            name="evaluate_regression_models",
            tags=["evaluation", "regression"]
        ),
        
        # Modelos de Clasificación
        node(
            func=train_classification_models,
            inputs=["classification_splits", "params:classification_models"],
            outputs="trained_classification_models",
            name="train_classification_models",
            tags=["modeling", "classification"]
        ),
        
        node(
            func=evaluate_classification_models,
            inputs=["trained_classification_models", "classification_splits"],
            outputs="classification_metrics",
            name="evaluate_classification_models",
            tags=["evaluation", "classification"]
        ),
        
        # Selección del mejor modelo
        node(
            func=select_best_models,
            inputs=["regression_metrics", "classification_metrics", "params:model_selection"],
            outputs="best_models_report",
            name="select_best_models",
            tags=["model_selection", "evaluation"]
        ),
        
        # Feature importance analysis
        node(
            func=analyze_feature_importance,
            inputs=["trained_regression_models", "trained_classification_models"],
            outputs="feature_importance_analysis",
            name="analyze_feature_importance",
            tags=["analysis", "interpretability"]
        )
    ])

# Importar funciones desde nodes.py
from .nodes import (
    prepare_modeling_data,
    train_regression_models,
    evaluate_regression_models,
    train_classification_models,
    evaluate_classification_models,
    select_best_models,
    analyze_feature_importance
)