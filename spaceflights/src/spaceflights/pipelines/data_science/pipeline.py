from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_modeling_data,
    train_regression_models,
    train_classification_models,
    evaluate_regression_models,
    evaluate_classification_models,
    select_best_models,
    analyze_feature_importance,
    tune_regression_models,
    tune_classification_models,
    create_tuning_comparison_table
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline completo de Data Science con hyperparameter tuning.
    """
    return pipeline([
        # 1. Preparación de datos
        node(
            func=prepare_modeling_data,
            inputs=["regression_dataset", "classification_dataset", "params:modeling"],
            outputs=["regression_splits", "classification_splits"],
            name="prepare_modeling_data"
        ),
        
        # 2. NUEVO: Hyperparameter Tuning para Regresión
        node(
            func=tune_regression_models,
            inputs=["regression_splits", "params:gridsearch_config"],
            outputs="tuned_regression_results",
            name="tune_regression_models"
        ),
        
        # 3. NUEVO: Hyperparameter Tuning para Clasificación
        node(
            func=tune_classification_models,
            inputs=["classification_splits", "params:gridsearch_config"],
            outputs="tuned_classification_results",
            name="tune_classification_models"
        ),
        
        # 4. Entrenamiento original (mantenido para comparación)
        node(
            func=train_regression_models,
            inputs=["regression_splits", "params:regression_models"],
            outputs="trained_regression_models",
            name="train_regression_models"
        ),
        
        node(
            func=train_classification_models,
            inputs=["classification_splits", "params:classification_models"],
            outputs="trained_classification_models",
            name="train_classification_models"
        ),
        
        # 5. Evaluación
        node(
            func=evaluate_regression_models,
            inputs=["trained_regression_models", "regression_splits"],
            outputs="regression_metrics",
            name="evaluate_regression_models"
        ),
        
        node(
            func=evaluate_classification_models,
            inputs=["trained_classification_models", "classification_splits"],
            outputs="classification_metrics",
            name="evaluate_classification_models"
        ),
        
        # 6. Feature Importance
        node(
            func=analyze_feature_importance,
            inputs=["trained_regression_models", "trained_classification_models"],
            outputs="feature_importance_analysis",
            name="analyze_feature_importance"
        ),
        
        # 7. Selección de mejores modelos
        node(
            func=select_best_models,
            inputs=["regression_metrics", "classification_metrics", "params:model_selection"],
            outputs="best_models_report",
            name="select_best_models"
        ),
        
        # 8. Tabla comparativa con mean±std
        node(
            func=create_tuning_comparison_table,
            inputs=[
                "tuned_regression_results",
                "tuned_classification_results",
                "regression_metrics",
                "classification_metrics"
            ],
            outputs="tuning_summary",
            name="create_tuning_comparison"
        )
    ])