"""
Pipeline de reporting para generar visualizaciones y reportes del proyecto COVID-19 Chile.
"""

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline de reporting - Visualizaciones y análisis de resultados COVID-19.
    
    Genera reportes visuales, análisis de modelos ML y resúmenes ejecutivos.
    """
    return pipeline([
        # Visualizaciones Temporales
        node(
            func=create_temporal_evolution_plot,
            inputs=["primary_covid_national", "params:reporting"],
            outputs="temporal_evolution_plot",
            name="create_temporal_plots",
            tags=["reporting", "visualization", "temporal"]
        ),
        
        node(
            func=create_weekly_comparison_plot,
            inputs=["primary_covid_national", "params:reporting"],
            outputs="weekly_comparison_plot",
            name="create_weekly_comparison",
            tags=["reporting", "visualization", "temporal"]
        ),
        
        # Análisis de Modelos ML
        node(
            func=create_model_performance_report,
            inputs=["regression_results", "classification_results", "params:reporting"],
            outputs="model_performance_report",
            name="create_model_performance_report",
            tags=["reporting", "ml_analysis"]
        ),
        
        node(
            func=create_confusion_matrices,
            inputs=["classification_results", "params:reporting"],
            outputs="confusion_matrices_plot",
            name="create_confusion_matrices",
            tags=["reporting", "ml_analysis", "classification"]
        ),
        
        node(
            func=create_feature_importance_plot,
            inputs=["regression_results", "params:reporting"],
            outputs="feature_importance_plot",
            name="create_feature_importance",
            tags=["reporting", "ml_analysis", "regression"]
        ),
        
        # Reportes Ejecutivos
        node(
            func=create_executive_summary,
            inputs=[
                "primary_covid_national",
                "regression_results",
                "classification_results",
                "params:reporting"
            ],
            outputs="executive_summary",
            name="generate_executive_summary",
            tags=["reporting", "executive"]
        ),
        
        node(
            func=create_comparative_analysis,
            inputs=["primary_covid_national", "params:reporting"],
            outputs="comparative_analysis",
            name="create_comparative_analysis",
            tags=["reporting", "analysis"]
        ),
    ])


# Importar funciones desde nodes.py
from .nodes import (
    create_temporal_evolution_plot,
    create_weekly_comparison_plot,
    create_model_performance_report,
    create_confusion_matrices,
    create_feature_importance_plot,
    create_executive_summary,
    create_comparative_analysis
)