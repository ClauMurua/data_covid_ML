"""
Nodos para el pipeline de reporting del proyecto COVID-19 Chile.
Genera visualizaciones, reportes y análisis para presentación de resultados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para generación de gráficos
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# =====================================================
# VISUALIZACIONES TEMPORALES
# =====================================================

def create_temporal_evolution_plot(
    df_national: pd.DataFrame, 
    params: Dict[str, Any]
) -> plt.Figure:
    """
    Crea visualización de evolución temporal de casos y muertes COVID-19.
    
    Args:
        df_national: Dataset nacional diario con datos COVID-19
        params: Parámetros de visualización desde parameters_reporting.yml
        
    Returns:
        Figure de matplotlib con gráficos de series temporales
    """
    logger.info("Creando visualización de evolución temporal")
    
    # Configuración de estilo
    plt.style.use(params.get("style", "seaborn-v0_8-whitegrid"))
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=params.get("figure_size", [16, 12]))
    axes = axes.ravel()
    
    # Configuración de colores
    colors = params.get("temporal_visualizations", {}).get("colors", {})
    confirmed_color = colors.get("confirmed", "#3498db")
    deceased_color = colors.get("deceased", "#e74c3c")
    
    # 1. Casos diarios con media móvil
    if 'new_confirmed' in df_national.columns:
        axes[0].plot(df_national['date'], df_national['new_confirmed'], 
                    alpha=0.3, color=confirmed_color, label='Casos diarios')
        
        # Media móvil 7 días
        if len(df_national) >= 7:
            rolling_7 = df_national['new_confirmed'].rolling(7).mean()
            axes[0].plot(df_national['date'], rolling_7, 
                        color=confirmed_color, linewidth=2, label='Media móvil 7d')
        
        axes[0].set_title('Evolución de Casos Confirmados Diarios', fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Casos confirmados')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # 2. Muertes diarias con media móvil
    if 'new_deceased' in df_national.columns:
        axes[1].plot(df_national['date'], df_national['new_deceased'], 
                    alpha=0.3, color=deceased_color, label='Muertes diarias')
        
        if len(df_national) >= 7:
            rolling_7 = df_national['new_deceased'].rolling(7).mean()
            axes[1].plot(df_national['date'], rolling_7, 
                        color=deceased_color, linewidth=2, label='Media móvil 7d')
        
        axes[1].set_title('Evolución de Muertes Diarias', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('Muertes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # 3. Casos acumulados
    if 'cumulative_confirmed' in df_national.columns:
        axes[2].plot(df_national['date'], df_national['cumulative_confirmed'], 
                    color=confirmed_color, linewidth=2)
        axes[2].fill_between(df_national['date'], df_national['cumulative_confirmed'], 
                            alpha=0.3, color=confirmed_color)
        axes[2].set_title('Casos Acumulados', fontweight='bold', fontsize=14)
        axes[2].set_ylabel('Casos acumulados')
        axes[2].grid(True, alpha=0.3)
    
    # 4. Tasa de letalidad
    if 'case_fatality_rate' in df_national.columns:
        axes[3].plot(df_national['date'], df_national['case_fatality_rate'], 
                    color='#e67e22', linewidth=2)
        axes[3].set_title('Tasa de Letalidad (%)', fontweight='bold', fontsize=14)
        axes[3].set_ylabel('Tasa de letalidad (%)')
        axes[3].grid(True, alpha=0.3)
    
    # Marcar eventos importantes
    key_events = params.get("temporal_visualizations", {}).get("key_events", {})
    for fecha_str, evento in key_events.items():
        try:
            fecha_dt = pd.to_datetime(fecha_str)
            if df_national['date'].min() <= fecha_dt <= df_national['date'].max():
                for ax in axes:
                    ax.axvline(fecha_dt, color='orange', linestyle='--', alpha=0.5)
        except:
            continue
    
    plt.tight_layout()
    plt.suptitle('Análisis Temporal COVID-19 Chile', 
                fontsize=16, fontweight='bold', y=1.02)
    
    logger.info("Visualización temporal creada exitosamente")
    return fig


def create_weekly_comparison_plot(
    df_national: pd.DataFrame, 
    params: Dict[str, Any]
) -> plt.Figure:
    """
    Crea gráfico comparativo por semana del año.
    
    Args:
        df_national: Dataset nacional diario
        params: Parámetros de visualización
        
    Returns:
        Figure con análisis semanal
    """
    logger.info("Creando comparación semanal")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Agrupar por semana
    df_weekly = df_national.copy()
    df_weekly['week'] = df_weekly['date'].dt.isocalendar().week
    df_weekly['year'] = df_weekly['date'].dt.year
    
    weekly_summary = df_weekly.groupby(['year', 'week']).agg({
        'new_confirmed': 'sum',
        'new_deceased': 'sum'
    }).reset_index()
    
    # Casos por semana
    for year in weekly_summary['year'].unique():
        year_data = weekly_summary[weekly_summary['year'] == year]
        axes[0].plot(year_data['week'], year_data['new_confirmed'], 
                    marker='o', label=f'{year}', linewidth=2)
    
    axes[0].set_title('Casos Confirmados por Semana', fontweight='bold')
    axes[0].set_xlabel('Semana del año')
    axes[0].set_ylabel('Casos semanales')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Muertes por semana
    for year in weekly_summary['year'].unique():
        year_data = weekly_summary[weekly_summary['year'] == year]
        axes[1].plot(year_data['week'], year_data['new_deceased'], 
                    marker='o', label=f'{year}', linewidth=2)
    
    axes[1].set_title('Muertes por Semana', fontweight='bold')
    axes[1].set_xlabel('Semana del año')
    axes[1].set_ylabel('Muertes semanales')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =====================================================
# ANÁLISIS DE MODELOS ML
# =====================================================

def create_model_performance_report(
    regression_results: Dict[str, Any],
    classification_results: Dict[str, Any],
    params: Dict[str, Any]
) -> plt.Figure:
    """
    Crea reporte visual de performance de modelos ML.
    
    Args:
        regression_results: Resultados de modelos de regresión
        classification_results: Resultados de modelos de clasificación
        params: Parámetros de visualización
        
    Returns:
        Figure con comparación de modelos
    """
    logger.info("Creando reporte de performance de modelos")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Comparación R² de modelos de regresión
    if 'model_comparison' in regression_results:
        models_reg = regression_results['model_comparison']
        model_names = list(models_reg.keys())
        r2_scores = [models_reg[m].get('r2', 0) for m in model_names]
        
        axes[0, 0].barh(model_names, r2_scores, color='skyblue')
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Comparación R² - Modelos de Regresión', fontweight='bold')
        axes[0, 0].axvline(0.85, color='red', linestyle='--', label='Objetivo (0.85)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. Comparación RMSE de modelos de regresión
    if 'model_comparison' in regression_results:
        rmse_scores = [models_reg[m].get('rmse', 0) for m in model_names]
        
        axes[0, 1].barh(model_names, rmse_scores, color='lightcoral')
        axes[0, 1].set_xlabel('RMSE')
        axes[0, 1].set_title('Comparación RMSE - Modelos de Regresión', fontweight='bold')
        axes[0, 1].axvline(500, color='red', linestyle='--', label='Objetivo (<500)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Comparación F1-Score de modelos de clasificación
    if 'model_comparison' in classification_results:
        models_clf = classification_results['model_comparison']
        model_names_clf = list(models_clf.keys())
        f1_scores = [models_clf[m].get('f1', 0) for m in model_names_clf]
        
        axes[1, 0].barh(model_names_clf, f1_scores, color='lightgreen')
        axes[1, 0].set_xlabel('F1-Score')
        axes[1, 0].set_title('Comparación F1 - Modelos de Clasificación', fontweight='bold')
        axes[1, 0].axvline(0.80, color='red', linestyle='--', label='Objetivo (0.80)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Comparación Accuracy de modelos de clasificación
    if 'model_comparison' in classification_results:
        accuracy_scores = [models_clf[m].get('accuracy', 0) for m in model_names_clf]
        
        axes[1, 1].barh(model_names_clf, accuracy_scores, color='plum')
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_title('Comparación Accuracy - Modelos de Clasificación', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.suptitle('Performance de Modelos Machine Learning - COVID-19', 
                fontsize=16, fontweight='bold', y=1.02)
    
    return fig


def create_confusion_matrices(
    classification_results: Dict[str, Any],
    params: Dict[str, Any]
) -> plt.Figure:
    """
    Crea matrices de confusión para modelos de clasificación.
    
    Args:
        classification_results: Resultados con matrices de confusión
        params: Parámetros de visualización
        
    Returns:
        Figure con matrices de confusión
    """
    logger.info("Creando matrices de confusión")
    
    # Obtener matrices de confusión de los resultados
    confusion_matrices = classification_results.get('confusion_matrices', {})
    
    if not confusion_matrices:
        # Crear figura vacía con mensaje
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No hay matrices de confusión disponibles', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Crear subplots para cada modelo
    n_models = len(confusion_matrices)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        if idx < len(axes):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Frecuencia'})
            axes[idx].set_title(f'Matriz de Confusión - {model_name}', fontweight='bold')
            axes[idx].set_ylabel('Real')
            axes[idx].set_xlabel('Predicho')
    
    # Ocultar ejes no usados
    for idx in range(len(confusion_matrices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Matrices de Confusión - Modelos de Clasificación', 
                fontsize=16, fontweight='bold', y=1.02)
    
    return fig


def create_feature_importance_plot(
    regression_results: Dict[str, Any],
    params: Dict[str, Any]
) -> plt.Figure:
    """
    Crea visualización de importancia de features.
    
    Args:
        regression_results: Resultados con feature importance
        params: Parámetros con top_n_features
        
    Returns:
        Figure con feature importance
    """
    logger.info("Creando gráfico de feature importance")
    
    feature_importance = regression_results.get('feature_importance', {})
    top_n = params.get("model_performance", {}).get("top_features", 20)
    
    if not feature_importance:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Feature importance no disponible', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Tomar las top N features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importance = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    ax.barh(range(len(features)), importance, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importancia')
    ax.set_title(f'Top {top_n} Features Más Importantes', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


# =====================================================
# REPORTES EJECUTIVOS
# =====================================================

def create_executive_summary(
    df_national: pd.DataFrame,
    regression_results: Dict[str, Any],
    classification_results: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Crea resumen ejecutivo con KPIs principales.
    
    Args:
        df_national: Dataset nacional
        regression_results: Resultados de regresión
        classification_results: Resultados de clasificación
        params: Parámetros con KPIs
        
    Returns:
        Dict con métricas ejecutivas
    """
    logger.info("Generando resumen ejecutivo")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "kpis": {},
        "model_performance": {},
        "recommendations": []
    }
    
    # KPIs epidemiológicos
    if not df_national.empty:
        summary["kpis"]["casos_totales"] = int(df_national['cumulative_confirmed'].max())
        summary["kpis"]["muertes_totales"] = int(df_national['cumulative_deceased'].max())
        
        if 'case_fatality_rate' in df_national.columns:
            summary["kpis"]["tasa_letalidad"] = round(df_national['case_fatality_rate'].iloc[-1], 2)
        
        # Tendencia última semana
        if len(df_national) >= 7:
            last_week = df_national.tail(7)['new_confirmed'].mean()
            prev_week = df_national.tail(14).head(7)['new_confirmed'].mean()
            trend_pct = ((last_week - prev_week) / prev_week * 100) if prev_week > 0 else 0
            summary["kpis"]["tendencia_semanal"] = round(trend_pct, 1)
    
    # Performance de modelos
    if 'best_model' in regression_results:
        best_reg = regression_results['best_model']
        summary["model_performance"]["mejor_modelo_regresion"] = best_reg.get('name', 'N/A')
        summary["model_performance"]["r2_mejor_modelo"] = round(best_reg.get('r2', 0), 3)
    
    if 'best_model' in classification_results:
        best_clf = classification_results['best_model']
        summary["model_performance"]["mejor_modelo_clasificacion"] = best_clf.get('name', 'N/A')
        summary["model_performance"]["f1_mejor_modelo"] = round(best_clf.get('f1', 0), 3)
    
    # Recomendaciones basadas en resultados
    if summary["model_performance"].get("r2_mejor_modelo", 0) >= 0.85:
        summary["recommendations"].append("✓ Modelos de regresión cumplen criterio de éxito (R² >= 0.85)")
    else:
        summary["recommendations"].append("⚠ Modelos de regresión requieren optimización")
    
    if summary["model_performance"].get("f1_mejor_modelo", 0) >= 0.80:
        summary["recommendations"].append("✓ Modelos de clasificación cumplen criterio de éxito (F1 >= 0.80)")
    else:
        summary["recommendations"].append("⚠ Modelos de clasificación requieren mejora")
    
    logger.info("Resumen ejecutivo generado exitosamente")
    return summary


def create_comparative_analysis(
    df_national: pd.DataFrame,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Crea análisis comparativo entre períodos (pre/post vacunación).
    
    Args:
        df_national: Dataset nacional
        params: Parámetros con períodos de comparación
        
    Returns:
        Dict con análisis comparativo
    """
    logger.info("Creando análisis comparativo de períodos")
    
    periods = params.get("comparisons", {}).get("periods", {})
    metrics = params.get("comparisons", {}).get("metrics", [])
    
    comparative_analysis = {
        "periods": {},
        "comparison": {}
    }
    
    for period_name, period_dates in periods.items():
        try:
            start_date = pd.to_datetime(period_dates[0])
            end_date = pd.to_datetime(period_dates[1])
            
            period_data = df_national[
                (df_national['date'] >= start_date) & 
                (df_national['date'] <= end_date)
            ]
            
            comparative_analysis["periods"][period_name] = {
                "media_casos": round(period_data['new_confirmed'].mean(), 1),
                "media_muertes": round(period_data['new_deceased'].mean(), 1),
                "total_casos": int(period_data['new_confirmed'].sum()),
                "total_muertes": int(period_data['new_deceased'].sum())
            }
        except Exception as e:
            logger.warning(f"Error procesando período {period_name}: {e}")
    
    logger.info("Análisis comparativo completado")
    return comparative_analysis