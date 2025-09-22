import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

logger = logging.getLogger(__name__)

def prepare_modeling_data(
    regression_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepara los datos para modelado dividiendo en train/test y validación.
    
    Args:
        regression_df: Dataset para problemas de regresión
        classification_df: Dataset para problemas de clasificación
        params: Parámetros de modelado
        
    Returns:
        Tupla con splits de regresión y clasificación
    """
    logger.info("Preparando datos para modelado")
    
    # Parámetros
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    
    # Identificar features (excluir targets y fecha)
    regression_features = [col for col in regression_df.columns 
                          if not col.startswith('target_') and col != 'date']
    classification_features = [col for col in classification_df.columns 
                              if not col.startswith('target_') and col != 'date']
    
    # Targets de regresión
    regression_targets = ['target_confirmed_next_7d', 'target_growth_rate_next_14d']
    
    # Targets de clasificación
    classification_targets = ['target_high_transmission', 'target_risk_level', 'target_trend_direction']
    
    # Preparar splits para regresión
    regression_splits = {}
    for target in regression_targets:
        if target in regression_df.columns:
            # Remover NaN en target
            valid_data = regression_df.dropna(subset=[target])
            
            X = valid_data[regression_features]
            y = valid_data[target]
            
            # Split temporal (importante para series de tiempo)
            split_idx = int(len(valid_data) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            regression_splits[target] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': regression_features
            }
    
    # Preparar splits para clasificación
    classification_splits = {}
    for target in classification_targets:
        if target in classification_df.columns:
            # Remover NaN en target
            valid_data = classification_df.dropna(subset=[target])
            
            X = valid_data[classification_features]
            y = valid_data[target]
            
            # Split temporal
            split_idx = int(len(valid_data) * (1 - test_size))
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            classification_splits[target] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': classification_features
            }
    
    logger.info(f"Splits preparados - Regresión: {len(regression_splits)}, Clasificación: {len(classification_splits)}")
    return regression_splits, classification_splits

def train_regression_models(
    splits: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena múltiples modelos de regresión para diferentes targets.
    
    Args:
        splits: Splits de datos de regresión
        params: Parámetros de modelos
        
    Returns:
        Diccionario con modelos entrenados
    """
    logger.info("Entrenando modelos de regresión")
    
    # Definir modelos a entrenar
    models = {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(alpha=params.get("ridge_alpha", 1.0)),
        'lasso_regression': Lasso(alpha=params.get("lasso_alpha", 1.0)),
        'random_forest': RandomForestRegressor(
            n_estimators=params.get("rf_n_estimators", 100),
            random_state=params.get("random_state", 42)
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=params.get("gb_n_estimators", 100),
            random_state=params.get("random_state", 42)
        )
    }
    
    trained_models = {}
    
    for target_name, split_data in splits.items():
        logger.info(f"Entrenando modelos para target: {target_name}")
        
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        
        target_models = {}
        
        for model_name, model in models.items():
            try:
                # Entrenar modelo
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_train, y_train)
                
                target_models[model_name] = model_clone
                logger.info(f"Modelo {model_name} entrenado para {target_name}")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name} para {target_name}: {e}")
        
        trained_models[target_name] = target_models
    
    logger.info(f"Entrenamiento completado para {len(trained_models)} targets")
    return trained_models

def evaluate_regression_models(
    trained_models: Dict[str, Any],
    splits: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evalúa los modelos de regresión entrenados.
    
    Args:
        trained_models: Modelos entrenados
        splits: Splits de datos
        
    Returns:
        Métricas de evaluación
    """
    logger.info("Evaluando modelos de regresión")
    
    evaluation_results = {}
    
    for target_name, models in trained_models.items():
        logger.info(f"Evaluando modelos para target: {target_name}")
        
        split_data = splits[target_name]
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        
        target_results = {}
        
        for model_name, model in models.items():
            try:
                # Predicciones
                y_pred = model.predict(X_test)
                
                # Métricas
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / np.abs(y_test))) * 100
                }
                
                target_results[model_name] = metrics
                logger.info(f"{model_name} - RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluando {model_name} para {target_name}: {e}")
        
        evaluation_results[target_name] = target_results
    
    return evaluation_results

def train_classification_models(
    splits: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena múltiples modelos de clasificación.
    
    Args:
        splits: Splits de datos de clasificación
        params: Parámetros de modelos
        
    Returns:
        Diccionario con modelos entrenados
    """
    logger.info("Entrenando modelos de clasificación")
    
    # Definir modelos a entrenar
    models = {
        'logistic_regression': LogisticRegression(
            random_state=params.get("random_state", 42),
            max_iter=1000
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=params.get("rf_n_estimators", 100),
            random_state=params.get("random_state", 42)
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=params.get("gb_n_estimators", 100),
            random_state=params.get("random_state", 42)
        ),
        'svm': SVC(
            kernel=params.get("svm_kernel", "rbf"),
            random_state=params.get("random_state", 42)
        )
    }
    
    trained_models = {}
    
    for target_name, split_data in splits.items():
        logger.info(f"Entrenando modelos para target: {target_name}")
        
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        
        target_models = {}
        
        for model_name, model in models.items():
            try:
                # Entrenar modelo
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_train, y_train)
                
                target_models[model_name] = model_clone
                logger.info(f"Modelo {model_name} entrenado para {target_name}")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name} para {target_name}: {e}")
        
        trained_models[target_name] = target_models
    
    return trained_models

def evaluate_classification_models(
    trained_models: Dict[str, Any],
    splits: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evalúa los modelos de clasificación entrenados.
    
    Args:
        trained_models: Modelos entrenados
        splits: Splits de datos
        
    Returns:
        Métricas de evaluación
    """
    logger.info("Evaluando modelos de clasificación")
    
    evaluation_results = {}
    
    for target_name, models in trained_models.items():
        logger.info(f"Evaluando modelos para target: {target_name}")
        
        split_data = splits[target_name]
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        
        target_results = {}
        
        for model_name, model in models.items():
            try:
                # Predicciones
                y_pred = model.predict(X_test)
                
                # Métricas
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                target_results[model_name] = metrics
                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluando {model_name} para {target_name}: {e}")
        
        evaluation_results[target_name] = target_results
    
    return evaluation_results

def select_best_models(
    regression_metrics: Dict[str, Any],
    classification_metrics: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Selecciona los mejores modelos basado en métricas.
    
    Args:
        regression_metrics: Métricas de regresión
        classification_metrics: Métricas de clasificación
        params: Parámetros de selección
        
    Returns:
        Reporte con mejores modelos
    """
    logger.info("Seleccionando mejores modelos")
    
    best_models = {
        'regression': {},
        'classification': {},
        'summary': {}
    }
    
    # Mejores modelos de regresión (basado en R²)
    for target, models in regression_metrics.items():
        best_model = max(models.items(), key=lambda x: x[1]['r2'])
        best_models['regression'][target] = {
            'model': best_model[0],
            'metrics': best_model[1]
        }
        logger.info(f"Mejor modelo para {target}: {best_model[0]} (R²: {best_model[1]['r2']:.3f})")
    
    # Mejores modelos de clasificación (basado en F1)
    for target, models in classification_metrics.items():
        best_model = max(models.items(), key=lambda x: x[1]['f1'])
        best_models['classification'][target] = {
            'model': best_model[0],
            'metrics': best_model[1]
        }
        logger.info(f"Mejor modelo para {target}: {best_model[0]} (F1: {best_model[1]['f1']:.3f})")
    
    # Resumen general
    best_models['summary'] = {
        'total_regression_targets': len(regression_metrics),
        'total_classification_targets': len(classification_metrics),
        'evaluation_completed': True,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    return best_models

def analyze_feature_importance(
    regression_models: Dict[str, Any],
    classification_models: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analiza la importancia de features en los modelos.
    
    Args:
        regression_models: Modelos de regresión entrenados
        classification_models: Modelos de clasificación entrenados
        
    Returns:
        Análisis de importancia de features
    """
    logger.info("Analizando importancia de features")
    
    feature_analysis = {
        'regression': {},
        'classification': {}
    }
    
    # Análisis para modelos de regresión
    for target, models in regression_models.items():
        target_analysis = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                # Random Forest, Gradient Boosting
                importances = model.feature_importances_
                target_analysis[model_name] = importances.tolist()
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
                target_analysis[model_name] = importances.tolist()
        
        feature_analysis['regression'][target] = target_analysis
    
    # Análisis para modelos de clasificación
    for target, models in classification_models.items():
        target_analysis = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                target_analysis[model_name] = importances.tolist()
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
                target_analysis[model_name] = importances.tolist()
        
        feature_analysis['classification'][target] = target_analysis
    
    logger.info("Análisis de feature importance completado")
    return feature_analysis