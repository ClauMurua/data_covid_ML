import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

# =====================================================
# PREPARACIÓN DE DATOS PARA MODELADO
# =====================================================

def prepare_modeling_data(
    regression_dataset: pd.DataFrame,
    classification_dataset: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Prepara y divide los datos para modelado ML usando split temporal.
    
    Esta función implementa división temporal de datos, manteniendo el orden
    cronológico para respetar la naturaleza secuencial de series de tiempo COVID-19.
    
    Args:
        regression_dataset (pd.DataFrame): Dataset con targets de regresión.
            Debe contener columna 'date' y columnas target_*.
        classification_dataset (pd.DataFrame): Dataset con targets de clasificación.
            Debe contener columna 'date' y columnas target_*.
        params (Dict[str, Any]): Parámetros de configuración:
            - test_size (float): Proporción para test set. Default: 0.2
            - val_size (float): Proporción para validation set. Default: 0.15
            - temporal_split (bool): Usar split temporal vs aleatorio. Default: True
        
    Returns:
        Tuple[Dict, Dict]: Diccionarios con splits de regresión y clasificación:
            - 'train': DataFrame de entrenamiento
            - 'val': DataFrame de validación
            - 'test': DataFrame de test
            - 'feature_columns': Lista de columnas de features
            - 'target_columns': Lista de columnas target
            
    Example:
        >>> params = {"test_size": 0.2, "val_size": 0.15, "temporal_split": True}
        >>> reg_splits, clf_splits = prepare_modeling_data(reg_df, clf_df, params)
        >>> print(reg_splits['train'].shape)
        
    Notes:
        - Split temporal es crítico para evitar data leakage en series temporales
        - Train: 65%, Val: 15%, Test: 20% (configuración por defecto)
        - Los datos se ordenan por fecha antes del split
    """
    logger.info("Preparando datos para modelado con split temporal")
    
    # LIMPIEZA DE VALORES INFINITOS Y NAN (CRÍTICO)
    logger.info("🧹 Limpiando valores infinitos y NaN de los datasets...")
    
    # Limpiar regression_dataset
    inf_count_reg = np.isinf(regression_dataset.select_dtypes(include=[np.number])).sum().sum()
    nan_count_reg = regression_dataset.isna().sum().sum()
    logger.info(f"   Regresión - Infinitos: {inf_count_reg}, NaN: {nan_count_reg}")
    
    regression_dataset = regression_dataset.replace([np.inf, -np.inf], np.nan)
    regression_dataset = regression_dataset.fillna(0)
    
    # Limpiar classification_dataset
    inf_count_clf = np.isinf(classification_dataset.select_dtypes(include=[np.number])).sum().sum()
    nan_count_clf = classification_dataset.isna().sum().sum()
    logger.info(f"   Clasificación - Infinitos: {inf_count_clf}, NaN: {nan_count_clf}")
    
    classification_dataset = classification_dataset.replace([np.inf, -np.inf], np.nan)
    classification_dataset = classification_dataset.fillna(0)
    
    logger.info(f"✅ Datos limpios - Regresión: {regression_dataset.shape}, Clasificación: {classification_dataset.shape}")
    
    test_size = params.get("test_size", 0.2)
    val_size = params.get("val_size", 0.15)
    temporal_split = params.get("temporal_split", True)
    random_state = params.get("random_state", 42)
    
    def temporal_train_test_split(df: pd.DataFrame, test_size: float, val_size: float):
        """
        Split temporal manteniendo orden cronológico.
        
        Args:
            df: DataFrame con columna 'date'
            test_size: Proporción para test
            val_size: Proporción para validation
            
        Returns:
            Tuple con train, val, test DataFrames
        """
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        n = len(df_sorted)
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size))
        
        train_df = df_sorted.iloc[:val_start].copy()
        val_df = df_sorted.iloc[val_start:test_start].copy()
        test_df = df_sorted.iloc[test_start:].copy()
        
        logger.debug(f"Split temporal: Train hasta {train_df['date'].max()}, "
                    f"Val hasta {val_df['date'].max()}, Test hasta {test_df['date'].max()}")
        
        return train_df, val_df, test_df
    
    # Split para regresión
    if temporal_split:
        reg_train, reg_val, reg_test = temporal_train_test_split(
            regression_dataset, test_size, val_size
        )
    else:
        logger.warning("Usando split aleatorio (no recomendado para series temporales)")
        reg_train, reg_temp = train_test_split(
            regression_dataset, test_size=(test_size + val_size), random_state=random_state
        )
        reg_val, reg_test = train_test_split(
            reg_temp, test_size=(test_size/(test_size+val_size)), random_state=random_state
        )
    
    # Split para clasificación
    if temporal_split:
        clf_train, clf_val, clf_test = temporal_train_test_split(
            classification_dataset, test_size, val_size
        )
    else:
        clf_train, clf_temp = train_test_split(
            classification_dataset, test_size=(test_size + val_size), random_state=random_state
        )
        clf_val, clf_test = train_test_split(
            clf_temp, test_size=(test_size/(test_size+val_size)), random_state=random_state
        )
    
    # Preparar diccionarios de splits
    regression_splits = {
        'train': reg_train,
        'val': reg_val,
        'test': reg_test,
        'feature_columns': [col for col in reg_train.columns 
                  if not col.startswith('target_') and col not in ['date', 'location_key']],
        'target_columns': [col for col in reg_train.columns if col.startswith('target_')]
    }
    
    classification_splits = {
        'train': clf_train,
        'val': clf_val,
        'test': clf_test,
        'feature_columns': [col for col in clf_train.columns 
                        if not col.startswith('target_') and col not in ['date', 'location_key']],
        'target_columns': [col for col in clf_train.columns if col.startswith('target_')]
}
    
    # Logging detallado
    logger.info(f"📊 SPLITS DE REGRESIÓN:")
    logger.info(f"   • Train: {len(reg_train)} registros ({len(reg_train)/len(regression_dataset)*100:.1f}%)")
    logger.info(f"   • Val: {len(reg_val)} registros ({len(reg_val)/len(regression_dataset)*100:.1f}%)")
    logger.info(f"   • Test: {len(reg_test)} registros ({len(reg_test)/len(regression_dataset)*100:.1f}%)")
    logger.info(f"   • Features: {len(regression_splits['feature_columns'])}")
    logger.info(f"   • Targets: {regression_splits['target_columns']}")
    
    logger.info(f"📊 SPLITS DE CLASIFICACIÓN:")
    logger.info(f"   • Train: {len(clf_train)} registros ({len(clf_train)/len(classification_dataset)*100:.1f}%)")
    logger.info(f"   • Val: {len(clf_val)} registros ({len(clf_val)/len(classification_dataset)*100:.1f}%)")
    logger.info(f"   • Test: {len(clf_test)} registros ({len(clf_test)/len(classification_dataset)*100:.1f}%)")
    logger.info(f"   • Features: {len(classification_splits['feature_columns'])}")
    logger.info(f"   • Targets: {classification_splits['target_columns']}")
    
    return regression_splits, classification_splits
    
    def temporal_train_test_split(df: pd.DataFrame, test_size: float, val_size: float):
        """
        Split temporal manteniendo orden cronológico.
        
        Args:
            df: DataFrame con columna 'date'
            test_size: Proporción para test
            val_size: Proporción para validation
            
        Returns:
            Tuple con train, val, test DataFrames
        """
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        n = len(df_sorted)
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size))
        
        train_df = df_sorted.iloc[:val_start].copy()
        val_df = df_sorted.iloc[val_start:test_start].copy()
        test_df = df_sorted.iloc[test_start:].copy()
        
        logger.debug(f"Split temporal: Train hasta {train_df['date'].max()}, "
                    f"Val hasta {val_df['date'].max()}, Test hasta {test_df['date'].max()}")
        
        return train_df, val_df, test_df
    
    # Split para regresión
    if temporal_split:
        reg_train, reg_val, reg_test = temporal_train_test_split(
            regression_dataset, test_size, val_size
        )
    else:
        logger.warning("Usando split aleatorio (no recomendado para series temporales)")
        reg_train, reg_temp = train_test_split(
            regression_dataset, test_size=(test_size + val_size), random_state=random_state
        )
        reg_val, reg_test = train_test_split(
            reg_temp, test_size=(test_size/(test_size+val_size)), random_state=random_state
        )
    
    # Split para clasificación
    if temporal_split:
        clf_train, clf_val, clf_test = temporal_train_test_split(
            classification_dataset, test_size, val_size
        )
    else:
        clf_train, clf_temp = train_test_split(
            classification_dataset, test_size=(test_size + val_size), random_state=random_state
        )
        clf_val, clf_test = train_test_split(
            clf_temp, test_size=(test_size/(test_size+val_size)), random_state=random_state
        )
    
    # Preparar diccionarios de splits
    regression_splits = {
        'train': reg_train,
        'val': reg_val,
        'test': reg_test,
        'feature_columns': [col for col in reg_train.columns 
                        if not col.startswith('target_') and col not in ['date', 'location_key']],
        'target_columns': [col for col in reg_train.columns if col.startswith('target_')]
}
    
    classification_splits = {
        'train': clf_train,
        'val': clf_val,
        'test': clf_test,
        'feature_columns': [col for col in clf_train.columns 
                          if not col.startswith('target_') and col != 'date'],
        'target_columns': [col for col in clf_train.columns if col.startswith('target_')]
    }
    
    # Logging detallado
    logger.info(f"📊 SPLITS DE REGRESIÓN:")
    logger.info(f"   • Train: {len(reg_train)} registros ({len(reg_train)/len(regression_dataset)*100:.1f}%)")
    logger.info(f"   • Val: {len(reg_val)} registros ({len(reg_val)/len(regression_dataset)*100:.1f}%)")
    logger.info(f"   • Test: {len(reg_test)} registros ({len(reg_test)/len(regression_dataset)*100:.1f}%)")
    logger.info(f"   • Features: {len(regression_splits['feature_columns'])}")
    logger.info(f"   • Targets: {regression_splits['target_columns']}")
    
    logger.info(f"📊 SPLITS DE CLASIFICACIÓN:")
    logger.info(f"   • Train: {len(clf_train)} registros ({len(clf_train)/len(classification_dataset)*100:.1f}%)")
    logger.info(f"   • Val: {len(clf_val)} registros ({len(clf_val)/len(classification_dataset)*100:.1f}%)")
    logger.info(f"   • Test: {len(clf_test)} registros ({len(clf_test)/len(classification_dataset)*100:.1f}%)")
    logger.info(f"   • Features: {len(classification_splits['feature_columns'])}")
    logger.info(f"   • Targets: {classification_splits['target_columns']}")
    
    return regression_splits, classification_splits

# =====================================================
# ENTRENAMIENTO - REGRESIÓN
# =====================================================

def train_regression_models(
    regression_splits: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena múltiples modelos de regresión para predicción de casos COVID-19.
    
    Esta función implementa 6 algoritmos de regresión diferentes para comparación:
    Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting y XGBoost.
    
    Args:
        regression_splits (Dict): Diccionario con splits de datos que debe contener:
            - 'train': DataFrame de entrenamiento
            - 'feature_columns': Lista de columnas de features
            - 'target_columns': Lista de targets a predecir
        params (Dict[str, Any]): Parámetros de configuración por modelo:
            - linear_regression: {'enabled': bool}
            - ridge_regression: {'enabled': bool, 'ridge_alpha': float}
            - lasso_regression: {'enabled': bool, 'lasso_alpha': float}
            - random_forest: {'enabled': bool, 'rf_n_estimators': int, 'max_depth': int}
            - gradient_boosting: {'enabled': bool, 'gb_n_estimators': int, 'learning_rate': float}
            - xgboost: {'enabled': bool, 'n_estimators': int, 'max_depth': int, 'learning_rate': float}
        
    Returns:
        Dict[str, Any]: Diccionario anidado estructura:
            {target_name: {model_name: trained_model}}
            
    Example:
        >>> params = {
        ...     "random_forest": {"enabled": True, "rf_n_estimators": 100}
        ... }
        >>> models = train_regression_models(regression_splits, params)
        >>> print(models['target_confirmed_next_7d']['random_forest'])
        
    Notes:
        - Cada target se modela independientemente
        - Valores NaN se rellenan con 0 antes del entrenamiento
        - Random state=42 para reproducibilidad
        - Modelos deshabilitados (enabled=False) se omiten
        
    See Also:
        evaluate_regression_models: Para evaluación de modelos entrenados
    """
    logger.info("🤖 Entrenando modelos de regresión")
    
    train_df = regression_splits['train']
    feature_cols = regression_splits['feature_columns']
    target_cols = regression_splits['target_columns']
    
    X_train = train_df[feature_cols].fillna(0)
    
    trained_models = {}
    total_models = 0
    
    # Entrenar un modelo por cada target
    for target in target_cols:
        logger.info(f"📌 Entrenando modelos para target: {target}")
        y_train = train_df[target].fillna(0)
        
        target_models = {}
        
        # 1. Linear Regression
        if params.get("linear_regression", {}).get("enabled", True):
            try:
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                target_models['linear_regression'] = lr
                total_models += 1
                logger.debug(f"   ✓ Linear Regression entrenado")
            except Exception as e:
                logger.error(f"   ✗ Error en Linear Regression: {e}")
        
        # 2. Ridge Regression
        if params.get("ridge_regression", {}).get("enabled", True):
            try:
                ridge_alpha = params.get("ridge_regression", {}).get("ridge_alpha", 1.0)
                ridge = Ridge(alpha=ridge_alpha, random_state=42)
                ridge.fit(X_train, y_train)
                target_models['ridge_regression'] = ridge
                total_models += 1
                logger.debug(f"   ✓ Ridge Regression entrenado (alpha={ridge_alpha})")
            except Exception as e:
                logger.error(f"   ✗ Error en Ridge: {e}")
        
        # 3. Lasso Regression
        if params.get("lasso_regression", {}).get("enabled", True):
            try:
                lasso_alpha = params.get("lasso_regression", {}).get("lasso_alpha", 1.0)
                lasso = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=42)
                lasso.fit(X_train, y_train)
                target_models['lasso_regression'] = lasso
                total_models += 1
                logger.debug(f"   ✓ Lasso Regression entrenado (alpha={lasso_alpha})")
            except Exception as e:
                logger.error(f"   ✗ Error en Lasso: {e}")
        
        # 4. Random Forest
        if params.get("random_forest", {}).get("enabled", True):
            try:
                rf = RandomForestRegressor(
                    n_estimators=params.get("random_forest", {}).get("rf_n_estimators", 100),
                    max_depth=params.get("random_forest", {}).get("max_depth", 10),
                    min_samples_split=params.get("random_forest", {}).get("min_samples_split", 5),
                    min_samples_leaf=params.get("random_forest", {}).get("min_samples_leaf", 2),
                    n_jobs=params.get("random_forest", {}).get("n_jobs", -1),
                    random_state=42
                )
                rf.fit(X_train, y_train)
                target_models['random_forest'] = rf
                total_models += 1
                logger.debug(f"   ✓ Random Forest entrenado")
            except Exception as e:
                logger.error(f"   ✗ Error en Random Forest: {e}")
        
        # 5. Gradient Boosting
        if params.get("gradient_boosting", {}).get("enabled", True):
            try:
                gb = GradientBoostingRegressor(
                    n_estimators=params.get("gradient_boosting", {}).get("gb_n_estimators", 100),
                    learning_rate=params.get("gradient_boosting", {}).get("learning_rate", 0.1),
                    max_depth=params.get("gradient_boosting", {}).get("max_depth", 5),
                    subsample=params.get("gradient_boosting", {}).get("subsample", 0.8),
                    random_state=42
                )
                gb.fit(X_train, y_train)
                target_models['gradient_boosting'] = gb
                total_models += 1
                logger.debug(f"   ✓ Gradient Boosting entrenado")
            except Exception as e:
                logger.error(f"   ✗ Error en Gradient Boosting: {e}")
        
        # 6. XGBoost (NUEVO)
        if params.get("xgboost", {}).get("enabled", True):
            try:
                from xgboost import XGBRegressor
                
                xgb_model = XGBRegressor(
                    n_estimators=params.get("xgboost", {}).get("n_estimators", 200),
                    max_depth=params.get("xgboost", {}).get("max_depth", 6),
                    learning_rate=params.get("xgboost", {}).get("learning_rate", 0.1),
                    subsample=params.get("xgboost", {}).get("subsample", 0.8),
                    colsample_bytree=params.get("xgboost", {}).get("colsample_bytree", 0.8),
                    gamma=params.get("xgboost", {}).get("gamma", 0),
                    min_child_weight=params.get("xgboost", {}).get("min_child_weight", 1),
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0  # Silenciar warnings
                )
                
                xgb_model.fit(X_train, y_train)
                target_models['xgboost'] = xgb_model
                total_models += 1
                logger.debug(f"   ✓ XGBoost entrenado")
            except ImportError:
                logger.error(f"   ✗ XGBoost no está instalado. Ejecuta: pip install xgboost")
            except Exception as e:
                logger.error(f"   ✗ Error en XGBoost: {e}")
        
        trained_models[target] = target_models
    
    logger.info(f"✅ Entrenamiento completado: {len(trained_models)} targets, {total_models} modelos totales")
    return trained_models

# =====================================================
# EVALUACIÓN - REGRESIÓN
# =====================================================

def evaluate_regression_models(
    trained_models: Dict[str, Any],
    regression_splits: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Evalúa modelos de regresión en conjuntos de validación y test.
    
    Calcula métricas estándar de regresión (RMSE, MAE, R²) para cada modelo
    en los conjuntos de validación y test, permitiendo comparación de performance.
    
    Args:
        trained_models (Dict): Modelos entrenados por target y algoritmo
        regression_splits (Dict): Splits de datos con 'val', 'test', 'feature_columns'
        
    Returns:
        Dict[str, Any]: Métricas anidadas por target y modelo:
            {
                target_name: {
                    model_name: {
                        'validation': {'rmse': float, 'mae': float, 'r2': float},
                        'test': {'rmse': float, 'mae': float, 'r2': float},
                        'predictions': {'val': list, 'test': list}
                    }
                }
            }
            
    Notes:
        - RMSE: Root Mean Squared Error (menor es mejor)
        - MAE: Mean Absolute Error (menor es mejor)
        - R²: Coeficiente de determinación (mayor es mejor, máximo 1.0)
        - Criterio de éxito rúbrica: R² >= 0.85, RMSE < 500
        
    Example:
        >>> results = evaluate_regression_models(trained_models, regression_splits)
        >>> best_r2 = results['target_confirmed_next_7d']['random_forest']['test']['r2']
    """
    logger.info("📊 Evaluando modelos de regresión")
    
    val_df = regression_splits['val']
    test_df = regression_splits['test']
    feature_cols = regression_splits['feature_columns']
    
    X_val = val_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    
    results = {}
    
    for target, models in trained_models.items():
        logger.info(f"📌 Evaluando target: {target}")
        
        y_val = val_df[target].fillna(0)
        y_test = test_df[target].fillna(0)
        
        target_results = {}
        
        for model_name, model in models.items():
            try:
                # Predicciones
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                
                # Métricas de validación
                val_metrics = {
                    'rmse': float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
                    'mae': float(mean_absolute_error(y_val, y_val_pred)),
                    'r2': float(r2_score(y_val, y_val_pred))
                }
                
                # Métricas de test
                test_metrics = {
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                    'mae': float(mean_absolute_error(y_test, y_test_pred)),
                    'r2': float(r2_score(y_test, y_test_pred))
                }
                
                target_results[model_name] = {
                    'validation': val_metrics,
                    'test': test_metrics,
                    'predictions': {
                        'val': y_val_pred.tolist()[:100],  # Limitar para almacenamiento
                        'test': y_test_pred.tolist()[:100]
                    }
                }
                
                # Logging con evaluación de criterios
                meets_criteria = test_metrics['r2'] >= 0.85 and test_metrics['rmse'] < 500
                status = "✅" if meets_criteria else "⚠️"
                
                logger.info(f"   {status} {model_name:20} - R²: {test_metrics['r2']:.3f}, "
                           f"RMSE: {test_metrics['rmse']:.2f}, MAE: {test_metrics['mae']:.2f}")
                
            except Exception as e:
                logger.error(f"   ✗ Error evaluando {model_name}: {e}")
        
        results[target] = target_results
    
    logger.info("✅ Evaluación de regresión completada")
    return results

# =====================================================
# ENTRENAMIENTO - CLASIFICACIÓN
# =====================================================

def train_classification_models(
    classification_splits: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena múltiples modelos de clasificación para detección de riesgo COVID-19.
    
    Implementa 4 algoritmos: Logistic Regression, Random Forest, Gradient Boosting y SVM
    para problemas de clasificación binaria y multiclase.
    
    Args:
        classification_splits (Dict): Splits con 'train', 'feature_columns', 'target_columns'
        params (Dict[str, Any]): Parámetros por modelo (similar a regresión)
        
    Returns:
        Dict[str, Any]: Modelos entrenados por target y algoritmo
        
    Notes:
        - class_weight='balanced' para manejar desbalance de clases
        - SVM deshabilitado por defecto (puede ser lento con muchos datos)
        - Targets deben ser valores enteros (0, 1, 2, etc.)
    """
    logger.info("🤖 Entrenando modelos de clasificación")
    
    train_df = classification_splits['train']
    feature_cols = classification_splits['feature_columns']
    target_cols = classification_splits['target_columns']
    
    X_train = train_df[feature_cols].fillna(0)
    
    trained_models = {}
    total_models = 0
    
    for target in target_cols:
        logger.info(f"📌 Entrenando modelos para target: {target}")
        y_train = train_df[target].fillna(0).astype(int)
        
        # Verificar distribución de clases
        class_distribution = y_train.value_counts().to_dict()
        logger.debug(f"   Distribución de clases: {class_distribution}")
        
        target_models = {}
        
        # 1. Logistic Regression
        if params.get("logistic_regression", {}).get("enabled", True):
            try:
                lr = LogisticRegression(
                    max_iter=params.get("logistic_regression", {}).get("max_iter", 1000),
                    solver=params.get("logistic_regression", {}).get("solver", 'lbfgs'),
                    class_weight='balanced',
                    random_state=42
                )
                lr.fit(X_train, y_train)
                target_models['logistic_regression'] = lr
                total_models += 1
                logger.debug(f"   ✓ Logistic Regression entrenado")
            except Exception as e:
                logger.error(f"   ✗ Error en Logistic Regression: {e}")
        
        # 2. Random Forest
        if params.get("random_forest", {}).get("enabled", True):
            try:
                rf = RandomForestClassifier(
                    n_estimators=params.get("random_forest", {}).get("rf_n_estimators", 100),
                    max_depth=params.get("random_forest", {}).get("max_depth", 10),
                    min_samples_split=params.get("random_forest", {}).get("min_samples_split", 5),
                    class_weight=params.get("random_forest", {}).get("class_weight", 'balanced'),
                    n_jobs=-1,
                    random_state=42
                )
                rf.fit(X_train, y_train)
                target_models['random_forest'] = rf
                total_models += 1
                logger.debug(f"   ✓ Random Forest entrenado")
            except Exception as e:
                logger.error(f"   ✗ Error en Random Forest: {e}")
        
        # 3. Gradient Boosting
        if params.get("gradient_boosting", {}).get("enabled", True):
            try:
                gb = GradientBoostingClassifier(
                    n_estimators=params.get("gradient_boosting", {}).get("gb_n_estimators", 100),
                    learning_rate=params.get("gradient_boosting", {}).get("learning_rate", 0.1),
                    max_depth=params.get("gradient_boosting", {}).get("max_depth", 5),
                    random_state=42
                )
                gb.fit(X_train, y_train)
                target_models['gradient_boosting'] = gb
                total_models += 1
                logger.debug(f"   ✓ Gradient Boosting entrenado")
            except Exception as e:
                logger.error(f"   ✗ Error en Gradient Boosting: {e}")
        
        # 4. SVM (opcional - deshabilitado por defecto)
        if params.get("svm", {}).get("enabled", False):
            try:
                svm = SVC(
                    kernel=params.get("svm", {}).get("svm_kernel", 'rbf'),
                    C=params.get("svm", {}).get("C", 1.0),
                    gamma=params.get("svm", {}).get("gamma", 'scale'),
                    class_weight='balanced',
                    random_state=42
                )
                svm.fit(X_train, y_train)
                target_models['svm'] = svm
                total_models += 1
                logger.debug(f"   ✓ SVM entrenado")
            except Exception as e:
                logger.error(f"   ✗ Error en SVM: {e}")
        
        # 5. K-Neighbors Classifier (NUEVO)
        if params.get("k_neighbors_classifier", {}).get("enabled", True):
            try:
                n_neighbors = params.get("k_neighbors_classifier", {}).get("n_neighbors", 5)
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
                knn.fit(X_train, y_train)
                target_models['k_neighbors_classifier'] = knn
                total_models += 1
                logger.debug(f"   ✓ K-Neighbors Classifier entrenado (n_neighbors={n_neighbors})")
            except Exception as e:
                logger.error(f"   ✗ Error en K-Neighbors Classifier: {e}")
        
        trained_models[target] = target_models
    
    logger.info(f"✅ Entrenamiento completado: {len(trained_models)} targets, {total_models} modelos totales")
    return trained_models

# =====================================================
# EVALUACIÓN - CLASIFICACIÓN
# =====================================================

def evaluate_classification_models(
    trained_models: Dict[str, Any],
    classification_splits: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Evalúa modelos de clasificación con métricas estándar.
    
    Calcula Accuracy, Precision, Recall, F1-Score y matriz de confusión
    para evaluación completa de modelos de clasificación.
    
    Args:
        trained_models (Dict): Modelos entrenados por target
        classification_splits (Dict): Splits con val/test
        
    Returns:
        Dict con métricas y matrices de confusión
        
    Notes:
        - average='binary' para clasificación binaria
        - average='weighted' para clasificación multiclase
        - Criterio de éxito rúbrica: F1 >= 0.80, Precision >= 0.85
    """
    logger.info("📊 Evaluando modelos de clasificación")
    
    val_df = classification_splits['val']
    test_df = classification_splits['test']
    feature_cols = classification_splits['feature_columns']
    
    X_val = val_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)
    
    results = {}
    
    for target, models in trained_models.items():
        logger.info(f"📌 Evaluando target: {target}")
        
        y_val = val_df[target].fillna(0).astype(int)
        y_test = test_df[target].fillna(0).astype(int)
        
        target_results = {}
        
        for model_name, model in models.items():
            try:
                # Predicciones
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                
                # Determinar average para métricas
                n_classes = len(np.unique(y_test))
                average = 'binary' if n_classes == 2 else 'weighted'
                
                # Métricas de validación
                val_metrics = {
                    'accuracy': float(accuracy_score(y_val, y_val_pred)),
                    'precision': float(precision_score(y_val, y_val_pred, average=average, zero_division=0)),
                    'recall': float(recall_score(y_val, y_val_pred, average=average, zero_division=0)),
                    'f1': float(f1_score(y_val, y_val_pred, average=average, zero_division=0))
                }
                
                # Métricas de test
                test_metrics = {
                    'accuracy': float(accuracy_score(y_test, y_test_pred)),
                    'precision': float(precision_score(y_test, y_test_pred, average=average, zero_division=0)),
                    'recall': float(recall_score(y_test, y_test_pred, average=average, zero_division=0)),
                    'f1': float(f1_score(y_test, y_test_pred, average=average, zero_division=0)),
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
                }
                
                target_results[model_name] = {
                    'validation': val_metrics,
                    'test': test_metrics,
                    'predictions': {
                        'val': y_val_pred.tolist()[:100],
                        'test': y_test_pred.tolist()[:100]
                    }
                }
                
                # Logging con evaluación de criterios
                meets_criteria = test_metrics['f1'] >= 0.80 and test_metrics['precision'] >= 0.85
                status = "✅" if meets_criteria else "⚠️"
                
                logger.info(f"   {status} {model_name:20} - F1: {test_metrics['f1']:.3f}, "
                           f"Acc: {test_metrics['accuracy']:.3f}, Prec: {test_metrics['precision']:.3f}")
                
            except Exception as e:
                logger.error(f"   ✗ Error evaluando {model_name}: {e}")
        
        results[target] = target_results
    
    logger.info("✅ Evaluación de clasificación completada")
    return results

# =====================================================
# SELECCIÓN DEL MEJOR MODELO
# =====================================================

def select_best_models(
    regression_metrics: Dict[str, Any],
    classification_metrics: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Selecciona los mejores modelos basándose en métricas de test.
    
    Implementa selección automática del mejor modelo por target usando
    la métrica especificada en parámetros.
    
    Args:
        regression_metrics (Dict): Métricas de todos los modelos de regresión
        classification_metrics (Dict): Métricas de todos los modelos de clasificación
        params (Dict): Parámetros con criterios de selección:
            - regression_metric (str): Métrica para regresión ('r2', 'rmse', 'mae')
            - classification_metric (str): Métrica para clasificación ('f1', 'accuracy')
            - success_criteria (Dict): Umbrales de éxito por tipo de modelo
        
    Returns:
        Dict[str, Any]: Reporte completo con:
            - Mejor modelo por target
            - Score del mejor modelo
            - Comparación con todos los modelos
            - Evaluación vs criterios de éxito
            
    Example:
        >>> report = select_best_models(reg_metrics, clf_metrics, params)
        >>> best_reg = report['regression']['target_confirmed_next_7d']['best_model']
        >>> print(f"Mejor modelo: {best_reg}")
        
    Notes:
        - Para R² se maximiza (mayor es mejor)
        - Para RMSE/MAE se minimiza (menor es mejor)
        - Para F1/Accuracy se maximiza
    """
    logger.info("🏆 Seleccionando mejores modelos")
    
    regression_metric = params.get("regression_metric", "r2")
    classification_metric = params.get("classification_metric", "f1")
    success_criteria = params.get("success_criteria", {})
    
    best_models_report = {
        "timestamp": datetime.now().isoformat(),
        "regression": {},
        "classification": {},
        "summary": {},
        "criteria_evaluation": {}
    }
    
    # Contadores para resumen
    reg_meets_criteria = 0
    clf_meets_criteria = 0
    
    # Seleccionar mejores modelos de regresión
    logger.info("📊 REGRESIÓN - Selección de mejores modelos:")
    for target, models_results in regression_metrics.items():
        best_model = None
        best_score = -np.inf if regression_metric == 'r2' else np.inf
        
        for model_name, metrics in models_results.items():
            score = metrics['test'][regression_metric]
            
            # Para R² maximizar, para RMSE/MAE minimizar
            if regression_metric == 'r2':
                if score > best_score:
                    best_score = score
                    best_model = model_name
            else:
                if score < best_score:
                    best_score = score
                    best_model = model_name
        
        # Evaluar criterios de éxito
        test_r2 = models_results[best_model]['test']['r2']
        test_rmse = models_results[best_model]['test']['rmse']
        
        meets_criteria_reg = (
            test_r2 >= success_criteria.get('regression', {}).get('min_r2', 0.85) and
            test_rmse <= success_criteria.get('regression', {}).get('max_rmse', 500)
        )
        
        if meets_criteria_reg:
            reg_meets_criteria += 1
        
        best_models_report["regression"][target] = {
            "best_model": best_model,
            "best_score": float(best_score),
            "metric_used": regression_metric,
            "meets_criteria": meets_criteria_reg,
            "test_metrics": models_results[best_model]['test'],
            "all_models_comparison": {
                model: {
                    'r2': metrics['test']['r2'],
                    'rmse': metrics['test']['rmse'],
                    'mae': metrics['test']['mae']
                }
                for model, metrics in models_results.items()
            }
        }
        
        status = "✅" if meets_criteria_reg else "⚠️"
        logger.info(f"   {status} {target}: {best_model} ({regression_metric}={best_score:.3f})")
    
    # Seleccionar mejores modelos de clasificación
    logger.info("📊 CLASIFICACIÓN - Selección de mejores modelos:")
    for target, models_results in classification_metrics.items():
        best_model = None
        best_score = -np.inf
        
        for model_name, metrics in models_results.items():
            score = metrics['test'][classification_metric]
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        # Evaluar criterios de éxito
        test_f1 = models_results[best_model]['test']['f1']
        test_precision = models_results[best_model]['test']['precision']
        
        meets_criteria_clf = (
            test_f1 >= success_criteria.get('classification', {}).get('min_f1', 0.80) and
            test_precision >= success_criteria.get('classification', {}).get('min_precision', 0.85)
        )
        
        if meets_criteria_clf:
            clf_meets_criteria += 1
        
        best_models_report["classification"][target] = {
            "best_model": best_model,
            "best_score": float(best_score),
            "metric_used": classification_metric,
            "meets_criteria": meets_criteria_clf,
            "test_metrics": models_results[best_model]['test'],
            "all_models_comparison": {
                model: {
                    'f1': metrics['test']['f1'],
                    'accuracy': metrics['test']['accuracy'],
                    'precision': metrics['test']['precision'],
                    'recall': metrics['test']['recall']
                }
                for model, metrics in models_results.items()
            }
        }
        
        status = "✅" if meets_criteria_clf else "⚠️"
        logger.info(f"   {status} {target}: {best_model} ({classification_metric}={best_score:.3f})")
    
    # Resumen general
    total_reg_targets = len(regression_metrics)
    total_clf_targets = len(classification_metrics)
    
    best_models_report["summary"] = {
        "total_regression_targets": total_reg_targets,
        "total_classification_targets": total_clf_targets,
        "regression_meeting_criteria": reg_meets_criteria,
        "classification_meeting_criteria": clf_meets_criteria,
        "regression_success_rate": round(reg_meets_criteria / total_reg_targets * 100, 1) if total_reg_targets > 0 else 0,
        "classification_success_rate": round(clf_meets_criteria / total_clf_targets * 100, 1) if total_clf_targets > 0 else 0,
        "selection_criteria": {
            "regression": regression_metric,
            "classification": classification_metric
        }
    }
    
    # Evaluación global
    overall_success = (reg_meets_criteria + clf_meets_criteria) / (total_reg_targets + total_clf_targets) * 100
    
    best_models_report["criteria_evaluation"] = {
        "overall_success_rate": round(overall_success, 1),
        "status": "Excelente" if overall_success >= 90 else "Bueno" if overall_success >= 70 else "Necesita mejora",
        "regression_criteria": success_criteria.get('regression', {}),
        "classification_criteria": success_criteria.get('classification', {}),
        "recommendations": []
    }
    
    # Generar recomendaciones
    if reg_meets_criteria < total_reg_targets:
        best_models_report["criteria_evaluation"]["recommendations"].append(
            f"Optimizar {total_reg_targets - reg_meets_criteria} modelos de regresión que no cumplen criterios"
        )
    
    if clf_meets_criteria < total_clf_targets:
        best_models_report["criteria_evaluation"]["recommendations"].append(
            f"Mejorar {total_clf_targets - clf_meets_criteria} modelos de clasificación bajo umbral"
        )
    
    if overall_success >= 90:
        best_models_report["criteria_evaluation"]["recommendations"].append(
            "✅ Modelos cumplen con criterios de la rúbrica. Proceder con deployment."
        )
    
    logger.info(f"🏆 Selección completada - Tasa de éxito global: {overall_success:.1f}%")
    logger.info("✅ Selección de mejores modelos finalizada")
    
    return best_models_report

# =====================================================
# ANÁLISIS DE FEATURE IMPORTANCE
# =====================================================

def analyze_feature_importance(
    trained_regression_models: Dict[str, Any],
    trained_classification_models: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analiza la importancia de features en modelos entrenados.
    
    Extrae feature importance de modelos tree-based (Random Forest, Gradient Boosting)
    y coeficientes de modelos lineales para interpretabilidad.
    
    Args:
        trained_regression_models (Dict): Modelos de regresión entrenados
        trained_classification_models (Dict): Modelos de clasificación entrenados
        
    Returns:
        Dict[str, Any]: Feature importance por modelo y tipo:
            {
                'regression': {model_target_name: [importance_values]},
                'classification': {model_target_name: [importance_values]},
                'summary': {top_features, analysis}
            }
            
    Notes:
        - Tree-based models: usa feature_importances_
        - Linear models: usa abs(coef_)
        - SVM: no proporciona feature importance interpretable
        - Valores normalizados a suma 1.0
        
    Example:
        >>> importance = analyze_feature_importance(reg_models, clf_models)
        >>> top_feature_idx = np.argmax(importance['regression']['target_cases_rf'])
        
    See Also:
        create_feature_importance_plot: Para visualización en pipeline reporting
    """
    logger.info("🔍 Analizando importancia de features")
    
    feature_importance_analysis = {
        "regression": {},
        "classification": {},
        "summary": {
            "total_models_analyzed": 0,
            "models_with_importance": 0
        }
    }
    
    total_analyzed = 0
    with_importance = 0
    
    # Analizar modelos de regresión
    logger.info("📊 Extrayendo feature importance - REGRESIÓN:")
    for target, models in trained_regression_models.items():
        for model_name, model in models.items():
            total_analyzed += 1
            key = f"{target}_{model_name}"
            
            if hasattr(model, 'feature_importances_'):
                # Modelos tree-based (Random Forest, Gradient Boosting)
                importance = model.feature_importances_
                feature_importance_analysis["regression"][key] = importance.tolist()
                with_importance += 1
                
                top_3_idx = np.argsort(importance)[-3:][::-1]
                logger.debug(f"   ✓ {key}: Top 3 features = {top_3_idx.tolist()}")
                
            elif hasattr(model, 'coef_'):
                # Modelos lineales (Linear, Ridge, Lasso)
                importance = np.abs(model.coef_)
                if importance.ndim > 1:
                    importance = importance.flatten()
                feature_importance_analysis["regression"][key] = importance.tolist()
                with_importance += 1
                
                top_3_idx = np.argsort(importance)[-3:][::-1]
                logger.debug(f"   ✓ {key}: Top 3 coefs = {top_3_idx.tolist()}")
            else:
                logger.debug(f"   ⊘ {key}: Sin feature importance disponible")
    
    # Analizar modelos de clasificación
    logger.info("📊 Extrayendo feature importance - CLASIFICACIÓN:")
    for target, models in trained_classification_models.items():
        for model_name, model in models.items():
            total_analyzed += 1
            key = f"{target}_{model_name}"
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based classifiers
                importance = model.feature_importances_
                feature_importance_analysis["classification"][key] = importance.tolist()
                with_importance += 1
                
                top_3_idx = np.argsort(importance)[-3:][::-1]
                logger.debug(f"   ✓ {key}: Top 3 features = {top_3_idx.tolist()}")
                
            elif hasattr(model, 'coef_'):
                # Linear classifiers (Logistic Regression)
                importance = np.abs(model.coef_)
                if importance.ndim > 1:
                    # Para multiclase, promediar importancia entre clases
                    importance = np.mean(importance, axis=0)
                importance = importance.flatten()
                feature_importance_analysis["classification"][key] = importance.tolist()
                with_importance += 1
                
                top_3_idx = np.argsort(importance)[-3:][::-1]
                logger.debug(f"   ✓ {key}: Top 3 coefs = {top_3_idx.tolist()}")
            else:
                logger.debug(f"   ⊘ {key}: Sin feature importance disponible")
    
    # Actualizar resumen
    feature_importance_analysis["summary"]["total_models_analyzed"] = total_analyzed
    feature_importance_analysis["summary"]["models_with_importance"] = with_importance
    feature_importance_analysis["summary"]["coverage_percentage"] = round(
        (with_importance / total_analyzed * 100) if total_analyzed > 0 else 0, 1
    )
    
    logger.info(f"✅ Feature importance analizada: {with_importance}/{total_analyzed} modelos ({feature_importance_analysis['summary']['coverage_percentage']:.1f}%)")
    logger.info("✅ Análisis de feature importance completado")
    
    return feature_importance_analysis

# =====================================================
# HYPERPARAMETER TUNING CON GRIDSEARCHCV
# =====================================================

def tune_regression_models(
    regression_splits: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Realiza búsqueda de hiperparámetros con GridSearchCV para modelos de regresión.
    
    Implementa k-fold cross-validation (k=5) y guarda resultados con mean±std
    para cumplir con requisitos de la rúbrica.
    
    Args:
        regression_splits: Diccionario con train/val/test splits
        params: Parámetros con grids de búsqueda
        
    Returns:
        Dict con mejores modelos, parámetros y scores con CV
    """
    from sklearn.model_selection import GridSearchCV
    
    logger.info("🔍 Iniciando GridSearchCV para REGRESIÓN")
    
    train_df = regression_splits['train']
    feature_cols = regression_splits['feature_columns']
    target_cols = regression_splits['target_columns']
    
    X_train = train_df[feature_cols].fillna(0)
    
    cv_folds = params.get('cv_folds', 5)
    n_jobs = params.get('n_jobs', -1)
    verbose = params.get('verbose', 1)
    
    tuned_models = {}
    tuning_results = {}
    
    for target in target_cols:
        logger.info(f"📌 Tuning para target: {target}")
        y_train = train_df[target].fillna(0)
        
        target_models = {}
        target_results = {}
        
        grids = params.get('regression_grids', {})
        
        # Random Forest
        if grids.get('random_forest', {}).get('enabled', True):
            try:
                logger.info("   🔧 GridSearch: Random Forest...")
                rf_grid = grids['random_forest']['param_grid']
                
                grid_search = GridSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=n_jobs),
                    param_grid=rf_grid,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['random_forest'] = grid_search.best_estimator_
                target_results['random_forest'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_]),
                    'mean_fit_time': float(np.mean(grid_search.cv_results_['mean_fit_time']))
                }
                
                logger.info(f"      ✓ Best R²: {grid_search.best_score_:.3f} ± "
                           f"{grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.3f}")
                logger.info(f"      ✓ Params: {grid_search.best_params_}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en RF GridSearch: {e}")
        
        # Gradient Boosting
        if grids.get('gradient_boosting', {}).get('enabled', True):
            try:
                logger.info("   🔧 GridSearch: Gradient Boosting...")
                gb_grid = grids['gradient_boosting']['param_grid']
                
                grid_search = GridSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    param_grid=gb_grid,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['gradient_boosting'] = grid_search.best_estimator_
                target_results['gradient_boosting'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_]),
                    'mean_fit_time': float(np.mean(grid_search.cv_results_['mean_fit_time']))
                }
                
                logger.info(f"      ✓ Best R²: {grid_search.best_score_:.3f} ± "
                           f"{grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.3f}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en GB GridSearch: {e}")
        
        # XGBoost
        if grids.get('xgboost', {}).get('enabled', True):
            try:
                from xgboost import XGBRegressor
                logger.info("   🔧 GridSearch: XGBoost...")
                xgb_grid = grids['xgboost']['param_grid']
                
                grid_search = GridSearchCV(
                    XGBRegressor(random_state=42, n_jobs=n_jobs, verbosity=0),
                    param_grid=xgb_grid,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['xgboost'] = grid_search.best_estimator_
                target_results['xgboost'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_]),
                    'mean_fit_time': float(np.mean(grid_search.cv_results_['mean_fit_time']))
                }
                
                logger.info(f"      ✓ Best R²: {grid_search.best_score_:.3f} ± "
                           f"{grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.3f}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en XGB GridSearch: {e}")
        
        # Ridge
        if grids.get('ridge', {}).get('enabled', True):
            try:
                logger.info("   🔧 GridSearch: Ridge...")
                ridge_grid = grids['ridge']['param_grid']
                
                grid_search = GridSearchCV(
                    Ridge(random_state=42),
                    param_grid=ridge_grid,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['ridge'] = grid_search.best_estimator_
                target_results['ridge'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_]),
                    'mean_fit_time': float(np.mean(grid_search.cv_results_['mean_fit_time']))
                }
                
                logger.info(f"      ✓ Best R²: {grid_search.best_score_:.3f}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en Ridge GridSearch: {e}")
        
        # Lasso
        if grids.get('lasso', {}).get('enabled', True):
            try:
                logger.info("   🔧 GridSearch: Lasso...")
                lasso_grid = grids['lasso']['param_grid']
                
                grid_search = GridSearchCV(
                    Lasso(random_state=42, max_iter=10000),
                    param_grid=lasso_grid,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=n_jobs,
                    verbose=verbose
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['lasso'] = grid_search.best_estimator_
                target_results['lasso'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_]),
                    'mean_fit_time': float(np.mean(grid_search.cv_results_['mean_fit_time']))
                }
                
                logger.info(f"      ✓ Best R²: {grid_search.best_score_:.3f}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en Lasso GridSearch: {e}")
        
        tuned_models[target] = target_models
        tuning_results[target] = target_results
    
    logger.info(f"✅ GridSearchCV completado: {len(tuning_results)} targets tunados")
    
    return {
        'tuned_models': tuned_models,
        'tuning_results': tuning_results
    }


def tune_classification_models(
    classification_splits: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    GridSearchCV para modelos de clasificación con k-fold=5.
    """
    from sklearn.model_selection import GridSearchCV
    
    logger.info("🔍 Iniciando GridSearchCV para CLASIFICACIÓN")
    
    train_df = classification_splits['train']
    feature_cols = classification_splits['feature_columns']
    target_cols = classification_splits['target_columns']
    
    X_train = train_df[feature_cols].fillna(0)
    
    cv_folds = params.get('cv_folds', 5)
    n_jobs = params.get('n_jobs', -1)
    
    # ✅ INICIALIZAR TODAS LAS VARIABLES
    tuned_models = {}
    tuning_results = {}
    total_models = 0  # ← AGREGADO
    
    for target in target_cols:
        logger.info(f"📌 Tuning para target: {target}")
        y_train = train_df[target].fillna(0).astype(int)
        
        # Saltar targets con una sola clase
        if len(np.unique(y_train)) < 2:
            logger.warning(f"   ⊘ Target {target} tiene solo 1 clase, saltando...")
            continue
        
        target_models = {}
        target_results = {}
        
        grids = params.get('classification_grids', {})
        
        # 1. Random Forest con GridSearch
        if grids.get('random_forest', {}).get('enabled', True):
            try:
                logger.info("   🔧 GridSearch: Random Forest...")
                rf_grid = grids['random_forest']['param_grid']
                
                grid_search = GridSearchCV(
                    RandomForestClassifier(random_state=42, n_jobs=n_jobs),
                    param_grid=rf_grid,
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=n_jobs
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['random_forest'] = grid_search.best_estimator_
                target_results['random_forest'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                }
                
                total_models += 1
                logger.info(f"      ✓ Best F1: {grid_search.best_score_:.3f} ± "
                           f"{grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.3f}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en Random Forest: {e}")
        
        # 2. Gradient Boosting con GridSearch
        if grids.get('gradient_boosting', {}).get('enabled', True):
            try:
                logger.info("   🔧 GridSearch: Gradient Boosting...")
                gb_grid = grids['gradient_boosting']['param_grid']
                
                grid_search = GridSearchCV(
                    GradientBoostingClassifier(random_state=42),
                    param_grid=gb_grid,
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=n_jobs
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['gradient_boosting'] = grid_search.best_estimator_
                target_results['gradient_boosting'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                }
                
                total_models += 1
                logger.info(f"      ✓ Best F1: {grid_search.best_score_:.3f}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en Gradient Boosting: {e}")
        
        # 3. Logistic Regression con GridSearch
        if grids.get('logistic_regression', {}).get('enabled', True):
            try:
                logger.info("   🔧 GridSearch: Logistic Regression...")
                lr_grid = grids['logistic_regression']['param_grid']
                
                grid_search = GridSearchCV(
                    LogisticRegression(random_state=42, class_weight='balanced'),
                    param_grid=lr_grid,
                    cv=cv_folds,
                    scoring='f1_weighted',
                    n_jobs=n_jobs
                )
                
                grid_search.fit(X_train, y_train)
                
                target_models['logistic_regression'] = grid_search.best_estimator_
                target_results['logistic_regression'] = {
                    'best_params': grid_search.best_params_,
                    'best_score': float(grid_search.best_score_),
                    'cv_mean': float(grid_search.best_score_),
                    'cv_std': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                }
                
                total_models += 1
                logger.info(f"      ✓ Best F1: {grid_search.best_score_:.3f}")
                
            except Exception as e:
                logger.error(f"      ✗ Error en Logistic Regression: {e}")
        
        # 4. SVM (opcional - deshabilitado por defecto)
        if params.get("svm", {}).get("enabled", False):
            try:
                logger.info("   🔧 Training: SVM...")
                svm = SVC(
                    kernel=params.get("svm", {}).get("svm_kernel", 'rbf'),
                    C=params.get("svm", {}).get("C", 1.0),
                    gamma=params.get("svm", {}).get("gamma", 'scale'),
                    class_weight='balanced',
                    random_state=42
                )
                svm.fit(X_train, y_train)
                target_models['svm'] = svm
                total_models += 1
                logger.debug(f"      ✓ SVM entrenado")
            except Exception as e:
                logger.error(f"      ✗ Error en SVM: {e}")
        
        # 5. K-Neighbors Classifier (opcional)
        if params.get("k_neighbors_classifier", {}).get("enabled", False):
            try:
                logger.info("   🔧 Training: K-Neighbors...")
                n_neighbors = params.get("k_neighbors_classifier", {}).get("n_neighbors", 5)
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
                knn.fit(X_train, y_train)
                target_models['k_neighbors_classifier'] = knn
                total_models += 1
                logger.debug(f"      ✓ K-Neighbors entrenado (n_neighbors={n_neighbors})")
            except Exception as e:
                logger.error(f"      ✗ Error en K-Neighbors Classifier: {e}")
        
        # Guardar modelos y resultados del target
        tuned_models[target] = target_models
        tuning_results[target] = target_results
    
    logger.info(f"✅ GridSearchCV completado: {len(tuning_results)} targets tunados, {total_models} modelos totales")
    
    return {
        'tuned_models': tuned_models,
        'tuning_results': tuning_results
    }