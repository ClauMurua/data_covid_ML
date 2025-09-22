import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import datetime

logger = logging.getLogger(__name__)

def create_temporal_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea features temporales avanzadas para análisis de series de tiempo.
    
    Args:
        df: Dataset nacional diario
        params: Parámetros de feature engineering
        
    Returns:
        DataFrame con features temporales
    """
    logger.info("Creando features temporales avanzadas")
    
    df_temporal = df.copy()
    
    # Variables de lag
    lag_days = params.get("lag_days", [1, 3, 7, 14, 21])
    for lag in lag_days:
        df_temporal[f'confirmed_lag_{lag}'] = df_temporal['new_confirmed'].shift(lag)
        df_temporal[f'deceased_lag_{lag}'] = df_temporal['new_deceased'].shift(lag)
    
    # Rolling statistics
    windows = params.get("rolling_windows", [3, 7, 14, 30])
    for window in windows:
        # Medias móviles
        df_temporal[f'confirmed_rolling_mean_{window}'] = df_temporal['new_confirmed'].rolling(window).mean()
        df_temporal[f'confirmed_rolling_std_{window}'] = df_temporal['new_confirmed'].rolling(window).std()
        df_temporal[f'confirmed_rolling_max_{window}'] = df_temporal['new_confirmed'].rolling(window).max()
        df_temporal[f'confirmed_rolling_min_{window}'] = df_temporal['new_confirmed'].rolling(window).min()
    
    # Variables de tendencia
    df_temporal['confirmed_trend_7d'] = df_temporal['new_confirmed'].rolling(7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else np.nan
    )
    
    # Variables estacionales
    df_temporal['day_of_year'] = df_temporal['date'].dt.dayofyear
    df_temporal['week_of_year'] = df_temporal['date'].dt.isocalendar().week
    df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
    df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
    df_temporal['day_sin'] = np.sin(2 * np.pi * df_temporal['day_of_week'] / 7)
    df_temporal['day_cos'] = np.cos(2 * np.pi * df_temporal['day_of_week'] / 7)
    
    # Variables de aceleración (segunda derivada)
    df_temporal['confirmed_acceleration'] = df_temporal['new_confirmed'].diff().diff()
    
    logger.info(f"Features temporales creadas: {df_temporal.shape}")
    return df_temporal

def create_epidemiological_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea features epidemiológicas específicas para análisis COVID.
    
    Args:
        df: Dataset completo por ubicación
        params: Parámetros de feature engineering
        
    Returns:
        DataFrame con features epidemiológicas
    """
    logger.info("Creando features epidemiológicas")
    
    df_epi = df.copy()
    
    # Tasas epidemiológicas
    df_epi['case_fatality_rate_daily'] = np.where(
        df_epi['new_confirmed'] > 0,
        df_epi['new_deceased'] / df_epi['new_confirmed'],
        0
    )
    
    df_epi['recovery_rate'] = np.where(
        df_epi['cumulative_confirmed'] > 0,
        df_epi['cumulative_recovered'] / df_epi['cumulative_confirmed'],
        0
    )
    
    # Tasas de crecimiento
    df_epi['confirmed_growth_rate'] = df_epi.groupby('location_key')['cumulative_confirmed'].pct_change()
    df_epi['deceased_growth_rate'] = df_epi.groupby('location_key')['cumulative_deceased'].pct_change()
    
    # Ratios comparativos
    df_epi['new_vs_cumulative_ratio'] = np.where(
        df_epi['cumulative_confirmed'] > 0,
        df_epi['new_confirmed'] / df_epi['cumulative_confirmed'],
        0
    )
    
    # Intensidad epidemiológica (casos por 100k habitantes - simulado)
    # En un caso real usarías datos de población reales
    df_epi['population_estimated'] = 50000  # Valor por defecto
    df_epi['cases_per_100k'] = (df_epi['cumulative_confirmed'] / df_epi['population_estimated']) * 100000
    
    # Variables de volatilidad
    df_epi['confirmed_volatility_7d'] = df_epi.groupby('location_key')['new_confirmed'].transform(
        lambda x: x.rolling(7).std()
    )
    
    logger.info(f"Features epidemiológicas creadas: {df_epi.shape}")
    return df_epi

def create_regional_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea features basadas en análisis regional.
    
    Args:
        df: Dataset completo por ubicación
        params: Parámetros de feature engineering
        
    Returns:
        DataFrame con features regionales
    """
    logger.info("Creando features regionales")
    
    df_regional = df.copy()
    
    # Ranking de regiones por casos
    daily_regional_totals = df_regional.groupby(['date', 'location_key'])['new_confirmed'].sum().reset_index()
    daily_regional_totals['regional_rank'] = daily_regional_totals.groupby('date')['new_confirmed'].rank(
        method='dense', ascending=False
    )
    
    # Merge ranking back
    df_regional = df_regional.merge(
        daily_regional_totals[['date', 'location_key', 'regional_rank']], 
        on=['date', 'location_key'], 
        how='left'
    )
    
    # Categorización de regiones por intensidad
    region_intensity = df_regional.groupby('location_key')['new_confirmed'].mean()
    intensity_quantiles = region_intensity.quantile([0.33, 0.66])
    
    def categorize_region(location_key):
        avg_cases = region_intensity.get(location_key, 0)
        if avg_cases <= intensity_quantiles.iloc[0]:
            return 'low_intensity'
        elif avg_cases <= intensity_quantiles.iloc[1]:
            return 'medium_intensity'
        else:
            return 'high_intensity'
    
    df_regional['region_intensity_category'] = df_regional['location_key'].apply(categorize_region)
    
    # Features de comparación con promedio nacional
    national_daily = df_regional.groupby('date')['new_confirmed'].mean()
    df_regional = df_regional.set_index('date')
    df_regional['vs_national_ratio'] = df_regional['new_confirmed'] / df_regional.index.map(national_daily)
    df_regional = df_regional.reset_index()
    
    logger.info(f"Features regionales creadas: {df_regional.shape}")
    return df_regional

def apply_transformations(
    df_temporal: pd.DataFrame, 
    df_epi: pd.DataFrame, 
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Aplica transformaciones de scaling y encoding a las features.
    
    Args:
        df_temporal: Features temporales
        df_epi: Features epidemiológicas
        params: Parámetros de transformación
        
    Returns:
        DataFrame con transformaciones aplicadas
    """
    logger.info("Aplicando transformaciones a features")
    
    # Integrar datasets por fecha (asumiendo que df_temporal es nacional y df_epi tiene múltiples ubicaciones)
    # Usar datos nacionales como base
    df_base = df_temporal.copy()
    
    # Agregar features epidemiológicas agregadas por fecha
    epi_national = df_epi.groupby('date').agg({
        'case_fatality_rate_daily': 'mean',
        'recovery_rate': 'mean',
        'confirmed_growth_rate': 'mean',
        'cases_per_100k': 'mean',
        'confirmed_volatility_7d': 'mean'
    }).reset_index()
    
    df_transformed = df_base.merge(epi_national, on='date', how='left')
    
    # Identificar columnas numéricas para scaling
    numeric_columns = df_transformed.select_dtypes(include=[np.number]).columns
    scaling_columns = [col for col in numeric_columns if col not in ['year', 'month', 'day_of_week']]
    
    # Aplicar StandardScaler a features principales
    if params.get("apply_standard_scaling", True):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_transformed[scaling_columns].fillna(0))
        
        # Crear nombres de columnas escaladas
        scaled_columns = [f"{col}_scaled" for col in scaling_columns]
        scaled_df = pd.DataFrame(scaled_features, columns=scaled_columns, index=df_transformed.index)
        
        df_transformed = pd.concat([df_transformed, scaled_df], axis=1)
    
    # Aplicar MinMaxScaler a features cíclicas
    cyclical_features = [col for col in df_transformed.columns if any(x in col for x in ['_sin', '_cos'])]
    if cyclical_features and params.get("apply_minmax_scaling", True):
        minmax_scaler = MinMaxScaler()
        df_transformed[cyclical_features] = minmax_scaler.fit_transform(df_transformed[cyclical_features])
    
    # Encoding de variables categóricas (si las hay)
    categorical_columns = df_transformed.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col != 'date':
            le = LabelEncoder()
            df_transformed[f'{col}_encoded'] = le.fit_transform(df_transformed[col].astype(str))
    
    logger.info(f"Transformaciones aplicadas: {df_transformed.shape}")
    return df_transformed

def identify_ml_targets(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identifica y crea variables objetivo para problemas de ML.
    
    Args:
        df: Dataset con features transformadas
        params: Parámetros para definición de targets
        
    Returns:
        DataFrame con targets de ML
    """
    logger.info("Identificando targets para Machine Learning")
    
    df_targets = df.copy()
    
    # TARGET 1: Regresión - Predicción de casos futuros
    future_days = params.get("prediction_horizon", 7)
    df_targets[f'target_confirmed_next_{future_days}d'] = df_targets['new_confirmed'].shift(-future_days)
    
    # TARGET 2: Regresión - Tasa de crecimiento futura
    df_targets['target_growth_rate_next_14d'] = df_targets['confirmed_growth_rate'].shift(-14)
    
    # TARGET 3: Clasificación - Período de alta transmisión
    confirmed_threshold = df_targets['new_confirmed'].quantile(params.get("high_transmission_quantile", 0.75))
    df_targets['target_high_transmission'] = (df_targets['new_confirmed'] > confirmed_threshold).astype(int)
    
    # TARGET 4: Clasificación - Nivel de riesgo (multiclase)
    def assign_risk_level(confirmed_cases):
        if confirmed_cases <= df_targets['new_confirmed'].quantile(0.33):
            return 0  # Bajo
        elif confirmed_cases <= df_targets['new_confirmed'].quantile(0.66):
            return 1  # Medio
        else:
            return 2  # Alto
    
    df_targets['target_risk_level'] = df_targets['new_confirmed'].apply(assign_risk_level)
    
    # TARGET 5: Clasificación - Tendencia (subida/bajada)
    df_targets['target_trend_direction'] = np.where(
        df_targets['confirmed_trend_7d'] > 0, 1, 0
    )
    
    # Información de targets para logging
    targets_info = {
        'regression_targets': [f'target_confirmed_next_{future_days}d', 'target_growth_rate_next_14d'],
        'classification_targets': ['target_high_transmission', 'target_risk_level', 'target_trend_direction']
    }
    
    logger.info(f"Targets creados: {targets_info}")
    return df_targets

def prepare_ml_datasets(
    df: pd.DataFrame, 
    df_targets: pd.DataFrame, 
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara datasets finales separados para regresión y clasificación.
    
    Args:
        df: Features transformadas
        df_targets: Targets identificados
        params: Parámetros de preparación
        
    Returns:
        Tuple con datasets de regresión y clasificación
    """
    logger.info("Preparando datasets finales para ML")
    
    # Combinar features y targets
    df_ml = df.merge(df_targets[['date'] + [col for col in df_targets.columns if col.startswith('target_')]], 
                     on='date', how='inner')
    
    # Remover filas con targets NaN (por shifts)
    df_ml_clean = df_ml.dropna()
    
    # Identificar columnas de features (excluir targets y metadata)
    feature_columns = [col for col in df_ml_clean.columns if not col.startswith('target_') and col != 'date']
    
    # Dataset para regresión
    regression_targets = ['target_confirmed_next_7d', 'target_growth_rate_next_14d']
    regression_dataset = df_ml_clean[feature_columns + regression_targets + ['date']].copy()
    
    # Dataset para clasificación
    classification_targets = ['target_high_transmission', 'target_risk_level', 'target_trend_direction']
    classification_dataset = df_ml_clean[feature_columns + classification_targets + ['date']].copy()
    
    logger.info(f"Dataset regresión: {regression_dataset.shape}")
    logger.info(f"Dataset clasificación: {classification_dataset.shape}")
    logger.info(f"Features disponibles: {len(feature_columns)}")
    
    return regression_dataset, classification_dataset