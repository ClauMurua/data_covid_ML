import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def validate_raw_data(
    df_2020: pd.DataFrame, 
    df_2021: pd.DataFrame, 
    df_2022: pd.DataFrame
) -> Dict[str, Any]:
    """
    Valida la calidad y estructura de los datasets raw.
    """
    logger.info("Iniciando validación de datos raw")
    
    validation_report = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
        "summary": {}
    }
    
    datasets = {"2020": df_2020, "2021": df_2021, "2022": df_2022}
    
    for year, df in datasets.items():
        logger.info(f"Validando dataset {year}: {df.shape}")
        
        # Convertir tipos a formato serializable
        dtypes_serializable = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing_values_serializable = {col: int(count) for col, count in df.isnull().sum().items()}
        
        validation_report["datasets"][year] = {
            "shape": list(df.shape),  # Convertir tuple a list
            "columns": df.columns.tolist(),
            "dtypes": dtypes_serializable,
            "missing_values": missing_values_serializable,
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # Validaciones específicas
        if 'date' in df.columns:
            validation_report["datasets"][year]["date_range"] = {
                "min": str(df['date'].min()) if pd.notna(df['date'].min()) else "No válido",
                "max": str(df['date'].max()) if pd.notna(df['date'].max()) else "No válido"
            }
    
    # Resumen general
    total_rows = sum(df.shape[0] for df in datasets.values())
    validation_report["summary"] = {
        "total_datasets": len(datasets),
        "total_rows": int(total_rows),
        "validation_passed": total_rows > 0,
        "consistent_structure": len(set(len(df.columns) for df in datasets.values())) <= 2
    }
    
    logger.info(f"Validación completada. Total filas: {total_rows}")
    return validation_report

def clean_covid_2020(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia y estandariza el dataset de COVID 2020.
    
    Args:
        df: Dataset raw 2020
        params: Parámetros de limpieza
        
    Returns:
        DataFrame limpio
    """
    logger.info("Limpiando datos COVID 2020")
    
    df_clean = df.copy()
    
    # Limpiar nombres de columnas
    df_clean.columns = df_clean.columns.str.strip().str.lower()
    
    # Convertir fechas
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Crear columna año si no existe
    if 'year' not in df_clean.columns:
        df_clean['year'] = 2020
    
    # Limpiar valores numéricos
    numeric_columns = ['new_confirmed', 'new_deceased', 'new_recovered', 'new_tested',
                      'cumulative_confirmed', 'cumulative_deceased', 'cumulative_recovered', 'cumulative_tested']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # Remover outliers extremos usando IQR
    if params.get("outlier_method") == "iqr":
        for col in ['new_confirmed', 'new_deceased']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                factor = params.get("outlier_factor", 1.5)
                
                # Mantener valores razonables (no eliminar completamente)
                upper_bound = Q3 + factor * IQR
                df_clean[col] = df_clean[col].clip(upper=upper_bound)
    
    logger.info(f"Dataset 2020 limpio: {df_clean.shape}")
    return df_clean

def clean_covid_2021(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia y estandariza el dataset de COVID 2021.
    Maneja el formato especial de fechas dd-mm-yyyy.
    
    Args:
        df: Dataset raw 2021
        params: Parámetros de limpieza
        
    Returns:
        DataFrame limpio
    """
    logger.info("Limpiando datos COVID 2021")
    
    df_clean = df.copy()
    
    # Limpiar nombres de columnas
    df_clean.columns = df_clean.columns.str.strip().str.lower()
    
    # Convertir fechas con formato específico
    date_format = params.get("date_format_2021", "%d-%m-%Y")
    try:
        df_clean['date'] = pd.to_datetime(df_clean['date'], format=date_format, errors='coerce')
        logger.info("Fechas convertidas con formato dd-mm-yyyy")
    except:
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'], dayfirst=True, errors='coerce')
            logger.info("Fechas convertidas con dayfirst=True")
        except:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            logger.warning("Conversión de fechas con método genérico")
    
    # Crear columna año si no existe
    if 'year' not in df_clean.columns:
        df_clean['year'] = 2021
    
    # Limpiar valores numéricos
    numeric_columns = ['new_confirmed', 'new_deceased', 'new_recovered', 'new_tested',
                      'cumulative_confirmed', 'cumulative_deceased', 'cumulative_recovered', 'cumulative_tested']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    logger.info(f"Dataset 2021 limpio: {df_clean.shape}")
    return df_clean

def clean_covid_2022(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia y estandariza el dataset de COVID 2022.
    
    Args:
        df: Dataset raw 2022
        params: Parámetros de limpieza
        
    Returns:
        DataFrame limpio
    """
    logger.info("Limpiando datos COVID 2022")
    
    df_clean = df.copy()
    
    # Limpiar nombres de columnas
    df_clean.columns = df_clean.columns.str.strip().str.lower()
    
    # Convertir fechas
    df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Crear columna año si no existe
    if 'year' not in df_clean.columns:
        df_clean['year'] = 2022
    
    # Limpiar valores numéricos
    numeric_columns = ['new_confirmed', 'new_deceased', 'new_recovered', 'new_tested',
                      'cumulative_confirmed', 'cumulative_deceased', 'cumulative_recovered', 'cumulative_tested']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    logger.info(f"Dataset 2022 limpio: {df_clean.shape}")
    return df_clean

def integrate_datasets(
    df_2020: pd.DataFrame, 
    df_2021: pd.DataFrame, 
    df_2022: pd.DataFrame
) -> pd.DataFrame:
    """
    Integra los datasets de los tres años en uno solo.
    
    Args:
        df_2020: Dataset limpio 2020
        df_2021: Dataset limpio 2021
        df_2022: Dataset limpio 2022
        
    Returns:
        DataFrame integrado
    """
    logger.info("Integrando datasets de tres años")
    
    # Identificar columnas comunes
    common_columns = set(df_2020.columns) & set(df_2021.columns) & set(df_2022.columns)
    logger.info(f"Columnas comunes identificadas: {len(common_columns)}")
    
    # Seleccionar solo columnas comunes para cada dataset
    df_2020_select = df_2020[list(common_columns)].copy()
    df_2021_select = df_2021[list(common_columns)].copy()
    df_2022_select = df_2022[list(common_columns)].copy()
    
    # Concatenar datasets
    df_integrated = pd.concat([df_2020_select, df_2021_select, df_2022_select], 
                             ignore_index=True, sort=False)
    
    # Ordenar por fecha y ubicación
    df_integrated = df_integrated.sort_values(['date', 'location_key']).reset_index(drop=True)
    
    # Crear columnas derivadas para análisis
    df_integrated['month'] = df_integrated['date'].dt.month
    df_integrated['quarter'] = df_integrated['date'].dt.quarter
    df_integrated['day_of_week'] = df_integrated['date'].dt.dayofweek
    df_integrated['week_of_year'] = df_integrated['date'].dt.isocalendar().week
    
    logger.info(f"Integración completada: {df_integrated.shape}")
    logger.info(f"Rango temporal: {df_integrated['date'].min()} - {df_integrated['date'].max()}")
    logger.info(f"Ubicaciones únicas: {df_integrated['location_key'].nunique()}")
    
    return df_integrated

def create_national_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea agregación nacional diaria de los datos COVID.
    
    Args:
        df: Dataset integrado completo
        
    Returns:
        DataFrame con datos nacionales diarios
    """
    logger.info("Creando agregación nacional diaria")
    
    # Agrupar por fecha para obtener totales nacionales
    df_national = df.groupby('date').agg({
        'new_confirmed': 'sum',
        'new_deceased': 'sum', 
        'new_recovered': 'sum',
        'cumulative_confirmed': 'max',  # Usar max para acumulativos
        'cumulative_deceased': 'max',
        'cumulative_recovered': 'max'
    }).reset_index()
    
    # Calcular promedios móviles para suavizar series
    for days in [3, 7, 14]:
        df_national[f'new_confirmed_ma{days}'] = df_national['new_confirmed'].rolling(
            window=days, center=True, min_periods=1
        ).mean()
        df_national[f'new_deceased_ma{days}'] = df_national['new_deceased'].rolling(
            window=days, center=True, min_periods=1
        ).mean()
    
    # Calcular tasas y métricas derivadas
    df_national['case_fatality_rate'] = np.where(
        df_national['cumulative_confirmed'] > 0,
        (df_national['cumulative_deceased'] / df_national['cumulative_confirmed'] * 100).round(3),
        0
    )
    
    # Calcular tasas de crecimiento
    df_national['confirmed_growth_rate'] = df_national['new_confirmed'].pct_change().fillna(0)
    df_national['deceased_growth_rate'] = df_national['new_deceased'].pct_change().fillna(0)
    
    # Agregar información temporal
    df_national['year'] = df_national['date'].dt.year
    df_national['month'] = df_national['date'].dt.month
    df_national['quarter'] = df_national['date'].dt.quarter
    df_national['day_of_week'] = df_national['date'].dt.dayofweek
    df_national['is_weekend'] = df_national['day_of_week'].isin([5, 6])
    
    logger.info(f"Agregación nacional creada: {df_national.shape}")
    return df_national

def quality_assessment(
    df_complete: pd.DataFrame, 
    df_national: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evalúa la calidad final de los datos procesados.
    
    Args:
        df_complete: Dataset completo integrado
        df_national: Dataset nacional diario
        
    Returns:
        Dict con reporte de calidad
    """
    logger.info("Evaluando calidad de datos procesados")
    
    quality_report = {
        "timestamp": datetime.now().isoformat(),
        "complete_dataset": {},
        "national_dataset": {},
        "overall_assessment": {}
    }
    
    # Evaluación dataset completo
    quality_report["complete_dataset"] = {
        "shape": df_complete.shape,
        "date_range": {
            "start": df_complete['date'].min().isoformat(),
            "end": df_complete['date'].max().isoformat(),
            "total_days": (df_complete['date'].max() - df_complete['date'].min()).days
        },
        "missing_values_pct": round(df_complete.isnull().sum().sum() / df_complete.size * 100, 2),
        "duplicate_rows": int(df_complete.duplicated().sum()),
        "unique_locations": df_complete['location_key'].nunique(),
        "data_consistency": {
            "negative_values": int((df_complete.select_dtypes(include=[np.number]) < 0).sum().sum()),
            "zero_variance_columns": int((df_complete.select_dtypes(include=[np.number]).var() == 0).sum())
        }
    }
    
    # Evaluación dataset nacional
    quality_report["national_dataset"] = {
        "shape": df_national.shape,
        "completeness_pct": round((1 - df_national.isnull().sum().sum() / df_national.size) * 100, 2),
        "temporal_consistency": {
            "consecutive_dates": bool(len(pd.date_range(
                df_national['date'].min(), df_national['date'].max(), freq='D'
            )) == len(df_national)),
            "missing_dates": len(pd.date_range(
                df_national['date'].min(), df_national['date'].max(), freq='D'
            )) - len(df_national)
        }
    }
    
    # Evaluación general
    quality_score = (
        (100 - quality_report["complete_dataset"]["missing_values_pct"]) * 0.3 +
        quality_report["national_dataset"]["completeness_pct"] * 0.3 +
        (100 if quality_report["national_dataset"]["temporal_consistency"]["consecutive_dates"] else 80) * 0.4
    )
    
    quality_report["overall_assessment"] = {
        "quality_score": round(quality_score, 1),
        "grade": "Excelente" if quality_score >= 95 else 
                "Bueno" if quality_score >= 85 else 
                "Aceptable" if quality_score >= 70 else "Requiere mejora",
        "ready_for_analysis": bool(quality_score >= 70),
        "recommended_next_steps": [
            "Proceder con feature engineering" if quality_score >= 70 else "Revisar limpieza de datos",
            "Análisis exploratorio detallado",
            "Identificación de targets para ML"
        ]
    }
    
    logger.info(f"Evaluación de calidad completada. Score: {quality_score:.1f}")
    return quality_report

def create_basic_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea features básicas para el análisis COVID.
    
    Args:
        df: Dataset nacional diario
        params: Parámetros de feature engineering
        
    Returns:
        DataFrame con features básicas
    """
    logger.info("Creando features básicas")
    
    df_featured = df.copy()
    
    # Variables lag básicas
    lag_days = params.get("basic_lags", [1, 3, 7, 14])
    for lag in lag_days:
        df_featured[f'confirmed_lag_{lag}'] = df_featured['new_confirmed'].shift(lag)
        df_featured[f'deceased_lag_{lag}'] = df_featured['new_deceased'].shift(lag)
    
    # Rolling statistics básicas
    windows = params.get("rolling_windows", [7, 14])
    for window in windows:
        df_featured[f'confirmed_rolling_mean_{window}'] = df_featured['new_confirmed'].rolling(window).mean()
        df_featured[f'confirmed_rolling_std_{window}'] = df_featured['new_confirmed'].rolling(window).std()
        df_featured[f'deceased_rolling_mean_{window}'] = df_featured['new_deceased'].rolling(window).mean()
    
    # Ratios epidemiológicos básicos
    df_featured['daily_fatality_rate'] = np.where(
        df_featured['new_confirmed'] > 0,
        df_featured['new_deceased'] / df_featured['new_confirmed'],
        0
    )
    
    # Variables de diferencias
    df_featured['confirmed_diff_1d'] = df_featured['new_confirmed'].diff(1)
    df_featured['confirmed_diff_7d'] = df_featured['new_confirmed'].diff(7)
    
    # Tendencias básicas
    df_featured['confirmed_trend_7d'] = df_featured['new_confirmed'].rolling(7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else np.nan
    )
    
    # Variables estacionales
    df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
    df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
    df_featured['day_of_week_sin'] = np.sin(2 * np.pi * df_featured['day_of_week'] / 7)
    df_featured['day_of_week_cos'] = np.cos(2 * np.pi * df_featured['day_of_week'] / 7)
    
    # Variables de volatilidad
    df_featured['confirmed_volatility_7d'] = df_featured['new_confirmed'].rolling(7).std()
    df_featured['confirmed_volatility_14d'] = df_featured['new_confirmed'].rolling(14).std()
    
    logger.info(f"Features básicas creadas: {df_featured.shape}")
    logger.info(f"Nuevas features: {df_featured.shape[1] - df.shape[1]}")
    return df_featured

def identify_preliminary_targets(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identifica targets preliminares para Machine Learning.
    
    Args:
        df: Dataset con features básicas
        params: Parámetros de targets
        
    Returns:
        DataFrame con targets preliminares identificados
    """
    logger.info("Identificando targets preliminares para Machine Learning")
    
    df_targets = df.copy()
    
    # TARGET 1: Casos futuros (regresión)
    future_horizon = params.get("prediction_horizon", 7)
    df_targets[f'target_cases_next_{future_horizon}d'] = df_targets['new_confirmed'].shift(-future_horizon)
    
    # TARGET 2: Crecimiento futuro (regresión)
    df_targets['target_growth_rate_next_14d'] = df_targets['confirmed_growth_rate'].shift(-14)
    
    # TARGET 3: Alta transmisión (clasificación binaria)
    threshold = df_targets['new_confirmed'].quantile(params.get("high_transmission_quantile", 0.75))
    df_targets['target_high_transmission'] = (df_targets['new_confirmed'] > threshold).astype(int)
    
    # TARGET 4: Nivel de riesgo (clasificación multiclase)
    def assign_risk_level(confirmed_cases):
        if pd.isna(confirmed_cases):
            return np.nan
        q33 = df_targets['new_confirmed'].quantile(0.33)
        q66 = df_targets['new_confirmed'].quantile(0.66)
        
        if confirmed_cases <= q33:
            return 0  # Bajo
        elif confirmed_cases <= q66:
            return 1  # Medio
        else:
            return 2  # Alto
    
    df_targets['target_risk_level'] = df_targets['new_confirmed'].apply(assign_risk_level)
    
    # TARGET 5: Tendencia (clasificación binaria)
    df_targets['target_trend_direction'] = np.where(
        df_targets['confirmed_trend_7d'] > 0, 1, 0
    )
    
    # Resumen de targets creados
    targets_created = [col for col in df_targets.columns if col.startswith('target_')]
    
    logger.info(f"Targets preliminares identificados: {len(targets_created)}")
    logger.info(f"Targets de regresión: {[t for t in targets_created if 'cases_next' in t or 'growth_rate' in t]}")
    logger.info(f"Targets de clasificación: {[t for t in targets_created if 'transmission' in t or 'risk_level' in t or 'trend' in t]}")
    
    return df_targets