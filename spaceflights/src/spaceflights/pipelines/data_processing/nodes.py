import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import datetime

logger = logging.getLogger(__name__)

def combine_all_years(
    df_2020: pd.DataFrame,
    df_2021: pd.DataFrame,
    df_2022: pd.DataFrame
) -> pd.DataFrame:
    """
    Combina datos de todos los años FILTRANDO solo regiones principales.
    
    OPTIMIZADO: Usa solo nivel nacional + 16 regiones (sin comunas) para:
    - Reducir consumo de memoria
    - Mantener granularidad regional
    - ~18,600 registros en vez de 99,193
    
    Returns:
        DataFrame con ~17 ubicaciones × 1095 días = ~18,600 registros
    """
    logger.info("🔄 Combinando datos 2020-2022 (nivel regional optimizado)")
    
    # Combinar años
    df_combined = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)
    
    logger.info(f"📊 Datos originales: {len(df_combined):,} registros, {df_combined['location_key'].nunique()} ubicaciones")
    
    # FILTRAR: Solo nivel nacional (CL) + regiones (2 caracteres después de CL_)
    # Ejemplo: CL, CL_RM, CL_BI (incluir)
    # Excluir: CL_RM_13101 (comunas - 3+ segmentos)
    def is_regional_level(location_key):
        """
        Retorna True si es nivel nacional o regional (no comunal).
        Ejemplos:
        - 'CL' → True (nacional)
        - 'CL_RM' → True (región)
        - 'CL_RM_13101' → False (comuna)
        """
        if location_key == 'CL':
            return True
        parts = location_key.split('_')
        return len(parts) == 2  # CL_XX = regional
    
    # Aplicar filtro
    df_combined['is_regional'] = df_combined['location_key'].apply(is_regional_level)
    df_filtered = df_combined[df_combined['is_regional']].drop('is_regional', axis=1).copy()
    
    logger.info(f"✅ Datos filtrados: {len(df_filtered):,} registros")
    logger.info(f"✅ Ubicaciones seleccionadas: {df_filtered['location_key'].nunique()}")
    logger.info(f"✅ Ubicaciones: {sorted(df_filtered['location_key'].unique())[:10]}...")
    
    # Convertir fecha con formato correcto
    df_filtered['date'] = pd.to_datetime(df_filtered['date'], format='%d-%m-%Y', errors='coerce')
    
    # Ordenar
    df_filtered = df_filtered.sort_values(['location_key', 'date']).reset_index(drop=True)
    
    # Rellenar NaN
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(0)
    
    # Variables temporales
    df_filtered['month'] = df_filtered['date'].dt.month
    df_filtered['day_of_week'] = df_filtered['date'].dt.dayofweek
    df_filtered['quarter'] = df_filtered['date'].dt.quarter
    
    logger.info(f"✅ Rango temporal: {df_filtered['date'].min()} a {df_filtered['date'].max()}")
    logger.info(f"✅ Dataset final optimizado: {df_filtered.shape}")
    
    return df_filtered

def create_temporal_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea features temporales avanzadas para análisis de series de tiempo COVID-19.
    
    Esta función genera múltiples tipos de features temporales incluyendo variables
    de rezago (lag), estadísticas rodantes (rolling), tendencias, componentes 
    estacionales y aceleraciones para capturar patrones temporales complejos.
    
    Args:
        df (pd.DataFrame): Dataset nacional diario con datos COVID-19. Debe contener:
            - 'date': Columna de fecha en formato datetime
            - 'new_confirmed': Casos confirmados diarios
            - 'new_deceased': Muertes diarias
            - 'month': Mes del año (1-12)
            - 'day_of_week': Día de la semana (0-6)
        
        params (Dict[str, Any]): Diccionario de parámetros de configuración:
            - 'lag_days' (List[int], opcional): Días de rezago a calcular. 
              Default: [1, 3, 7, 14, 21]
            - 'rolling_windows' (List[int], opcional): Ventanas para estadísticas rodantes.
              Default: [3, 7, 14, 30]
    
    Returns:
        pd.DataFrame: DataFrame original con features temporales adicionales:
            - confirmed_lag_N: Casos confirmados N días atrás
            - deceased_lag_N: Muertes N días atrás
            - confirmed_rolling_mean_N: Media móvil de casos ventana N
            - confirmed_rolling_std_N: Desviación estándar móvil ventana N
            - confirmed_rolling_max_N: Máximo móvil ventana N
            - confirmed_rolling_min_N: Mínimo móvil ventana N
            - confirmed_trend_7d: Pendiente de tendencia en ventana 7 días
            - month_sin, month_cos: Componentes sinusoidales del mes
            - day_sin, day_cos: Componentes sinusoidales del día
            - confirmed_acceleration: Segunda derivada de casos (aceleración)
    
    Raises:
        KeyError: Si faltan columnas requeridas ('date', 'new_confirmed', etc.)
        ValueError: Si los parámetros contienen valores inválidos
    
    Example:
        >>> params = {
        ...     "lag_days": [7, 14],
        ...     "rolling_windows": [7, 30]
        ... }
        >>> df_with_features = create_temporal_features(df_nacional, params)
        >>> print(df_with_features.columns)
        
    Notes:
        - Las variables lag generan valores NaN en las primeras N observaciones
        - Las rolling statistics generan NaN hasta completar la ventana
        - Los componentes sinusoidales capturan estacionalidad sin discontinuidades
        - La tendencia usa regresión lineal local en ventana móvil
        
    See Also:
        create_epidemiological_features: Para features epidemiológicas
        create_regional_features: Para features de análisis regional
    """
    logger.info("Creando features temporales avanzadas")
    
    df_temporal = df.copy()
    
    # Validar columnas requeridas
    required_columns = ['date', 'new_confirmed', 'new_deceased', 'month', 'day_of_week']
    missing_columns = [col for col in required_columns if col not in df_temporal.columns]
    if missing_columns:
        raise KeyError(f"Faltan columnas requeridas: {missing_columns}")
    
    # Variables de lag con logging (MÁS LAGS para mejor predicción)
    lag_days = params.get("lag_days", [1, 3, 7, 14, 21, 28])  # Agregado 28 días
    logger.debug(f"Creando {len(lag_days)} variables lag: {lag_days}")
    
    for lag in lag_days:
        df_temporal[f'confirmed_lag_{lag}'] = df_temporal['new_confirmed'].shift(lag)
        df_temporal[f'deceased_lag_{lag}'] = df_temporal['new_deceased'].shift(lag)
    
    # Rolling statistics con manejo robusto
    windows = params.get("rolling_windows", [3, 7, 14, 30])
    logger.debug(f"Creando rolling statistics para ventanas: {windows}")
    
    for window in windows:
        # Medias móviles con mínimo de períodos
        df_temporal[f'confirmed_rolling_mean_{window}'] = df_temporal['new_confirmed'].rolling(
            window, min_periods=max(1, window // 2)
        ).mean()
        
        df_temporal[f'confirmed_rolling_std_{window}'] = df_temporal['new_confirmed'].rolling(
            window, min_periods=max(1, window // 2)
        ).std()
        
        df_temporal[f'confirmed_rolling_max_{window}'] = df_temporal['new_confirmed'].rolling(
            window, min_periods=max(1, window // 2)
        ).max()
        
        df_temporal[f'confirmed_rolling_min_{window}'] = df_temporal['new_confirmed'].rolling(
            window, min_periods=max(1, window // 2)
        ).min()
    
    # Variables de tendencia usando regresión lineal local
    logger.debug("Calculando tendencias temporales")
    df_temporal['confirmed_trend_7d'] = df_temporal['new_confirmed'].rolling(7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 and not x.isna().any() else np.nan,
        raw=False
    )
    
    # Variables estacionales (transformación sinusoidal para continuidad)
    logger.debug("Creando componentes estacionales")
    df_temporal['day_of_year'] = df_temporal['date'].dt.dayofyear
    df_temporal['week_of_year'] = df_temporal['date'].dt.isocalendar().week
    
    # Transformaciones sin/cos para capturar estacionalidad sin discontinuidades
    df_temporal['month_sin'] = np.sin(2 * np.pi * df_temporal['month'] / 12)
    df_temporal['month_cos'] = np.cos(2 * np.pi * df_temporal['month'] / 12)
    df_temporal['day_sin'] = np.sin(2 * np.pi * df_temporal['day_of_week'] / 7)
    df_temporal['day_cos'] = np.cos(2 * np.pi * df_temporal['day_of_week'] / 7)
    
    # Variables de aceleración (segunda derivada para detectar cambios en la tasa de cambio)
    logger.debug("Calculando aceleración de casos")
    df_temporal['confirmed_acceleration'] = df_temporal['new_confirmed'].diff().diff()
    
    # Estadísticas finales
    new_features = df_temporal.shape[1] - df.shape[1]
    logger.info(f"Features temporales creadas: {new_features} nuevas columnas")
    logger.info(f"Forma final del dataset: {df_temporal.shape}")
    
    return df_temporal

def create_epidemiological_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea features epidemiológicas específicas para análisis COVID.
    
    Args:
        df: Dataset completo por ubicación con columna 'location_key'
        params: Parámetros de feature engineering que deben incluir:
               - population_data: Dict con poblaciones por ubicación
               - default_population: Población por defecto si no se encuentra
        
    Returns:
        DataFrame con features epidemiológicas calculadas
        
    Raises:
        KeyError: Si faltan columnas requeridas en el DataFrame
        
    Example:
        params = {
            "population_data": {
                "default_population": 19116201,
                "regional_populations": {"CL_RM": 7112808}
            }
        }
    """
    logger.info("Creando features epidemiológicas")
    
    df_epi = df.copy()
    
    # Obtener configuración de población desde parámetros
    population_config = params.get("population_data", {})
    default_population = population_config.get("default_population", 19116201)  # Población Chile
    regional_populations = population_config.get("regional_populations", {})
    
    # Asignar población según ubicación desde parámetros
    def get_population(location_key):
        """Obtiene población para una ubicación desde configuración."""
        return regional_populations.get(location_key, default_population)
    
    df_epi['population_estimated'] = df_epi['location_key'].apply(get_population)
    
    # Tasas epidemiológicas con manejo robusto de divisiones por cero
    df_epi['case_fatality_rate_daily'] = np.where(
        df_epi['new_confirmed'] > 0,
        (df_epi['new_deceased'] / df_epi['new_confirmed']).clip(0, 1),  # Limitar a [0,1]
        0
    )
    
    df_epi['recovery_rate'] = np.where(
        df_epi['cumulative_confirmed'] > 0,
        (df_epi['cumulative_recovered'] / df_epi['cumulative_confirmed']).clip(0, 1),
        0
    )
    
    # Tasas de crecimiento con manejo de valores infinitos
    df_epi['confirmed_growth_rate'] = df_epi.groupby('location_key')['cumulative_confirmed'].pct_change()
    df_epi['confirmed_growth_rate'] = df_epi['confirmed_growth_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    
    df_epi['deceased_growth_rate'] = df_epi.groupby('location_key')['cumulative_deceased'].pct_change()
    df_epi['deceased_growth_rate'] = df_epi['deceased_growth_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Ratios comparativos con validación
    df_epi['new_vs_cumulative_ratio'] = np.where(
        df_epi['cumulative_confirmed'] > 0,
        df_epi['new_confirmed'] / df_epi['cumulative_confirmed'],
        0
    )
    
    # Intensidad epidemiológica usando población real desde parámetros
    df_epi['cases_per_100k'] = np.where(
        df_epi['population_estimated'] > 0,
        (df_epi['cumulative_confirmed'] / df_epi['population_estimated']) * 100000,
        0
    )
    
    df_epi['deaths_per_100k'] = np.where(
        df_epi['population_estimated'] > 0,
        (df_epi['cumulative_deceased'] / df_epi['population_estimated']) * 100000,
        0
    )
    
    # Variables de volatilidad con ventanas configurables
    volatility_window = params.get("volatility_window", 7)
    df_epi['confirmed_volatility_7d'] = df_epi.groupby('location_key')['new_confirmed'].transform(
        lambda x: x.rolling(volatility_window, min_periods=1).std()
    )
    
    # ========================================
    # FEATURES EPIDEMIOLÓGICAS AVANZADAS
    # ========================================
    logger.debug("Creando features epidemiológicas avanzadas")

    # Features de aceleración (segunda derivada)
    df_epi['confirmed_acceleration'] = df_epi.groupby('location_key')['new_confirmed'].transform(
        lambda x: x.diff().diff()
    )

    df_epi['deaths_acceleration'] = df_epi.groupby('location_key')['new_deceased'].transform(
        lambda x: x.diff().diff()
    )

    # Ratios temporales (comparación con promedio 7d)
    rolling_7d_confirmed = df_epi.groupby('location_key')['new_confirmed'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df_epi['cases_vs_7d_avg'] = np.where(
        rolling_7d_confirmed > 0,
        df_epi['new_confirmed'] / rolling_7d_confirmed,
        1.0
    )

    rolling_7d_deaths = df_epi.groupby('location_key')['new_deceased'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    df_epi['deaths_vs_7d_avg'] = np.where(
        rolling_7d_deaths > 0,
        df_epi['new_deceased'] / rolling_7d_deaths,
        1.0
    )

    # Momentum (cambio en la tasa de cambio)
    df_epi['confirmed_momentum'] = df_epi.groupby('location_key')['confirmed_growth_rate'].transform(
        lambda x: x.diff()
    )

    # Limpiar infinitos y NaN de las nuevas features
    advanced_features = [
        'confirmed_acceleration', 'deaths_acceleration', 
        'cases_vs_7d_avg', 'deaths_vs_7d_avg', 'confirmed_momentum'
    ]

    for col in advanced_features:
        df_epi[col] = df_epi[col].replace([np.inf, -np.inf], 0).fillna(0)

    logger.debug(f"Features avanzadas creadas: {advanced_features}")
    
    # Estadísticas de resumen
    logger.info(f"Features epidemiológicas creadas: {df_epi.shape}")
    logger.info(f"Ubicaciones procesadas: {df_epi['location_key'].nunique()}")
    logger.info(f"Rango de población: {df_epi['population_estimated'].min():.0f} - {df_epi['population_estimated'].max():.0f}")
    
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
    Aplica transformaciones de scaling y encoding, integrando datos temporales y epidemiológicos.
    
    Esta función combina features temporales (nivel nacional) con features epidemiológicas
    (nivel regional agregado), aplica transformaciones de escalado y codifica variables
    categóricas para preparar los datos para modelado ML.
    
    Args:
        df_temporal (pd.DataFrame): Features temporales a nivel nacional diario.
            Debe contener 'date' como índice o columna.
            
        df_epi (pd.DataFrame): Features epidemiológicas por ubicación y fecha.
            Debe contener columnas 'date' y 'location_key' para agregación.
            
        params (Dict[str, Any]): Parámetros de transformación:
            - 'apply_standard_scaling' (bool): Aplicar StandardScaler. Default: True
            - 'apply_minmax_scaling' (bool): Aplicar MinMaxScaler. Default: True
            - 'scaling_method' (str): Método de agregación ('mean', 'median', 'sum')
    
    Returns:
        pd.DataFrame: Dataset integrado con transformaciones aplicadas:
            - Features originales temporales y epidemiológicas
            - Features escaladas (sufijo _scaled)
            - Variables categóricas codificadas (sufijo _encoded)
            
    Raises:
        ValueError: Si los DataFrames no tienen columna 'date'
        KeyError: Si faltan columnas requeridas para merge
    
    Notes:
        - El merge se realiza agregando primero df_epi por fecha a nivel nacional
        - StandardScaler se aplica a features numéricas principales
        - MinMaxScaler se aplica específicamente a features cíclicas
        - Variables categóricas se codifican con LabelEncoder
        
    Warning:
        Esta función modifica la granularidad de df_epi de regional a nacional
        mediante agregación. Asegúrate de que esto sea el comportamiento deseado
        para tu análisis.
    
    Example:
        >>> params = {
        ...     "apply_standard_scaling": True,
        ...     "apply_minmax_scaling": True
        ... }
        >>> df_final = apply_transformations(df_temporal, df_epi, params)
    """
    logger.info("Aplicando transformaciones e integrando features temporales y epidemiológicas")
    
    # Validar presencia de columna date
    if 'date' not in df_temporal.columns:
        raise ValueError("df_temporal debe contener columna 'date'")
    if 'date' not in df_epi.columns:
        raise ValueError("df_epi debe contener columna 'date'")
    
    # Dataset base: copiar datos temporales
    df_base = df_temporal.copy()
    logger.debug(f"Dataset base (temporal): {df_base.shape}")
    
    # CORRECCIÓN: Agregar features epidemiológicas correctamente por fecha
    # Identificar columnas numéricas para agregar
    epi_numeric_cols = df_epi.select_dtypes(include=[np.number]).columns
    epi_features = [col for col in epi_numeric_cols if col not in ['year', 'month', 'day_of_week']]
    
    # Método de agregación desde parámetros
    aggregation_method = params.get("scaling_method", "mean")
    logger.info(f"Agregando features epidemiológicas por fecha usando método: {aggregation_method}")
    
    # Agregar por fecha a nivel nacional
    epi_aggregation = {}
    for col in epi_features:
        if col in df_epi.columns:
            if aggregation_method == 'mean':
                epi_aggregation[col] = 'mean'
            elif aggregation_method == 'median':
                epi_aggregation[col] = 'median'
            elif aggregation_method == 'sum':
                epi_aggregation[col] = 'sum'
            else:
                epi_aggregation[col] = 'mean'  # default
    
    epi_national = df_epi.groupby('date').agg(epi_aggregation).reset_index()
    logger.debug(f"Features epidemiológicas agregadas: {epi_national.shape}")
    
    # Merge robusto con manejo de fechas
    df_transformed = df_base.merge(
        epi_national, 
        on='date', 
        how='left',
        suffixes=('', '_epi')
    )
    
    logger.info(f"Dataset integrado: {df_transformed.shape}")
    logger.info(f"Registros después del merge: {len(df_transformed)}")
    
    # Identificar columnas numéricas para scaling (excluir metadata temporal)
    numeric_columns = df_transformed.select_dtypes(include=[np.number]).columns
    scaling_columns = [
        col for col in numeric_columns 
        if col not in ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year']
        and not col.endswith('_sin') 
        and not col.endswith('_cos')
    ]
    
    logger.debug(f"Columnas para StandardScaler: {len(scaling_columns)}")
    
    # Aplicar StandardScaler a features principales
    if params.get("apply_standard_scaling", True) and scaling_columns:
        try:
            scaler = StandardScaler()
            # Rellenar NaN antes de scaling
            df_for_scaling = df_transformed[scaling_columns].fillna(0)
            scaled_features = scaler.fit_transform(df_for_scaling)
            
            # Crear nombres de columnas escaladas
            scaled_columns = [f"{col}_scaled" for col in scaling_columns]
            scaled_df = pd.DataFrame(
                scaled_features, 
                columns=scaled_columns, 
                index=df_transformed.index
            )
            
            df_transformed = pd.concat([df_transformed, scaled_df], axis=1)
            logger.info(f"StandardScaler aplicado a {len(scaling_columns)} columnas")
        except Exception as e:
            logger.warning(f"Error aplicando StandardScaler: {e}")
    
    # Aplicar MinMaxScaler a features cíclicas
    cyclical_features = [
        col for col in df_transformed.columns 
        if any(x in col for x in ['_sin', '_cos'])
    ]
    
    if cyclical_features and params.get("apply_minmax_scaling", True):
        try:
            minmax_scaler = MinMaxScaler()
            df_transformed[cyclical_features] = minmax_scaler.fit_transform(
                df_transformed[cyclical_features].fillna(0)
            )
            logger.info(f"MinMaxScaler aplicado a {len(cyclical_features)} features cíclicas")
        except Exception as e:
            logger.warning(f"Error aplicando MinMaxScaler: {e}")
    
    # Encoding de variables categóricas
    categorical_columns = df_transformed.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col not in ['date']]
    
    for col in categorical_columns:
        try:
            le = LabelEncoder()
            df_transformed[f'{col}_encoded'] = le.fit_transform(
                df_transformed[col].astype(str)
            )
            logger.debug(f"LabelEncoder aplicado a: {col}")
        except Exception as e:
            logger.warning(f"Error encodificando {col}: {e}")
    
    # Estadísticas finales
    logger.info(f"Transformaciones completadas: {df_transformed.shape}")
    logger.info(f"Valores faltantes: {df_transformed.isna().sum().sum()}")
    
    return df_transformed

def identify_ml_targets(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Identifica y crea variables objetivo MEJORADAS para problemas de ML.
    
    MEJORAS vs versión anterior:
    - Targets de cambio relativo en vez de valores absolutos
    - Transformación logarítmica para reducir escala
    - Targets suavizados con rolling means
    
    Args:
        df: Dataset con features transformadas
        params: Parámetros para definición de targets
        
    Returns:
        DataFrame con targets de ML mejorados
    """
    logger.info("Identificando targets MEJORADOS para Machine Learning")
    
    df_targets = df.copy()
    
    # Parámetros
    future_days = params.get("prediction_horizon", 7)
    
    # ========================================
    # TARGETS DE REGRESIÓN MEJORADOS
    # ========================================
    
    # TARGET 1: Cambio relativo porcentual (MÁS PREDECIBLE)
    # En vez de predecir 10,000 casos, predecir +15% de cambio
    current_cases = df_targets['new_confirmed']
    future_cases = df_targets['new_confirmed'].shift(-future_days)
    
    # Evitar división por cero
    df_targets[f'target_change_pct_{future_days}d'] = (
        (future_cases - current_cases) / (current_cases + 1)
    ) * 100
    
    # Limitar cambios extremos a [-200%, +200%]
    df_targets[f'target_change_pct_{future_days}d'] = df_targets[f'target_change_pct_{future_days}d'].clip(-200, 200)
    
    # TARGET 2: Log-transform de casos futuros (ESCALA REDUCIDA)
    # Log transforma valores grandes: log(10000) ≈ 9.2, más fácil de predecir
    df_targets[f'target_confirmed_next_{future_days}d_log'] = np.log1p(future_cases)
    
    # TARGET 3: Cambio en media móvil 7d (SUAVIZADO)
    # Menos ruidoso que casos diarios individuales
    rolling_current = df_targets['new_confirmed'].rolling(7, min_periods=1).mean()
    rolling_future = rolling_current.shift(-future_days)
    df_targets[f'target_rolling_change_{future_days}d'] = rolling_future - rolling_current
    
    # TARGET 4: Tasa de crecimiento futura (YA EXISTENTE, mejorado)
    # Asegurar que no tenga infinitos
    growth_rate = df_targets['confirmed_growth_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    df_targets['target_growth_rate_next_14d'] = growth_rate.shift(-14).clip(-1, 1)
    
    # ========================================
    # TARGETS DE CLASIFICACIÓN (SIN CAMBIOS)
    # ========================================
    
    # TARGET 5: Clasificación - Período de alta transmisión
    confirmed_threshold = df_targets['new_confirmed'].quantile(params.get("high_transmission_quantile", 0.75))
    df_targets['target_high_transmission'] = (df_targets['new_confirmed'] > confirmed_threshold).astype(int)
    
    # TARGET 6: Clasificación - Nivel de riesgo (multiclase)
    q33 = df_targets['new_confirmed'].quantile(0.33)
    q66 = df_targets['new_confirmed'].quantile(0.66)
    
    def assign_risk_level(cases):
        if pd.isna(cases):
            return np.nan
        elif cases <= q33:
            return 0  # Bajo
        elif cases <= q66:
            return 1  # Medio
        else:
            return 2  # Alto
    
    df_targets['target_risk_level'] = df_targets['new_confirmed'].apply(assign_risk_level)
    
    # TARGET 7: Clasificación - Tendencia (subida/bajada)
    # Usar tendencia más robusta
    future_trend = df_targets['confirmed_trend_7d'].shift(-7)  # Tendencia en 7 días
    df_targets['target_trend_direction'] = np.where(
        future_trend > 0, 1, 0
    ).astype(int)
    
    # ========================================
    # LIMPIEZA FINAL
    # ========================================
    
    # Remover infinitos y NaN de todos los targets
    target_cols = [col for col in df_targets.columns if col.startswith('target_')]
    for col in target_cols:
        df_targets[col] = df_targets[col].replace([np.inf, -np.inf], np.nan)
    
    # Información de targets para logging
    regression_targets_new = [
        f'target_change_pct_{future_days}d',
        f'target_confirmed_next_{future_days}d_log', 
        f'target_rolling_change_{future_days}d',
        'target_growth_rate_next_14d'
    ]
    
    classification_targets = [
        'target_high_transmission', 
        'target_risk_level', 
        'target_trend_direction'
    ]
    
    targets_info = {
        'regression_targets': regression_targets_new,
        'classification_targets': classification_targets
    }
    
    logger.info(f"✅ Targets MEJORADOS creados: {targets_info}")
    logger.info(f"   • Cambio % en vez de valores absolutos")
    logger.info(f"   • Log-transform para escala reducida")
    logger.info(f"   • Media móvil para suavizado")
    
    # Estadísticas de los nuevos targets
    for target in regression_targets_new:
        if target in df_targets.columns:
            data = df_targets[target].dropna()
            if len(data) > 0:
                logger.info(f"   • {target}: min={data.min():.2f}, max={data.max():.2f}, media={data.mean():.2f}")
    
    return df_targets

def prepare_ml_datasets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara datasets ML - OPTIMIZADO PARA MEMORIA
    """
    logger.info("Preparando datasets ML (optimizado para memoria)")
    
    merge_keys = params.get('merge_keys', ['location_key', 'date'])
    
    # CRÍTICO: Muestrear ANTES del merge
    max_before_merge = 1000
    
    if len(features) > max_before_merge:
        features = features.sample(n=max_before_merge, random_state=42)
        logger.info(f"✅ Features pre-merge: {max_before_merge}")
    
    if len(targets) > max_before_merge:
        targets = targets.sample(n=max_before_merge, random_state=42)
        logger.info(f"✅ Targets pre-merge: {max_before_merge}")
    
    # Merge con datos ya reducidos
    df_merged = features.merge(targets, on=merge_keys, how='inner')
    logger.info(f"✅ Merge: {df_merged.shape}")
    
    # Si aún es muy grande después del merge, reducir más
    if len(df_merged) > 2000:
        df_merged = df_merged.sample(n=2000, random_state=42)
        logger.info(f"✅ Post-merge sample: {df_merged.shape}")
    
    # Targets
    reg_targets = ['target_change_pct_7d']
    cls_targets = ['target_high_transmission']
    
    # Verificar que existan
    reg_targets = [t for t in reg_targets if t in df_merged.columns]
    cls_targets = [t for t in cls_targets if t in df_merged.columns]
    
    # Para regresión: mantener solo filas con target válido
    df_reg = df_merged.copy()
    if reg_targets:
        df_reg = df_reg[df_reg[reg_targets].notna().any(axis=1)]
    
    # Para clasificación: mantener solo filas con target válido
    df_cls = df_merged.copy()
    if cls_targets:
        df_cls = df_cls[df_cls[cls_targets].notna().any(axis=1)]
    
    logger.info(f"✅ Regresión limpia: {df_reg.shape}")
    logger.info(f"✅ Clasificación limpia: {df_cls.shape}")
    
    # Features
    feature_cols = [c for c in df_reg.columns 
                   if c not in reg_targets + cls_targets + merge_keys]
    
    # Crear datasets finales
    regression_dataset = df_reg[merge_keys + feature_cols + reg_targets].copy()
    classification_dataset = df_cls[merge_keys + feature_cols + cls_targets].copy()
    
    logger.info(f"✅ FINAL Regression: {regression_dataset.shape}")
    logger.info(f"✅ FINAL Classification: {classification_dataset.shape}")
    
    return regression_dataset, classification_dataset