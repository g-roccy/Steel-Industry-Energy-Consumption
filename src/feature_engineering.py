import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats


def lag_features(df, lag=1):
    """Creates lag features for the DataFrame"""
    return df.shift(lag)


def rolling_features(df, window=3, func='mean'):  
    """Creates rolling features based on specified function"""  
    if func == 'mean':
        return df.rolling(window=window).mean()
    elif func == 'sum':
        return df.rolling(window=window).sum()
    elif func == 'min':
        return df.rolling(window=window).min()
    elif func == 'max':
        return df.rolling(window=window).max()
    else:
        raise ValueError('Function not recognized.')


def temporal_features(df, date_col):  
    """Extracts temporal features from a date column"""  
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    return df


def interaction_features(df, features):
    """Creates interaction features from specified columns"""  
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            df[f'{features[i]}_x_{features[j]}'] = df[features[i]] * df[features[j]]
    return df


def normalize(df):
    """Normalizes the DataFrame using StandardScaler"""  
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df


def outlier_handling(df, threshold=3):
    """Removes outliers from the DataFrame based on Z-score"""  
    df_no_outliers = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number]))) < threshold).all(axis=1)]
    return df_no_outliers


def categorical_encoding(df, categorical_cols):
    """Encodes categorical variables using OneHotEncoder"""  
    encoder = OneHotEncoder(drop='first', sparse=False)
    transformed = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(transformed, columns=encoder.get_feature_names_out(categorical_cols))
    return encoded_df


def run_feature_engineering(data, date_col='date', target_col='Usage_kWh'):
    """Run complete feature engineering pipeline"""
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Starting feature engineering...")
    df = data.copy()

    # Temporal features (from date column)
    if date_col in df.columns:
        df = temporal_features(df, date_col)
        logger.info("✓ Temporal features added")

    # Lag & rolling features on target
    if target_col in df.columns:
        df[f'{target_col}_lag1'] = lag_features(df[target_col], lag=1)
        df[f'{target_col}_rolling_mean3'] = rolling_features(df[target_col], window=3, func='mean')
        df[f'{target_col}_rolling_max3'] = rolling_features(df[target_col], window=3, func='max')
        logger.info("✓ Lag and rolling features added")

    # Drop NaN introduced by lag/rolling before outlier detection
    df = df.dropna()
    logger.info(f"✓ NaN rows dropped. Shape: {df.shape}")

    # Outlier handling: IQR on target column only (Z-score unsuitable for non-normal industrial data)
    if target_col in df.columns:
        before = len(df)
        q1 = df[target_col].quantile(0.25)
        q3 = df[target_col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[target_col] >= q1 - 3 * iqr) & (df[target_col] <= q3 + 3 * iqr)]
        logger.info(f"✓ Outliers removed: {before - len(df)} rows dropped. Shape: {df.shape}")

    logger.info(f"✓ Feature engineering complete. Final shape: {df.shape}")
    return df

