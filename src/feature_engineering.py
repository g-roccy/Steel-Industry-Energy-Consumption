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


# Example usage (commented out):
# df = pd.DataFrame(...)  # Load your DataFrame
# df = temporal_features(df, 'date_column')
# df['lag_feature'] = lag_features(df['value_column'])
# df['rolling_mean'] = rolling_features(df['value_column'], window=5)
# df = normalize(df)
# df = outlier_handling(df)

