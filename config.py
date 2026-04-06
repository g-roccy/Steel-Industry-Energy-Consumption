# Project Configuration Settings

# Paths
DATA_RAW_PATH = 'data/raw/Steel_industry_data.csv'
DATA_PROCESSED_PATH = 'data/processed'
MODELS_PATH = 'models'
FIGURES_PATH = 'reports/figures'

# Data Configuration
TARGET_COL = 'Usage_kWh'
DATE_COL = 'date'
MISSING_VALUE_METHOD = 'forward_fill'

# Feature Engineering
LAG_WINDOW = 1
ROLLING_WINDOW = 3
OUTLIER_THRESHOLD = 3

# Model Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'random_state': 42,
    'verbosity': 0,
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'random_state': 42,
    'n_jobs': -1,
}

# Visualization
PLOT_STYLE = 'darkgrid'
FIGURE_SIZE = (14, 8)
FIGURE_DPI = 300
