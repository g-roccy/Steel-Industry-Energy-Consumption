# Steel Industry Energy Consumption Analysis

A comprehensive machine learning pipeline for analyzing and predicting energy consumption patterns in the steel manufacturing industry using DAEWOO Steel Co. Ltd operational data.

## Project Overview

This project implements an end-to-end data science workflow for energy consumption prediction in industrial steel production. The pipeline includes data preprocessing, exploratory data analysis, statistical testing, feature engineering, and predictive modeling using ensemble methods.

**Dataset**: DAEWOO Steel Co. Ltd, Korea  
**Time Period**: Full year 2018 (January 1 - December 31)  
**Temporal Resolution**: 15-minute intervals  
**Total Records**: 35,040 observations  
**Features**: 11 variables including reactive power, power factor, CO2 emissions, and operational metadata

## Key Features

- **Automated Data Pipeline**: Complete workflow from raw data to trained models
- **Statistical Analysis**: Normality testing (Shapiro-Wilk), correlation analysis (Pearson/Spearman), and distribution visualization
- **Advanced Feature Engineering**: Temporal features, lag features, rolling window statistics
- **Ensemble Modeling**: XGBoost and Random Forest regressors with hyperparameter optimization
- **Robust Outlier Detection**: IQR-based filtering appropriate for non-normal industrial data
- **Comprehensive Visualization**: Time series plots, distribution analysis, correlation heatmaps

## Project Structure

```
Steel-Industry-Energy-Consumption/
├── config.py                    # Centralized configuration
├── main.py                      # Main pipeline orchestrator
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                     # Original dataset
│   │   └── Steel_industry_data.csv
│   └── processed/               # Processed data outputs
│       ├── processed_data.csv
│       └── features_engineered.csv
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── eda.py                  # Exploratory data analysis
│   ├── statistical_analysis.py # Statistical testing and analysis
│   ├── feature_engineering.py  # Feature creation and selection
│   ├── modeling.py             # Model training and evaluation
│   └── visualization.py        # Plotting utilities
├── models/                      # Trained model artifacts
│   ├── xgboost_model.pkl
│   └── random_forest_model.pkl
└── reports/
    └── figures/                 # Generated visualizations
        ├── 01_time_series.png
        ├── 02_distributions.png
        └── 03_correlation_heatmap.png
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/g-roccy/Steel-Industry-Energy-Consumption.git
cd Steel-Industry-Energy-Consumption

# Install dependencies
pip install -r requirements.txt

# Ensure data file is in place
# Place Steel_industry_data.csv in data/raw/
```

## Usage

### Run Complete Pipeline

```bash
python main.py
```

This executes all five pipeline stages:

1. **Data Loading & Preprocessing**: Load raw data, handle missing values, convert date columns
2. **Exploratory Data Analysis**: Generate time series plots, distributions, correlation analysis
3. **Statistical Analysis**: Normality tests, correlation matrices, statistical summaries
4. **Feature Engineering**: Create temporal, lag, and rolling window features; remove outliers
5. **Predictive Modeling**: Train XGBoost and Random Forest models, evaluate performance

### Configuration

All parameters are centralized in `config.py`:

```python
# Key configurations
TARGET_COL = 'Usage_kWh'              # Target variable
TEST_SIZE = 0.2                       # Train/test split ratio
LAG_WINDOW = 1                        # Lag feature window
ROLLING_WINDOW = 3                    # Rolling statistics window
OUTLIER_THRESHOLD = 3                 # IQR multiplier for outlier detection

# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 200,
    ...
}
```

## Model Performance

### XGBoost Regressor
- **Test R²**: 0.9986
- **Test RMSE**: 1.24 kWh
- **Test MAE**: 0.53 kWh

### Random Forest Regressor
- **Test R²**: 0.9981
- **Test RMSE**: 1.48 kWh
- **Test MAE**: 0.62 kWh

Both models achieve excellent predictive accuracy, with XGBoost slightly outperforming Random Forest. The high R² values indicate that temporal patterns and engineered features capture the underlying energy consumption dynamics effectively.

## Data Description

### Target Variable
- `Usage_kWh`: Energy consumption in kilowatt-hours

### Features
- `Lagging_Current_Reactive.Power_kVarh`: Lagging reactive power
- `Leading_Current_Reactive_Power_kVarh`: Leading reactive power
- `CO2(tCO2)`: Carbon dioxide emissions
- `Lagging_Current_Power_Factor`: Lagging power factor
- `Leading_Current_Power_Factor`: Leading power factor
- `NSM`: Number of seconds from midnight
- `WeekStatus`: Weekday/Weekend indicator
- `Day_of_week`: Day of the week
- `Load_Type`: Type of operational load (Light/Medium/Maximum)

### Engineered Features
- **Temporal**: Year, month, day, day_of_week (extracted from date)
- **Lag Features**: Previous timestep values (lag=1)
- **Rolling Statistics**: 3-period rolling mean and max

## Statistical Findings

- **Non-Normal Distributions**: Shapiro-Wilk tests (p ≈ 0) confirm all variables deviate from normality
- **High Correlation**: Strong positive correlation between `Usage_kWh` and reactive power metrics
- **Temporal Patterns**: Clear daily and weekly seasonality in energy consumption
- **Outlier Handling**: IQR-based filtering (3×IQR range) preserves 100% of data, indicating clean industrial measurements

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
```

## Future Enhancements

- Real-time prediction API endpoint
- LSTM/GRU models for sequence modeling
- Anomaly detection for equipment failure prediction
- Multi-step ahead forecasting
- Integration with industrial IoT sensors
- Dashboard for live monitoring

## License

This project is licensed under the MIT License.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

## Acknowledgments

Dataset provided by DAEWOO Steel Co. Ltd, Korea. This project is for educational and research purposes.
