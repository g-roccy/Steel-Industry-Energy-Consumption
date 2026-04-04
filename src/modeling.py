"""
Predictive Modeling Module

Implements machine learning models for energy consumption prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnergyConsumptionModel:
    """Machine Learning Model for Energy Consumption Prediction"""
    
    def __init__(self, data, target_var='Usage_kWh'):
        self.data = data
        self.target_var = target_var
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, test_size=0.2):
        """Prepare data for modeling"""
        logger.info("Preparing data for modeling...")
        
        self.y = self.data[self.target_var]
        self.X = self.data.drop(columns=[self.target_var, 'date'], errors='ignore')
        
        numeric_cols = self.X.select_dtypes(include=[np.number]).columns
        self.X = self.X[numeric_cols]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"✓ Data prepared. Training: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_xgboost(self):
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=7,
            learning_rate=0.1,
            n_estimators=200,
            random_state=42,
            verbosity=0
        )
        model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=False)
        
        self.models['xgboost'] = model
        logger.info("✓ XGBoost trained successfully")
        return model
    
    def train_random_forest(self):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['random_forest'] = model
        logger.info("✓ Random Forest trained successfully")
        return model
    
    def evaluate_model(self, model_name):
        """Evaluate model performance"""
        logger.info(f"\nEvaluating {model_name}...")
        
        model = self.models[model_name]
        
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"\n✓ {model_name.upper()} RESULTS:")
        logger.info(f"  Train RMSE: {train_rmse:.4f} kWh | Test RMSE: {test_rmse:.4f} kWh")
        logger.info(f"  Train MAE: {train_mae:.4f} kWh | Test MAE: {test_mae:.4f} kWh")
        logger.info(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        
        return metrics
    
    def save_model(self, model_name):
        """Save model to disk"""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return
        
        model_path = Path('models') / f'{model_name}_model.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        logger.info(f"✓ Model saved to {model_path}")
        return model_path

def run_modeling(data):
    """Run complete modeling pipeline"""
    logger.info("\n" + "="*60)
    logger.info("STARTING PREDICTIVE MODELING")
    logger.info("="*60)
    
    model = EnergyConsumptionModel(data)
    model.prepare_data()
    
    model.train_xgboost()
    model.train_random_forest()
    
    model.evaluate_model('xgboost')
    model.evaluate_model('random_forest')
    
    model.save_model('xgboost')
    model.save_model('random_forest')
    
    logger.info("\n" + "="*60)
    logger.info("✓ MODELING COMPLETE")
    logger.info("="*60)
    return model

if __name__ == "__main__":
    data = pd.read_csv("data/processed/processed_data.csv")
    run_modeling(data)
