"""
Main Execution Script - Complete Pipeline
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_and_preprocess_data
from eda import run_eda
from statistical_analysis import run_statistical_analysis
from feature_engineering import run_feature_engineering
from modeling import run_modeling

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run complete analysis pipeline"""
    
    logger.info("\n" + "#"*60)
    logger.info("# STEEL INDUSTRY ENERGY CONSUMPTION ANALYSIS")
    logger.info("#"*60)
    
    # Step 1: Data Loading & Preprocessing
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA LOADING & PREPROCESSING")
    logger.info("="*60)
    
    raw_data_path = Path('data/raw/Steel_industry_data.csv')
    
    if not raw_data_path.exists():
        logger.error(f"✗ Raw data file not found: {raw_data_path}")
        logger.error("  Please place Steel_industry_data.csv in data/raw/ directory")
        return
    
    processed_data = load_and_preprocess_data(str(raw_data_path))
    if processed_data is None:
        logger.error("✗ Data loading failed!")
        return
    
    # Step 2: Exploratory Data Analysis
    logger.info("\n" + "="*60)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("="*60)
    
    run_eda(processed_data)
    
    # Step 3: Statistical Analysis
    logger.info("\n" + "="*60)
    logger.info("STEP 3: STATISTICAL ANALYSIS")
    logger.info("="*60)
    
    run_statistical_analysis(processed_data)
    
    # Step 4: Feature Engineering
    logger.info("\n" + "="*60)
    logger.info("STEP 4: FEATURE ENGINEERING")
    logger.info("="*60)
    
    engineered_data = run_feature_engineering(processed_data)
    
    # Step 5: Predictive Modeling
    logger.info("\n" + "="*60)
    logger.info("STEP 5: PREDICTIVE MODELING")
    logger.info("="*60)
    
    run_modeling(engineered_data)
    
    # Final Summary
    logger.info("\n" + "#"*60)
    logger.info("# ✓ COMPLETE ANALYSIS PIPELINE FINISHED!")
    logger.info("#"*60)
    logger.info("\n📊 Output Files Generated:")
    logger.info("  - Processed data: data/processed/processed_data.csv")
    logger.info("  - Engineered features: data/processed/features_engineered.csv")
    logger.info("  - Visualizations: reports/figures/")
    logger.info("  - Trained models: models/")

if __name__ == "__main__":
    main()
