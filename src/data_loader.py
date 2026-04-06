"""
Data Loading and Preprocessing Module
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_PROCESSED_PATH, DATE_COL, MISSING_VALUE_METHOD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess Steel industry energy data"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.original_data = None

    def load_data(self):
        logger.info(f"Loading data from {self.file_path}...")
        try:
            self.data = pd.read_csv(self.file_path)
            self.original_data = self.data.copy()
            logger.info(f"✓ Data loaded successfully. Shape: {self.data.shape}")
            logger.info(f"  Columns: {list(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            logger.error(f"✗ File not found: {self.file_path}")
            return None
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}")
            return None

    def inspect_data(self):
        logger.info("\n" + "="*60)
        logger.info("DATA INSPECTION REPORT")
        logger.info("="*60)
        logger.info(f"\n📊 SHAPE: {self.data.shape}")
        logger.info(f"\n📋 COLUMN TYPES:")
        for col, dtype in self.data.dtypes.items():
            logger.info(f"   {col}: {dtype}")
        logger.info(f"\n❌ MISSING VALUES:")
        missing = self.data.isnull().sum()
        if missing.sum() == 0:
            logger.info("   None - All data is complete!")
        else:
            for col, count in missing[missing > 0].items():
                pct = (count / len(self.data)) * 100
                logger.info(f"   {col}: {count} ({pct:.2f}%)")

    def convert_date_column(self, date_col=DATE_COL):
        if date_col not in self.data.columns:
            logger.warning(f"⚠️  Column '{date_col}' not found")
            return
        try:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            logger.info(f"✓ Converted '{date_col}' to datetime")
            logger.info(f"  Date range: {self.data[date_col].min()} to {self.data[date_col].max()}")
        except Exception as e:
            logger.error(f"✗ Error converting date: {e}")

    def handle_missing_values(self, method=MISSING_VALUE_METHOD):
        logger.info(f"\n🔧 Handling missing values using '{method}'...")
        initial_missing = self.data.isnull().sum().sum()
        if method == 'forward_fill':
            self.data = self.data.ffill()
        elif method == 'backward_fill':
            self.data = self.data.bfill()
        elif method == 'mean':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                self.data[col] = self.data[col].fillna(self.data[col].mean())
        final_missing = self.data.isnull().sum().sum()
        logger.info(f"✓ Missing values: {initial_missing} → {final_missing}")

    def remove_duplicates(self):
        initial_shape = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        removed = initial_shape - self.data.shape[0]
        if removed > 0:
            logger.info(f"✓ Removed {removed} duplicate rows")
        else:
            logger.info("✓ No duplicates found")

    def save_processed_data(self, filename='processed_data.csv'):
        output_path = Path(DATA_PROCESSED_PATH) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(output_path, index=False)
        logger.info(f"✓ Processed data saved to {output_path}")
        return output_path

    def get_processed_data(self):
        return self.data


def load_and_preprocess_data(file_path='data/raw/Steel_industry_data.csv'):
    """Complete data loading and preprocessing pipeline"""
    logger.info("\n" + "="*60)
    logger.info("STARTING DATA LOADING & PREPROCESSING")
    logger.info("="*60)

    loader = DataLoader(file_path)
    if loader.load_data() is None:
        return None

    loader.inspect_data()
    loader.convert_date_column()
    loader.handle_missing_values()
    loader.remove_duplicates()
    loader.save_processed_data()

    logger.info("\n" + "="*60)
    logger.info("✓ DATA LOADING & PREPROCESSING COMPLETE")
    logger.info("="*60)

    return loader.get_processed_data()


if __name__ == "__main__":
    data = load_and_preprocess_data()
