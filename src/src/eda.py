"""
Exploratory Data Analysis (EDA) Module

Generates visualizations and analysis for energy consumption patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EDA:
    """Exploratory Data Analysis for energy consumption"""
    
    def __init__(self, data):
        self.data = data
        sns.set_style('darkgrid')
        plt.rcParams['figure.figsize'] = (14, 8)
        
    def time_series_plot(self, date_col='date', target_col='Usage_kWh'):
        """Plot time series of energy consumption"""
        logger.info("Creating time series plot...")
        
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(self.data[date_col], self.data[target_col], linewidth=1.5, color='#1f77b4')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
        ax.set_title('Energy Consumption Over Time - DAEWOO Steel', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = Path('reports/figures/01_time_series.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {path}")
    
    def distribution_plots(self):
        """Plot distributions of numeric variables"""
        logger.info("Creating distribution plots...")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows*4))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(self.data[col], bins=50, color='#ff7f0e', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Distribution: {col}', fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        for idx in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        path = Path('reports/figures/02_distributions.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {path}")
    
    def correlation_heatmap(self):
        """Plot correlation heatmap"""
        logger.info("Creating correlation heatmap...")
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    cbar_kws={'label': 'Correlation'}, ax=ax, square=True)
        ax.set_title('Correlation Matrix - Energy Consumption Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        path = Path('reports/figures/03_correlation_heatmap.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {path}")
    
    def run_all_eda(self):
        """Run complete EDA"""
        logger.info("\n" + "="*60)
        logger.info("STARTING EXPLORATORY DATA ANALYSIS")
        logger.info("="*60)
        
        self.time_series_plot()
        self.distribution_plots()
        self.correlation_heatmap()
        
        logger.info("\n" + "="*60)
        logger.info("✓ EDA COMPLETE")
        logger.info("="*60)

def run_eda(data):
    """Run EDA pipeline"""
    eda = EDA(data)
    eda.run_all_eda()

if __name__ == "__main__":
    data = pd.read_csv("data/processed/processed_data.csv")
    run_eda(data)
