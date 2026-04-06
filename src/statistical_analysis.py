"""
Statistical Analysis Module

Provides statistical tests and analysis for energy consumption data
"""

import pandas as pd
import numpy as np
import logging
import sys
from scipy.stats import shapiro, anderson
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TARGET_COL, PLOT_STYLE, FIGURES_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Statistical analysis of energy consumption"""
    
    def __init__(self, data):
        self.data = data
        self.results = {}
        sns.set_style(PLOT_STYLE)
        
    def descriptive_statistics(self):
        """Calculate descriptive statistics"""
        logger.info("\n" + "="*60)
        logger.info("DESCRIPTIVE STATISTICS")
        logger.info("="*60)
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        stats_df = numeric_data.describe().T
        
        stats_df['skewness'] = numeric_data.skew()
        stats_df['kurtosis'] = numeric_data.kurtosis()
        
        logger.info(f"\n{stats_df.to_string()}")
        self.results['descriptive_stats'] = stats_df
        return stats_df
    
    def normality_tests(self, columns=None):
        """Test for normality"""
        logger.info("\n" + "="*60)
        logger.info("NORMALITY TESTS")
        logger.info("="*60)
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        normality_results = {}
        
        for col in columns:
            logger.info(f"\n📊 {col}:")
            
            shapiro_stat, shapiro_p = shapiro(self.data[col].dropna())
            shapiro_normal = "Normal" if shapiro_p > 0.05 else "Not Normal"
            logger.info(f"   Shapiro-Wilk: p-value = {shapiro_p:.6f} → {shapiro_normal}")
            
            anderson_result = anderson(self.data[col].dropna())
            logger.info(f"   Anderson-Darling: statistic = {anderson_result.statistic:.6f}")
            
            normality_results[col] = {
                'shapiro_p': shapiro_p,
                'anderson_stat': anderson_result.statistic
            }
        
        self.results['normality_tests'] = normality_results
        return normality_results
    
    def correlation_analysis(self):
        """Analyze correlations"""
        logger.info("\n" + "="*60)
        logger.info("CORRELATION ANALYSIS")
        logger.info("="*60)
        
        numeric_data = self.data.select_dtypes(include=[np.number])
        pearson_corr = numeric_data.corr(method='pearson')
        
        if TARGET_COL in pearson_corr.columns:
            target_corr = pearson_corr[TARGET_COL].sort_values(ascending=False)
            logger.info(f"\n🎯 Correlation with Usage_kWh:")
            for var, corr in target_corr.items():
                logger.info(f"   {var}: {corr:.4f}")
        
        self.results['correlation'] = pearson_corr
        return pearson_corr
    
    def energy_efficiency_metrics(self):
        """Calculate energy efficiency metrics"""
        logger.info("\n" + "="*60)
        logger.info("ENERGY EFFICIENCY METRICS")
        logger.info("="*60)
        
        if TARGET_COL not in self.data.columns:
            logger.warning("Usage_kWh column not found")
            return None

        usage = self.data[TARGET_COL]
        
        peak_load = usage.max()
        base_load = usage.quantile(0.25)
        average_load = usage.mean()
        load_factor = average_load / peak_load if peak_load > 0 else 0
        
        logger.info(f"\n📈 LOAD ANALYSIS:")
        logger.info(f"   Peak Load: {peak_load:.2f} kWh")
        logger.info(f"   Base Load: {base_load:.2f} kWh")
        logger.info(f"   Average Load: {average_load:.2f} kWh")
        logger.info(f"   Load Factor: {load_factor:.4f} (ideal: >0.8)")
        
        efficiency_metrics = {
            'peak_load': peak_load,
            'base_load': base_load,
            'average_load': average_load,
            'load_factor': load_factor
        }
        
        self.results['efficiency_metrics'] = efficiency_metrics
        return efficiency_metrics
    
    def run_all_analysis(self):
        """Run complete statistical analysis"""
        logger.info("\n" + "#"*60)
        logger.info("# RUNNING COMPLETE STATISTICAL ANALYSIS")
        logger.info("#"*60)
        
        self.descriptive_statistics()
        self.normality_tests()
        self.correlation_analysis()
        self.energy_efficiency_metrics()
        
        logger.info("\n" + "#"*60)
        logger.info("# ✓ STATISTICAL ANALYSIS COMPLETE")
        logger.info("#"*60)
        
        return self.results

def run_statistical_analysis(data):
    """Run statistical analysis pipeline"""
    analyzer = StatisticalAnalyzer(data)
    return analyzer.run_all_analysis()

if __name__ == "__main__":
    data = pd.read_csv("data/processed/processed_data.csv")
    run_statistical_analysis(data)
