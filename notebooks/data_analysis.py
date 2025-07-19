"""
Data Analysis and Visualization for Telecom Churn Prediction

This script performs exploratory data analysis (EDA) on the telecom dataset,
generating visualizations to understand the data distribution and relationships
between features and the target variable (churn).
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnull, mean, stddev

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DATA_DIR, PROCESSED_DATA_DIR, LOG_DIR, 
    SparkConfig, LoggingConfig
)

# Set up logging
LoggingConfig.setup_logging()
import logging
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

class DataAnalyzer:
    """
    A class for performing exploratory data analysis on the telecom dataset.
    Handles data loading, summary statistics, and visualization.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the data analyzer.
        
        Args:
            data_path (str, optional): Path to the processed parquet data.
                                     If None, uses the default path from settings.
        """
        self.data_path = data_path or str(PROCESSED_DATA_DIR / "telecom_processed")
        self.spark = self._create_spark_session()
        self.df = None
        
        # Create output directories
        self.plots_dir = os.path.join(DATA_DIR, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def _create_spark_session(self):
        """Create and return a Spark session."""
        return SparkSession.builder \
            .appName("TelecomDataAnalysis") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()
    
    def load_data(self):
        """Load the processed data from parquet files."""
        logger.info(f"Loading data from {self.data_path}")
        self.df = self.spark.read.parquet(self.data_path)
        logger.info(f"Loaded {self.df.count():,} rows with {len(self.df.columns)} columns")
        
        # Cache the DataFrame for faster access
        self.df.cache()
        
        return self.df
    
    def get_basic_stats(self):
        """
        Calculate and display basic statistics for the dataset.
        
        Returns:
            dict: Dictionary containing basic statistics
        """
        if self.df is None:
            self.load_data()
        
        logger.info("Calculating basic statistics...")
        
        # Get total number of records
        total_records = self.df.count()
        
        # Calculate churn rate
        churn_stats = self.df.groupBy("churn").count().collect()
        churn_counts = {row["churn"]: row["count"] for row in churn_stats}
        churn_rate = churn_counts.get(1, 0) / total_records * 100
        
        # Get numeric and categorical columns
        numeric_cols = [f.name for f in self.df.schema.fields 
                       if f.dataType.simpleString() in ['integer', 'double', 'float', 'long']]
        
        # Calculate basic statistics for numeric columns
        stats = {}
        for col_name in numeric_cols:
            stats[col_name] = {
                'mean': self.df.select(mean(col(col_name))).collect()[0][0],
                'std': self.df.select(stddev(col(col_name))).collect()[0][0],
                'min': self.df.select(col(col_name)).rdd.min()[0],
                'max': self.df.select(col(col_name)).rdd.max()[0],
                'null_count': self.df.filter(col(col_name).isNull()).count()
            }
        
        # Create summary dictionary
        summary = {
            'total_records': total_records,
            'churn_rate': churn_rate,
            'churn_distribution': churn_counts,
            'numeric_columns': numeric_cols,
            'statistics': stats
        }
        
        # Log summary
        logger.info(f"Total records: {total_records:,}")
        logger.info(f"Churn rate: {churn_rate:.2f}%")
        logger.info(f"Churn distribution: {churn_counts}")
        
        return summary
    
    def plot_churn_distribution(self, save_plot=True):
        """
        Plot the distribution of the target variable (churn).
        
        Args:
            save_plot (bool): Whether to save the plot to a file
            
        Returns:
            str: Path to the saved plot if save_plot is True, else None
        """
        if self.df is None:
            self.load_data()
        
        logger.info("Plotting churn distribution...")
        
        # Convert to pandas for plotting
        churn_counts = self.df.groupBy("churn").count().toPandas()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        ax = sns.barplot(x="churn", y="count", data=churn_counts)
        
        # Customize plot
        plt.title("Distribution of Churn", fontsize=16)
        plt.xlabel("Churn (0 = No, 1 = Yes)", fontsize=12)
        plt.ylabel("Number of Customers", fontsize=12)
        
        # Add counts on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height()):,}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=12, color='black',
                        xytext=(0, 5),
                        textcoords='offset points')
        
        # Save or show plot
        plot_path = os.path.join(self.plots_dir, "churn_distribution.png")
        
        if save_plot:
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Churn distribution plot saved to {plot_path}")
            return plot_path
        else:
            plt.show()
            return None
    
    def plot_numerical_distributions(self, columns=None, save_plot=True):
        """
        Plot distributions of numerical features.
        
        Args:
            columns (list, optional): List of column names to plot. If None, plots all numeric columns.
            save_plot (bool): Whether to save the plots to files
            
        Returns:
            list: Paths to the saved plots if save_plot is True, else None
        """
        if self.df is None:
            self.load_data()
        
        logger.info("Plotting numerical feature distributions...")
        
        # Get numeric columns if not provided
        if columns is None:
            numeric_cols = [f.name for f in self.df.schema.fields 
                          if f.dataType.simpleString() in ['integer', 'double', 'float', 'long']
                          and f.name != 'churn']
        else:
            numeric_cols = [col for col in columns 
                          if col in self.df.columns 
                          and self.df.schema[col].dataType.simpleString() in ['integer', 'double', 'float', 'long']
                          and col != 'churn']
        
        # Convert to pandas for plotting
        pdf = self.df.select(["churn"] + numeric_cols).toPandas()
        
        saved_plots = []
        
        # Plot each numerical feature
        for col_name in numeric_cols:
            plt.figure(figsize=(12, 6))
            
            # Create subplots for churn vs non-churn distributions
            plt.subplot(1, 2, 1)
            sns.histplot(data=pdf, x=col_name, hue="churn", kde=True, element="step", 
                        stat="density", common_norm=False)
            plt.title(f"{col_name} Distribution by Churn")
            plt.xlabel(col_name)
            plt.ylabel("Density")
            
            # Create boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x="churn", y=col_name, data=pdf)
            plt.title(f"{col_name} by Churn")
            plt.xlabel("Churn (0 = No, 1 = Yes)")
            plt.ylabel(col_name)
            
            # Save or show plot
            plt.tight_layout()
            
            if save_plot:
                plot_path = os.path.join(self.plots_dir, f"{col_name}_distribution.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                saved_plots.append(plot_path)
                plt.close()
            else:
                plt.show()
        
        if save_plot:
            logger.info(f"Saved {len(saved_plots)} numerical distribution plots to {self.plots_dir}")
            return saved_plots
        
        return None
    
    def plot_categorical_distributions(self, columns=None, save_plot=True):
        """
        Plot distributions of categorical features.
        
        Args:
            columns (list, optional): List of column names to plot. If None, plots all categorical columns.
            save_plot (bool): Whether to save the plots to files
            
        Returns:
            list: Paths to the saved plots if save_plot is True, else None
        """
        if self.df is None:
            self.load_data()
        
        logger.info("Plotting categorical feature distributions...")
        
        # Get categorical columns if not provided
        if columns is None:
            categorical_cols = [f.name for f in self.df.schema.fields 
                              if f.dataType.simpleString() in ['string', 'boolean']
                              and f.name != 'churn']
        else:
            categorical_cols = [col for col in columns 
                              if col in self.df.columns 
                              and self.df.schema[col].dataType.simpleString() in ['string', 'boolean']
                              and col != 'churn']
        
        # Convert to pandas for plotting
        pdf = self.df.select(["churn"] + categorical_cols).toPandas()
        
        saved_plots = []
        
        # Plot each categorical feature
        for col_name in categorical_cols:
            plt.figure(figsize=(10, 6))
            
            # Create count plot
            ax = sns.countplot(x=col_name, hue="churn", data=pdf)
            
            # Customize plot
            plt.title(f"{col_name} Distribution by Churn", fontsize=14)
            plt.xlabel(col_name, fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            # Add counts on top of bars
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height()):,}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', fontsize=10, color='black',
                           xytext=(0, 5),
                           textcoords='offset points')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or show plot
            if save_plot:
                plot_path = os.path.join(self.plots_dir, f"{col_name}_distribution.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                saved_plots.append(plot_path)
                plt.close()
            else:
                plt.show()
        
        if save_plot:
            logger.info(f"Saved {len(saved_plots)} categorical distribution plots to {self.plots_dir}")
            return saved_plots
        
        return None
    
    def plot_correlation_matrix(self, columns=None, save_plot=True):
        """
        Plot a correlation matrix of numerical features.
        
        Args:
            columns (list, optional): List of column names to include in the correlation matrix.
                                    If None, uses all numeric columns.
            save_plot (bool): Whether to save the plot to a file
            
        Returns:
            str: Path to the saved plot if save_plot is True, else None
        """
        if self.df is None:
            self.load_data()
        
        logger.info("Plotting correlation matrix...")
        
        # Get numeric columns if not provided
        if columns is None:
            numeric_cols = [f.name for f in self.df.schema.fields 
                          if f.dataType.simpleString() in ['integer', 'double', 'float', 'long']]
        else:
            numeric_cols = [col for col in columns 
                          if col in self.df.columns 
                          and self.df.schema[col].dataType.simpleString() in ['integer', 'double', 'float', 'long']]
        
        # Convert to pandas for correlation calculation
        pdf = self.df.select(numeric_cols).toPandas()
        
        # Calculate correlation matrix
        corr = pdf.corr()
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .8}
        )
        
        # Customize plot
        plt.title("Correlation Matrix of Numerical Features", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show plot
        plot_path = os.path.join(self.plots_dir, "correlation_matrix.png")
        
        if save_plot:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Correlation matrix plot saved to {plot_path}")
            return plot_path
        else:
            plt.show()
            return None
    
    def generate_report(self):
        """
        Generate a comprehensive EDA report with all visualizations.
        
        Returns:
            dict: Dictionary containing paths to all generated plots and summary statistics
        """
        logger.info("Generating EDA report...")
        
        # Get basic statistics
        stats = self.get_basic_stats()
        
        # Generate all plots
        churn_plot = self.plot_churn_distribution(save_plot=True)
        num_plots = self.plot_numerical_distributions(save_plot=True)
        cat_plots = self.plot_categorical_distributions(save_plot=True)
        corr_plot = self.plot_correlation_matrix(save_plot=True)
        
        # Create report dictionary
        report = {
            'summary_statistics': stats,
            'plots': {
                'churn_distribution': churn_plot,
                'numerical_distributions': num_plots,
                'categorical_distributions': cat_plots,
                'correlation_matrix': corr_plot
            },
            'data_info': {
                'total_records': stats['total_records'],
                'churn_rate': stats['churn_rate'],
                'numerical_columns': stats['numeric_columns'],
                'categorical_columns': [f.name for f in self.df.schema.fields 
                                      if f.dataType.simpleString() in ['string', 'boolean']
                                      and f.name != 'churn']
            }
        }
        
        logger.info("EDA report generation complete")
        return report

def main():
    """Main function to run the data analysis."""
    # Initialize the data analyzer
    analyzer = DataAnalyzer()
    
    try:
        # Generate the full EDA report
        report = analyzer.generate_report()
        
        # Print summary information
        print("\n" + "="*50)
        print("TELECOM CHURN PREDICTION - EXPLORATORY DATA ANALYSIS")
        print("="*50)
        print(f"\nTotal records: {report['data_info']['total_records']:,}")
        print(f"Churn rate: {report['summary_statistics']['churn_rate']:.2f}%")
        print(f"\nNumerical features: {', '.join(report['data_info']['numerical_columns'])}")
        print(f"\nCategorical features: {', '.join(report['data_info']['categorical_columns'])}")
        print("\n" + "="*50)
        print(f"\nEDA report generated successfully!")
        print(f"Visualizations saved to: {os.path.join(DATA_DIR, 'plots')}")
        
    except Exception as e:
        logger.error(f"Error during data analysis: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
