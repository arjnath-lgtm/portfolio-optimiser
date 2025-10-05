"""
Fundamental Data Analyzer - Updated for Your CSV Format
Integrates fundamental data with portfolio weights to calculate weighted averages
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import os

class FundamentalAnalyzer:
    """
    Analyzes fundamental data with portfolio weights
    Updated for your specific CSV format
    """
    
    def __init__(self, csv_path='datadump.csv'):
        self.csv_path = csv_path
        self.fundamental_data = None
        self.load_fundamental_data()
    
    def load_fundamental_data(self):
        """Load fundamental data from CSV"""
        try:
            if os.path.exists(self.csv_path):
                self.fundamental_data = pd.read_csv(self.csv_path)
                print(f"SUCCESS: Loaded fundamental data: {len(self.fundamental_data)} stocks")
                print(f"Columns: {list(self.fundamental_data.columns)}")
            else:
                print(f"ERROR: Fundamental data file not found: {self.csv_path}")
                self.fundamental_data = None
        except Exception as e:
            print(f"ERROR: Error loading fundamental data: {e}")
            self.fundamental_data = None
    
    def get_fundamental_metrics(self, tickers, weights):
        """
        Get fundamental metrics for portfolio stocks
        
        Args:
            tickers: List of stock tickers (e.g., ['RELIANCE.NS', 'TCS.NS'])
            weights: List of portfolio weights (same order as tickers)
            
        Returns:
            DataFrame with fundamental data and weighted averages
        """
        if self.fundamental_data is None:
            return None
        
        # Convert tickers from RELIANCE.NS format to RELIANCE format for matching
        clean_tickers = [ticker.replace('.NS', '') for ticker in tickers]
        
        # Create portfolio DataFrame
        portfolio_df = pd.DataFrame({
            'ticker': tickers,
            'clean_ticker': clean_tickers,
            'weight': weights
        })
        
        # Merge with fundamental data using "NSE Code" column
        nse_code_column = "NSE Code"
        
        if nse_code_column not in self.fundamental_data.columns:
            print("ERROR: Could not find 'NSE Code' column in fundamental data")
            print(f"Available columns: {list(self.fundamental_data.columns)}")
            return None
        
        # Merge data using clean tickers
        merged_df = portfolio_df.merge(
            self.fundamental_data, 
            left_on='clean_ticker', 
            right_on=nse_code_column, 
            how='left'
        )
        
        return merged_df
    
    def calculate_weighted_averages(self, portfolio_df):
        """
        Calculate weighted averages of fundamental metrics
        
        Args:
            portfolio_df: DataFrame with fundamental data and weights
            
        Returns:
            Dictionary with weighted averages
        """
        if portfolio_df is None or portfolio_df.empty:
            return None
        
        # Get numeric columns (fundamental metrics)
        numeric_cols = portfolio_df.select_dtypes(include=[np.number]).columns
        # Exclude weight column
        numeric_cols = [col for col in numeric_cols if col != 'weight']
        
        weighted_averages = {}
        
        for col in numeric_cols:
            # Calculate weighted average
            weighted_avg = np.average(
                portfolio_df[col].fillna(0), 
                weights=portfolio_df['weight']
            )
            weighted_averages[col] = weighted_avg
        
        return weighted_averages
    
    def get_key_metrics(self, weighted_averages):
        """
        Extract key fundamental metrics for display
        
        Args:
            weighted_averages: Dictionary with all weighted averages
            
        Returns:
            Dictionary with key metrics
        """
        key_metrics = {}
        
        # Map your column names to display names
        metric_mapping = {
            'Price to Earning': 'PE_Ratio',
            'Price to book value': 'PB_Ratio', 
            'Average return on equity 5Years': 'ROE_5Y',
            'Average return on equity 3Years': 'ROE_3Y',
            'Return on assets 3years': 'ROA_3Y',
            'ROCE3yr avg': 'ROCE_3Y',
            'Price to Sales': 'PS_Ratio',
            'Price to Cash Flow': 'PCF_Ratio',
            'Market Capitalization': 'Market_Cap',
            'Return over 1year': 'Return_1Y',
            'Return over 3years': 'Return_3Y',
            'Return over 5years': 'Return_5Y',
            'Sales growth 3Years': 'Sales_Growth_3Y',
            'Sales growth 5Years': 'Sales_Growth_5Y',
            'Profit growth 3Years': 'Profit_Growth_3Y',
            'Profit growth 5Years': 'Profit_Growth_5Y',
            'EPS growth 3Years': 'EPS_Growth_3Y',
            'EPS growth 5Years': 'EPS_Growth_5Y',
            'Debt': 'Debt',
            'Working capital': 'Working_Capital',
            'Free cash flow last year': 'FCF_Last_Year',
            'Altman Z Score': 'Altman_Z_Score'
        }
        
        for original_name, display_name in metric_mapping.items():
            if original_name in weighted_averages:
                key_metrics[display_name] = weighted_averages[original_name]
        
        return key_metrics
    
    def display_fundamental_analysis(self, tickers, weights):
        """
        Display fundamental analysis for portfolio
        
        Args:
            tickers: List of stock tickers
            weights: List of portfolio weights
        """
        st.subheader("📊 Fundamental Analysis")
        
        # Get fundamental data
        portfolio_df = self.get_fundamental_metrics(tickers, weights)
        
        if portfolio_df is None:
            st.error("❌ Could not load fundamental data. Please check your datadump.csv file.")
            return
        
        # Show portfolio stocks with fundamental data
        st.write("**Portfolio Stocks with Fundamental Data:**")
        
        # Display key columns for better readability
        key_columns = [
            'ticker', 'weight', 'Name', 'Industry', 'Current Price',
            'Price to Earning', 'Price to book value', 'Average return on equity 5Years',
            'Market Capitalization', 'Return over 1year', 'Return over 3years'
        ]
        
        # Filter to available columns
        available_columns = [col for col in key_columns if col in portfolio_df.columns]
        
        st.dataframe(
            portfolio_df[available_columns].fillna('N/A'),
            use_container_width=True
        )
        
        # Calculate weighted averages
        weighted_averages = self.calculate_weighted_averages(portfolio_df)
        
        if weighted_averages:
            st.write("**Weighted Portfolio Averages:**")
            
            # Get key metrics for display
            key_metrics = self.get_key_metrics(weighted_averages)
            
            # Create metrics display in columns
            col1, col2, col3 = st.columns(3)
            
            metrics_list = list(key_metrics.items())
            
            with col1:
                for i, (metric, value) in enumerate(metrics_list):
                    if i % 3 == 0:
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{value:.2f}" if not pd.isna(value) else "N/A"
                        )
            
            with col2:
                for i, (metric, value) in enumerate(metrics_list):
                    if i % 3 == 1:
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{value:.2f}" if not pd.isna(value) else "N/A"
                        )
            
            with col3:
                for i, (metric, value) in enumerate(metrics_list):
                    if i % 3 == 2:
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{value:.2f}" if not pd.isna(value) else "N/A"
                        )
            
            # Create downloadable summary
            summary_df = pd.DataFrame([
                {'Metric': metric, 'Weighted Average': f"{value:.2f}" if not pd.isna(value) else "N/A"}
                for metric, value in key_metrics.items()
            ])
            
            st.download_button(
                label="📥 Download Fundamental Analysis",
                data=summary_df.to_csv(index=False),
                file_name="portfolio_fundamental_analysis.csv",
                mime="text/csv"
            )
            
            # Show additional metrics in expandable section
            with st.expander("📈 Additional Metrics"):
                additional_metrics = {k: v for k, v in weighted_averages.items() 
                                    if k not in key_metrics}
                
                if additional_metrics:
                    additional_df = pd.DataFrame([
                        {'Metric': metric, 'Weighted Average': f"{value:.2f}" if not pd.isna(value) else "N/A"}
                        for metric, value in additional_metrics.items()
                    ])
                    st.dataframe(additional_df, use_container_width=True)
        else:
            st.warning("⚠️ Could not calculate weighted averages")
    
    def get_portfolio_summary(self, tickers, weights):
        """
        Get a summary of portfolio fundamental metrics
        
        Args:
            tickers: List of stock tickers
            weights: List of portfolio weights
            
        Returns:
            Dictionary with portfolio summary
        """
        portfolio_df = self.get_fundamental_metrics(tickers, weights)
        
        if portfolio_df is None:
            return None
        
        weighted_averages = self.calculate_weighted_averages(portfolio_df)
        
        return {
            'portfolio_stocks': len(tickers),
            'stocks_with_data': len(portfolio_df.dropna(subset=['weight'])),
            'weighted_averages': weighted_averages,
            'data_coverage': len(portfolio_df.dropna(subset=['weight'])) / len(tickers) * 100
        }

def integrate_fundamental_analysis():
    """
    Integration function to add fundamental analysis to portfolio optimizer
    """
    
    # Check if fundamental data exists
    csv_paths = [
        '03_Database_Files/datadump.csv',  # Your organized location
        'datadump.csv',
        'data/datadump.csv',
        'fundamental_data/datadump.csv'
    ]
    
    csv_path = None
    for path in csv_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        st.warning("⚠️ Fundamental data file (datadump.csv) not found. Please place it in the project root.")
        return None
    
    return FundamentalAnalyzer(csv_path)

# Example usage function
def example_usage():
    """Example of how to use the fundamental analyzer"""
    
    # Sample portfolio
    tickers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    weights = [0.25, 0.20, 0.20, 0.20, 0.15]
    
    # Initialize analyzer
    analyzer = FundamentalAnalyzer('datadump.csv')
    
    # Get fundamental data
    portfolio_df = analyzer.get_fundamental_metrics(tickers, weights)
    print("Portfolio with fundamental data:")
    print(portfolio_df)
    
    # Calculate weighted averages
    weighted_averages = analyzer.calculate_weighted_averages(portfolio_df)
    print("\nWeighted averages:")
    for metric, value in weighted_averages.items():
        print(f"{metric}: {value:.2f}")
    
    # Get portfolio summary
    summary = analyzer.get_portfolio_summary(tickers, weights)
    print(f"\nPortfolio summary: {summary}")

if __name__ == "__main__":
    example_usage()
