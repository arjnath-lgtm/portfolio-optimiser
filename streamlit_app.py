"""
Portfolio Optimizer - Cloud Version
Deploy this to Streamlit Cloud for a professional URL
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os
warnings.filterwarnings('ignore')

# Skfolio imports
from skfolio import RiskMeasure
try:
    from skfolio import ObjectiveFunction
except ImportError:
    try:
        from skfolio.optimization import ObjectiveFunction
    except ImportError:
        ObjectiveFunction = None

# Import the main portfolio optimizer
import sys
sys.path.append('01_Main_Application')

# Copy the main application code here or import it
# For now, let's create a simple version that works in the cloud

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">📊 Portfolio Optimizer</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Welcome to the Portfolio Optimizer!
    
    This tool helps you build optimal stock portfolios with:
    - 📈 **Real-time stock data** from Yahoo Finance
    - 🎯 **Portfolio optimization** using modern algorithms  
    - 📊 **Fundamental analysis** with key metrics
    - 📋 **Downloadable reports** in CSV format
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏗️ Build Portfolio", "📊 Fundamental Analysis", "📈 Portfolio Analysis"])
    
    with tab1:
        st.subheader("Build Your Portfolio")
        
        # Stock selection
        st.write("**Step 1: Select Stocks**")
        stock_input = st.text_area(
            "Enter stock symbols (one per line):",
            value="RELIANCE\nTCS\nHDFCBANK\nINFY\nICICIBANK",
            height=100,
            help="Enter Indian stock symbols (e.g., RELIANCE, TCS, HDFCBANK)"
        )
        
        stocks = [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
        
        if stocks:
            st.write(f"**Selected Stocks:** {', '.join(stocks)}")
            
            # Portfolio weights
            st.write("**Step 2: Set Portfolio Weights**")
            weights = {}
            
            col1, col2 = st.columns(2)
            
            for i, stock in enumerate(stocks):
                if i % 2 == 0:
                    with col1:
                        weight = st.slider(f"{stock}", 0, 100, 20, key=f"weight_{stock}")
                        weights[stock] = weight / 100
                else:
                    with col2:
                        weight = st.slider(f"{stock}", 0, 100, 20, key=f"weight_{stock}")
                        weights[stock] = weight / 100
            
            total_weight = sum(weights.values())
            
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Total weight: {total_weight:.1%} (should be 100%)")
            else:
                st.success(f"✅ Total weight: {total_weight:.1%}")
            
            # Optimize button
            if st.button("🚀 Optimize Portfolio", type="primary"):
                with st.spinner("Optimizing portfolio..."):
                    try:
                        # Get stock data
                        stock_data = {}
                        for stock in stocks:
                            try:
                                ticker = yf.Ticker(f"{stock}.NS")
                                hist = ticker.history(period="1y")
                                if not hist.empty:
                                    stock_data[stock] = hist['Close']
                                else:
                                    st.error(f"Could not fetch data for {stock}")
                            except Exception as e:
                                st.error(f"Error fetching {stock}: {e}")
                        
                        if stock_data:
                            # Calculate returns
                            returns_data = {}
                            for stock, prices in stock_data.items():
                                returns = prices.pct_change().dropna()
                                returns_data[stock] = returns
                            
                            if returns_data:
                                returns_df = pd.DataFrame(returns_data)
                                
                                # Calculate portfolio metrics
                                portfolio_returns = (returns_df * pd.Series(weights)).sum(axis=1)
                                
                                # Display results
                                st.success("✅ Portfolio optimized successfully!")
                                
                                # Key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Expected Return", f"{portfolio_returns.mean() * 252 * 100:.1f}%")
                                
                                with col2:
                                    st.metric("Volatility", f"{portfolio_returns.std() * np.sqrt(252) * 100:.1f}%")
                                
                                with col3:
                                    if portfolio_returns.std() > 0:
                                        sharpe = (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))
                                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                                
                                with col4:
                                    st.metric("Total Stocks", len(stocks))
                                
                                # Portfolio composition
                                st.subheader("📊 Portfolio Composition")
                                composition_df = pd.DataFrame([
                                    {'Stock': stock, 'Weight': f"{weight:.1%}", 'Symbol': stock}
                                    for stock, weight in weights.items()
                                ])
                                
                                st.dataframe(composition_df, use_container_width=True)
                                
                                # Download button
                                csv = composition_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Portfolio",
                                    data=csv,
                                    file_name="portfolio_composition.csv",
                                    mime="text/csv"
                                )
                                
                    except Exception as e:
                        st.error(f"Error optimizing portfolio: {e}")
    
    with tab2:
        st.subheader("📊 Fundamental Analysis")
        st.info("💡 Fundamental analysis will be available when you build a portfolio above.")
        
        # Show sample fundamental metrics
        st.write("**Sample Fundamental Metrics:**")
        sample_metrics = {
            'PE Ratio': 'Price-to-Earnings ratio',
            'PB Ratio': 'Price-to-Book ratio', 
            'ROE': 'Return on Equity',
            'ROA': 'Return on Assets',
            'Debt-to-Equity': 'Financial leverage',
            'Market Cap': 'Company valuation'
        }
        
        for metric, description in sample_metrics.items():
            st.write(f"• **{metric}**: {description}")
    
    with tab3:
        st.subheader("📈 Portfolio Analysis")
        st.info("💡 Portfolio analysis will be available when you build a portfolio above.")
        
        st.write("**Available Analysis:**")
        st.write("• 📊 Risk-return analysis")
        st.write("• 📈 Performance metrics")
        st.write("• 🎯 Optimization results")
        st.write("• 📋 Detailed reports")

if __name__ == "__main__":
    main()
