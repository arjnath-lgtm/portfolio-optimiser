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

from skfolio.optimization import (
    MeanRisk,
    EqualWeighted,
    InverseVolatility,
)

# Set page configuration
st.set_page_config(
    page_title="Fixed Portfolio Optimizer", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Top 50 Indian stocks
TOP_50_INDIA = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
    'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
    'BAJFINANCE.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'WIPRO.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATAMOTORS.NS',
    'TATASTEEL.NS', 'M&M.NS', 'ADANIPORTS.NS', 'JSWSTEEL.NS', 'BAJAJFINSV.NS',
    'TECHM.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'CIPLA.NS', 'COALINDIA.NS',
    'DRREDDY.NS', 'BRITANNIA.NS', 'EICHERMOT.NS', 'BPCL.NS', 'GRASIM.NS',
    'SBILIFE.NS', 'SHREECEM.NS', 'DIVISLAB.NS', 'TRENT.NS', 'APOLLOHOSP.NS',
    'ADANIENT.NS', 'HDFCLIFE.NS', 'BAJAJ-AUTO.NS', 'VEDL.NS', 'HEROMOTOCO.NS'
]

@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance"""
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    failed_tickers = []
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"Fetching data for {ticker}... ({i+1}/{len(tickers)})")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            if len(hist) > 20:
                data[ticker] = hist['Close']
            else:
                failed_tickers.append(ticker)
            progress_bar.progress((i + 1) / len(tickers))
        except Exception as e:
            failed_tickers.append(ticker)
    
    progress_bar.empty()
    status_text.empty()
    
    if failed_tickers:
        st.warning(f"⚠️ Could not fetch data for {len(failed_tickers)} stocks")
    
    if len(data) == 0:
        st.error("❌ No data could be fetched. Please check your internet connection.")
        return None
    
    df = pd.DataFrame(data)
    df = df.dropna(axis=1, how='any')
    
    if len(df.columns) < 10:
        st.error("❌ Insufficient stocks with complete data. Need at least 10 stocks.")
        return None
    
    st.success(f"✅ Successfully fetched data for {len(df.columns)} stocks with {len(df)} trading days")
    return df

def calculate_portfolio_returns(test_prices, weights):
    """
    FIXED: Calculate portfolio returns manually from prices and weights.
    DO NOT use portfolio.returns - it returns wrong values!
    """
    # Calculate daily returns for each stock
    daily_returns = test_prices.pct_change().dropna()
    
    # Calculate portfolio returns as weighted sum
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    
    return portfolio_returns

def get_simple_models(min_weight=0.0, max_weight=1.0):
    """Create 4 simple portfolio optimization models"""
    
    if ObjectiveFunction is not None:
        max_sharpe_obj = ObjectiveFunction.MAXIMIZE_RATIO
        min_risk_obj = ObjectiveFunction.MINIMIZE_RISK
    else:
        max_sharpe_obj = "max_sharpe"
        min_risk_obj = "min_risk"
    
    models = {
        "Equal Weighted": EqualWeighted(),
        
        "Inverse Volatility": InverseVolatility(),
        
        "Mean-Variance (Max Sharpe)": MeanRisk(
            risk_measure=RiskMeasure.VARIANCE,
            objective_function=max_sharpe_obj,
            min_weights=min_weight,
            max_weights=max_weight,
        ),
        
        "Mean-Variance (Min Risk)": MeanRisk(
            risk_measure=RiskMeasure.VARIANCE,
            objective_function=min_risk_obj,
            min_weights=min_weight,
            max_weights=max_weight,
        ),
    }
    
    return models

def fit_models(prices, models, train_test_split=0.7):
    """Fit all models and return portfolios with CORRECT returns"""
    split_idx = int(len(prices) * train_test_split)
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]
    
    portfolios = {}
    weights_dict = {}
    returns_dict = {}  # Store correctly calculated returns
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        try:
            status_text.text(f"Optimizing {name}...")
            
            # Fit model
            model.fit(train_prices)
            
            # Get weights
            weights = model.weights_
            weights_dict[name] = dict(zip(prices.columns, weights))
            
            # FIXED: Calculate returns manually instead of using portfolio.returns
            portfolio_returns = calculate_portfolio_returns(test_prices, weights)
            returns_dict[name] = portfolio_returns
            
            # Still predict to get other metrics
            portfolio = model.predict(test_prices)
            portfolios[name] = portfolio
            
            progress_bar.progress((i + 1) / len(models))
        except Exception as e:
            st.warning(f"⚠️ Failed to fit {name}: {str(e)}")
    
    progress_bar.empty()
    status_text.empty()
    
    return portfolios, weights_dict, returns_dict, train_prices, test_prices

def plot_cumulative_returns(returns_dict):
    """Plot cumulative returns using CORRECT returns"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, (name, returns) in enumerate(returns_dict.items()):
        cumulative_returns = (1 + returns).cumprod()
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=(cumulative_returns - 1) * 100,  # Convert to percentage
            mode='lines',
            name=name,
            line=dict(width=2, color=colors[i % len(colors)]),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Cumulative Returns Comparison (Test Period)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

def plot_drawdown(returns_dict):
    """Plot drawdown using CORRECT returns"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Pastel
    
    for i, (name, returns) in enumerate(returns_dict.items()):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name=name,
            fill='tozeroy',
            line=dict(width=1, color=colors[i % len(colors)]),
            hovertemplate='%{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def calculate_portfolio_metrics(returns_dict):
    """Calculate metrics using CORRECT returns"""
    metrics = []
    
    for name, returns in returns_dict.items():
        # Calculate metrics manually from returns
        daily_mean = returns.mean()
        daily_std = returns.std()
        
        # Total return
        total_return = (1 + returns).prod() - 1
        num_days = len(returns)
        
        # Annualize using geometric method
        ann_return = (1 + total_return) ** (252 / num_days) - 1
        ann_volatility = daily_std * np.sqrt(252)
        
        # Sharpe ratio (assuming 7% risk-free rate)
        risk_free_rate = 0.07
        sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0
        
        # Sortino ratio (only penalize downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else ann_volatility
        sortino_ratio = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics.append({
            'Strategy': name,
            'Ann. Return (%)': ann_return * 100,
            'Ann. Volatility (%)': ann_volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Total Return (%)': total_return * 100,
        })
    
    return pd.DataFrame(metrics)

def plot_weights_heatmap(weights_dict, top_n=20):
    """Plot heatmap of weights across methods"""
    all_tickers = list(next(iter(weights_dict.values())).keys())
    avg_weights = {}
    
    for ticker in all_tickers:
        weights = [weights_dict[method].get(ticker, 0) for method in weights_dict.keys()]
        avg_weights[ticker] = np.mean(weights)
    
    top_tickers = sorted(avg_weights.keys(), key=lambda x: avg_weights[x], reverse=True)[:top_n]
    
    matrix = []
    methods = list(weights_dict.keys())
    
    for ticker in top_tickers:
        row = [weights_dict[method].get(ticker, 0) * 100 for method in methods]
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=methods,
        y=[t.replace('.NS', '') for t in top_tickers],
        colorscale='RdYlGn',
        text=[[f'{val:.1f}%' for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='Weight (%)')
    ))
    
    fig.update_layout(
        title=f'Weight Allocation Heatmap (Top {top_n} Stocks)',
        xaxis_title='Optimization Method',
        yaxis_title='Stock',
        height=max(500, top_n * 20),
        xaxis={'side': 'top'}
    )
    
    return fig

def main():
    st.markdown('<div class="main-header">📊 Fixed Portfolio Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Bug Fixed: Correct Returns Calculation</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("📊 Data Parameters")
    years_back = st.sidebar.slider("Years of Historical Data", 1, 5, 3)
    train_test_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.5, 0.9, 0.7, 0.05)
    
    st.sidebar.subheader("🎯 Constraints")
    min_weight = st.sidebar.number_input("Minimum Weight per Stock (%)", 0.0, 10.0, 0.0) / 100
    max_weight = st.sidebar.number_input("Maximum Weight per Stock (%)", 5.0, 100.0, 20.0) / 100
    
    st.sidebar.subheader("🎨 Display Options")
    top_n_display = st.sidebar.slider("Top N Stocks to Display", 10, 30, 20)
    
    # Run analysis
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Fetching data..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back*365)
            
            prices = fetch_stock_data(TOP_50_INDIA, start_date, end_date)
            
            if prices is None or prices.empty:
                st.error("❌ Failed to fetch data")
                return
            
            st.session_state['prices'] = prices
            st.session_state['min_weight'] = min_weight
            st.session_state['max_weight'] = max_weight
            st.session_state['train_test_ratio'] = train_test_ratio
            st.session_state['top_n_display'] = top_n_display
            
            with st.spinner("Optimizing..."):
                models = get_simple_models(min_weight, max_weight)
                portfolios, weights_dict, returns_dict, train_prices, test_prices = fit_models(
                    prices, models, train_test_ratio
                )
                st.session_state['portfolios'] = portfolios
                st.session_state['weights_dict'] = weights_dict
                st.session_state['returns_dict'] = returns_dict
                st.session_state['train_prices'] = train_prices
                st.session_state['test_prices'] = test_prices
            
            st.success("✅ Analysis complete!")
    
    if 'returns_dict' not in st.session_state:
        st.info("👈 Configure parameters and click '🚀 Run Analysis'")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 4 Simple Methods")
            st.markdown("""
            1. **Equal Weighted**
            2. **Inverse Volatility**
            3. **Mean-Variance (Max Sharpe)**
            4. **Mean-Variance (Min Risk)**
            """)
        
        with col2:
            st.markdown("### ✅ Bug Fixed!")
            st.markdown("""
            - ✅ Returns calculated correctly
            - ✅ No more million % returns
            - ✅ Proper annualization
            - ✅ Accurate metrics
            """)
        
        return
    
    # Retrieve from session
    returns_dict = st.session_state['returns_dict']
    weights_dict = st.session_state['weights_dict']
    prices = st.session_state['prices']
    train_prices = st.session_state['train_prices']
    test_prices = st.session_state['test_prices']
    top_n_display = st.session_state['top_n_display']
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance", "📊 Allocations", "📉 Metrics", "💰 Build Portfolio", "💾 Export"
    ])
    
    with tab1:
        st.header("Portfolio Performance")
        
        metrics_df = calculate_portfolio_metrics(returns_dict)
        
        best_return = metrics_df.loc[metrics_df['Ann. Return (%)'].idxmax(), 'Strategy']
        best_sharpe = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax(), 'Strategy']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Return", best_return, 
                     f"{metrics_df[metrics_df['Strategy']==best_return]['Ann. Return (%)'].values[0]:.2f}%")
        with col2:
            st.metric("⭐ Best Sharpe", best_sharpe,
                     f"{metrics_df[metrics_df['Strategy']==best_sharpe]['Sharpe Ratio'].values[0]:.3f}")
        with col3:
            st.metric("📊 Stocks", len(prices.columns))
        
        st.plotly_chart(plot_cumulative_returns(returns_dict), use_container_width=True)
        
        st.subheader("📉 Drawdown Analysis")
        st.plotly_chart(plot_drawdown(returns_dict), use_container_width=True)
    
    with tab2:
        st.header("Weight Allocations")
        
        st.plotly_chart(
            plot_weights_heatmap(weights_dict, top_n=top_n_display),
            use_container_width=True
        )
        
        # Weights table
        all_tickers = list(next(iter(weights_dict.values())).keys())
        weights_data = []
        
        for ticker in all_tickers:
            row = {'Stock': ticker.replace('.NS', '')}
            for method in weights_dict.keys():
                row[method] = weights_dict[method].get(ticker, 0)
            weights_data.append(row)
        
        weights_df = pd.DataFrame(weights_data)
        weights_df = weights_df.set_index('Stock')
        weights_df['Average'] = weights_df.mean(axis=1)
        weights_df = weights_df.sort_values('Average', ascending=False)
        
        display_df = weights_df.head(top_n_display).style.format(
            {col: '{:.2%}' for col in weights_df.columns}
        ).background_gradient(cmap='RdYlGn', axis=1)
        
        st.dataframe(display_df, use_container_width=True)
    
    with tab3:
        st.header("Performance Metrics")
        
        styled_metrics = metrics_df.style.format({
            'Ann. Return (%)': '{:.2f}',
            'Ann. Volatility (%)': '{:.2f}',
            'Sharpe Ratio': '{:.3f}',
            'Sortino Ratio': '{:.3f}',
            'Max Drawdown (%)': '{:.2f}',
            'Total Return (%)': '{:.2f}',
        }).background_gradient(
            subset=['Sharpe Ratio', 'Sortino Ratio'],
            cmap='RdYlGn'
        ).background_gradient(
            subset=['Ann. Volatility (%)', 'Max Drawdown (%)'],
            cmap='RdYlGn_r'
        )
        
        st.dataframe(styled_metrics, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", len(prices.columns))
        with col2:
            st.metric("Training Days", len(train_prices))
        with col3:
            st.metric("Testing Days", len(test_prices))
    
    with tab4:
        st.header("💰 Build Your Portfolio")
        st.markdown("Get exact investment instructions based on the best performing strategy")
        
        # Calculate metrics if not already done
        if 'metrics_df' not in locals():
            metrics_df = calculate_portfolio_metrics(returns_dict)
        
        # Let user choose strategy
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_choice = st.radio(
                "Choose Strategy Based On:",
                ["Best Sharpe Ratio (Risk-Adjusted)", "Best Absolute Return (Highest Gain)"],
                help="Sharpe = Best risk-adjusted return | Absolute = Highest total return"
            )
        
        with col2:
            investment_amount = st.number_input(
                "Total Investment Amount (₹)",
                min_value=10000,
                max_value=100000000,
                value=100000,
                step=10000,
                help="Enter how much money you want to invest"
            )
        
        # Determine which strategy to use
        if "Sharpe" in strategy_choice:
            best_strategy = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax(), 'Strategy']
            best_metric_value = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax(), 'Sharpe Ratio']
            metric_name = "Sharpe Ratio"
        else:
            best_strategy = metrics_df.loc[metrics_df['Ann. Return (%)'].idxmax(), 'Strategy']
            best_metric_value = metrics_df.loc[metrics_df['Ann. Return (%)'].idxmax(), 'Ann. Return (%)']
            metric_name = "Ann. Return"
        
        # Show selected strategy
        st.success(f"✅ Selected Strategy: **{best_strategy}** ({metric_name}: {best_metric_value:.2f})")
        
        # Get weights for selected strategy
        selected_weights = weights_dict[best_strategy]
        
        # Fetch current prices
        with st.spinner("Fetching current stock prices..."):
            current_prices = {}
            tickers_list = list(selected_weights.keys())
            
            for ticker in tickers_list:
                try:
                    stock = yf.Ticker(ticker)
                    current_price = stock.info.get('currentPrice', None)
                    if current_price is None:
                        # Fallback to latest close price
                        hist = stock.history(period='1d')
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                    current_prices[ticker] = current_price
                except:
                    current_prices[ticker] = None
        
        # Build portfolio allocation
        portfolio_allocation = []
        total_allocated = 0
        
        for ticker, weight in selected_weights.items():
            if weight > 0.001:  # Only show stocks with >0.1% allocation
                amount_to_invest = investment_amount * weight
                current_price = current_prices.get(ticker)
                
                if current_price and current_price > 0:
                    num_shares = int(amount_to_invest / current_price)
                    actual_amount = num_shares * current_price
                    total_allocated += actual_amount
                else:
                    num_shares = None
                    actual_amount = amount_to_invest
                
                portfolio_allocation.append({
                    'Stock': ticker.replace('.NS', ''),
                    'Ticker': ticker,
                    'Weight (%)': weight * 100,
                    'Amount (₹)': amount_to_invest,
                    'Current Price (₹)': current_price if current_price else 'N/A',
                    'Shares to Buy': num_shares if num_shares else 'N/A',
                    'Actual Investment (₹)': actual_amount if num_shares else amount_to_invest,
                })
        
        # Create DataFrame
        allocation_df = pd.DataFrame(portfolio_allocation)
        allocation_df = allocation_df.sort_values('Weight (%)', ascending=False)
        
        # Display summary
        st.subheader("📋 Investment Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Budget", f"₹{investment_amount:,.0f}")
        with col2:
            st.metric("Actual Investment", f"₹{total_allocated:,.0f}")
        with col3:
            st.metric("Remaining Cash", f"₹{investment_amount - total_allocated:,.0f}")
        with col4:
            st.metric("Stocks to Buy", len(allocation_df[allocation_df['Shares to Buy'] != 'N/A']))
        
        # Display allocation table
        st.subheader("🎯 Exact Buy List")
        st.markdown("**Copy this list and place your orders:**")
        
        # Format for display
        display_allocation = allocation_df.copy()
        display_allocation['Weight (%)'] = display_allocation['Weight (%)'].apply(lambda x: f"{x:.2f}%")
        display_allocation['Amount (₹)'] = display_allocation['Amount (₹)'].apply(lambda x: f"₹{x:,.0f}")
        display_allocation['Actual Investment (₹)'] = display_allocation['Actual Investment (₹)'].apply(lambda x: f"₹{x:,.0f}")
        
        # Style the dataframe
        st.dataframe(
            display_allocation[['Stock', 'Weight (%)', 'Current Price (₹)', 'Shares to Buy', 'Actual Investment (₹)']],
            use_container_width=True,
            height=400
        )
        
        # Downloadable shopping list
        st.subheader("📥 Download Shopping List")
        
        # Create detailed CSV
        csv_data = allocation_df.copy()
        csv_data['Instructions'] = csv_data.apply(
            lambda row: f"BUY {row['Shares to Buy']} shares at market price" if row['Shares to Buy'] != 'N/A' else 'Price unavailable',
            axis=1
        )
        
        csv_string = csv_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Shopping List (CSV)",
            data=csv_string,
            file_name=f"portfolio_shopping_list_{best_strategy.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Show pie chart
        st.subheader("📊 Portfolio Composition")
        fig = go.Figure(data=[go.Pie(
            labels=allocation_df['Stock'],
            values=allocation_df['Weight (%)'],
            hole=.3,
            textinfo='label+percent',
            textposition='auto',
        )])
        
        fig.update_layout(
            title=f"{best_strategy} - Asset Allocation",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Important notes
        st.info("""
        **📝 Important Notes:**
        - Prices shown are current/latest available prices
        - Actual prices may differ when you place orders
        - Use "Market Order" or "Limit Order" based on your preference
        - Consider brokerage fees and taxes in your budget
        - Rebalance periodically to maintain target allocation
        """)
        
        st.warning("""
        **⚠️ Disclaimer:**
        This is for educational purposes only. Not financial advice.
        Past performance does not guarantee future results.
        Please consult a financial advisor before investing.
        """)
        
        # Add fundamental analysis section
        st.markdown("---")
        st.subheader("📊 Fundamental Analysis")
        
        # Check if fundamental data exists
        csv_paths = [
            '../03_Database_Files/datadump.csv',  # Your organized location (relative to 01_Main_Application)
            '03_Database_Files/datadump.csv',     # Alternative path
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
            st.warning("⚠️ Fundamental data file (datadump.csv) not found.")
            st.info("Please place your datadump.csv file in one of these locations:")
            st.code("""
            ../03_Database_Files/datadump.csv    ← Recommended (your organized structure)
            03_Database_Files/datadump.csv       ← Alternative path
            datadump.csv                         ← Project root
            data/datadump.csv                   ← Data folder
            fundamental_data/datadump.csv        ← Fundamental folder
            """)
            
            # Show current working directory for debugging
            st.info(f"Current working directory: {os.getcwd()}")
            st.info("Make sure your CSV file is in the correct relative path from this directory.")
        else:
            try:
                # Load fundamental data
                fundamental_data = pd.read_csv(csv_path)
                st.success(f"✅ Loaded fundamental data: {len(fundamental_data)} stocks")
                
                # Show sample of fundamental data for debugging
                with st.expander("🔍 Debug: Sample Fundamental Data"):
                    st.write("**Sample NSE Codes from your CSV:**")
                    if 'NSE Code' in fundamental_data.columns:
                        sample_codes = fundamental_data['NSE Code'].head(10).tolist()
                        st.write(sample_codes)
                    else:
                        st.write("Available columns:", list(fundamental_data.columns))
                
                # Get portfolio tickers and weights
                portfolio_tickers = list(selected_weights.keys())
                portfolio_weights = list(selected_weights.values())
                
                # Remove .NS suffix from tickers for matching with datadump
                clean_tickers = [ticker.replace('.NS', '') for ticker in portfolio_tickers]
                
                # Show ticker mapping for debugging
                with st.expander("🔍 Debug: Ticker Mapping"):
                    st.write("**Portfolio Tickers (with .NS):**")
                    st.write(portfolio_tickers)
                    st.write("**Clean Tickers (for matching):**")
                    st.write(clean_tickers)
                
                # Create portfolio DataFrame
                portfolio_df = pd.DataFrame({
                    'ticker': portfolio_tickers,
                    'clean_ticker': clean_tickers,
                    'weight': portfolio_weights
                })
                
                # Merge with fundamental data using clean tickers
                if 'NSE Code' in fundamental_data.columns:
                    merged_df = portfolio_df.merge(
                        fundamental_data, 
                        left_on='clean_ticker', 
                        right_on='NSE Code', 
                        how='left'
                    )
                    
                    # Show portfolio stocks with fundamental data
                    st.write("**Portfolio Stocks with Fundamental Data:**")
                    
                    # Display key columns for better readability
                    key_columns = [
                        'ticker', 'weight', 'Name', 'Industry', 'Current Price',
                        'Price to Earning', 'Price to book value', 'Average return on equity 5Years',
                        'Market Capitalization', 'Return over 1year', 'Return over 3years'
                    ]
                    
                    # Filter to available columns
                    available_columns = [col for col in key_columns if col in merged_df.columns]
                    
                    st.dataframe(
                        merged_df[available_columns].fillna('N/A'),
                        use_container_width=True
                    )
                    
                    # Calculate weighted averages
                    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
                    numeric_cols = [col for col in numeric_cols if col != 'weight']
                    
                    if len(numeric_cols) > 0:
                        weighted_averages = {}
                        
                        for col in numeric_cols:
                            weighted_avg = np.average(
                                merged_df[col].fillna(0), 
                                weights=merged_df['weight']
                            )
                            weighted_averages[col] = weighted_avg
                        
                        # Display key metrics
                        st.write("**📈 Weighted Portfolio Averages:**")
                        
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
                        
                        # Get key metrics for display
                        key_metrics = {}
                        for original_name, display_name in metric_mapping.items():
                            if original_name in weighted_averages:
                                key_metrics[display_name] = weighted_averages[original_name]
                        
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
                        st.warning("⚠️ No numeric fundamental metrics found")
                else:
                    st.error("❌ Could not find 'NSE Code' column in fundamental data")
                    st.info("Please ensure your CSV has a column named 'NSE Code'")
                    
            except Exception as e:
                st.error(f"❌ Error loading fundamental data: {e}")
                st.info("Please check your CSV file format and try again")
    
    with tab5:
        st.header("Export Results")
        
        # Export weights
        weights_csv = weights_df.to_csv()
        st.download_button(
            label="📥 Download Weights (CSV)",
            data=weights_csv,
            file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Export metrics
        metrics_csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=metrics_csv,
            file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
        
        # Export returns
        returns_data = pd.DataFrame(returns_dict)
        returns_csv = returns_data.to_csv()
        st.download_button(
            label="📥 Download Returns (CSV)",
            data=returns_csv,
            file_name=f"portfolio_returns_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.caption(f"Analysis run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Fixed Version")

if __name__ == "__main__":
    main()

