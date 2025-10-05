# 📊 Portfolio Optimizer

A powerful web application for building optimal stock portfolios with fundamental analysis.

## 🚀 Live Demo

**Deployed on Streamlit Cloud:** [https://portfolio-optimizer.streamlit.app](https://portfolio-optimizer.streamlit.app)

## ✨ Features

- 📈 **Portfolio Optimization** - Build optimal stock portfolios using modern algorithms
- 📊 **Fundamental Analysis** - Analyze stocks with 50+ fundamental metrics
- 🎯 **Risk Management** - Multiple risk measures and optimization strategies
- 📋 **Downloadable Reports** - Export results in CSV format
- 📱 **Mobile Friendly** - Works on any device with internet
- 🌐 **Real-time Data** - Live stock data from Yahoo Finance

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **Data:** Yahoo Finance API
- **Optimization:** Scikit-folio
- **Visualization:** Plotly
- **Backend:** Python

## 📦 Installation

### Local Development

1. Clone this repository:
```bash
git clone https://github.com/yourusername/portfolio-optimizer.git
cd portfolio-optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment

This app is designed to run on Streamlit Cloud:

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Deploy!

## 📊 Usage

### Building a Portfolio

1. **Select Stocks:** Enter stock symbols (e.g., RELIANCE, TCS, HDFCBANK)
2. **Set Weights:** Allocate percentages to each stock
3. **Optimize:** Click "Optimize Portfolio" to get the best allocation
4. **Analyze:** View fundamental metrics and risk analysis
5. **Download:** Export results as CSV

### Example Portfolio

Try these popular Indian stocks:
- **RELIANCE** (30%)
- **TCS** (25%)
- **HDFCBANK** (20%)
- **INFY** (15%)
- **ICICIBANK** (10%)

## 📈 Key Metrics

- **Expected Return** - Annualized portfolio return
- **Volatility** - Risk measure (standard deviation)
- **Sharpe Ratio** - Risk-adjusted return
- **PE Ratio** - Price-to-earnings ratio
- **ROE** - Return on equity
- **Market Cap** - Company valuation

## 🔧 Configuration

The app uses the following data sources:
- **Stock Data:** Yahoo Finance (real-time)
- **Fundamental Data:** Custom CSV file (datadump.csv)
- **Optimization:** Scikit-folio algorithms

## 📁 File Structure

```
├── streamlit_app.py              # Main application
├── requirements.txt              # Dependencies
├── portfolio_optimizer_fixed.py  # Core optimization logic
├── fundamental_analyzer_updated.py # Fundamental analysis
├── datadump.csv                  # Fundamental data
└── README.md                     # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure you have internet connection for stock data

## 🎯 Roadmap

- [ ] Add more optimization strategies
- [ ] Include international markets
- [ ] Add backtesting functionality
- [ ] Implement portfolio rebalancing
- [ ] Add more fundamental metrics

---

**Made with ❤️ for portfolio optimization**
