# ML_Trader
# ETF Analysis with Machine Learning

This project analyzes Exchange-Traded Funds (ETFs) using historical price data and machine learning to provide buy, sell, or hold suggestions. The analysis utilizes a Random Forest Classifier to predict the direction of price movements based on historical returns.

> **Disclaimer**: This project is intended for educational and informational purposes only. The analysis and suggestions provided by this project are not financial advice. I am not a finance professional, and users should conduct their own research or consult with a qualified financial advisor before making any investment decisions.

## Features

- Fetches historical data for specified ETFs from Yahoo Finance.
- Creates features for machine learning models based on historical price movements.
- Uses a Random Forest Classifier to predict price direction.
- Generates buy, sell, or hold signals based on predictions.
- Visualizes cumulative returns and suggestions in a user-friendly chart with a dark mode theme.

## Requirements

Make sure you have Python installed along with the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `yfinance`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn yfinance
