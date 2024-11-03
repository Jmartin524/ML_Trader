import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Define the list of ETFs you want to analyze
etf_list = ['SPY', 'IVV', 'VOO', 'SCHD', 'QQQ']  # Example ETF tickers

def fetch_data(etf):
    """Fetch historical data for the specified ETF."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)  # 5 years ago from today
    data = yf.download(etf, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    return data

def create_features(data):
    """Create features for the model."""
    data['Returns'] = data['Adj Close'].pct_change()
    data['Direction'] = np.where(data['Returns'] > 0, 1, 0)
    return data.dropna()

def analyze_etf(etf):
    historical_data = fetch_data(etf)
    historical_data = create_features(historical_data)
    
    # Prepare data for the model
    X = historical_data[['Returns']].shift().dropna()  # Use previous returns as features
    y = historical_data['Direction'].shift().dropna()  # Use current direction as target
    
    # Align X and y
    X = X.loc[y.index]  # Align indices
    
    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train the Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Store results
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions
    })

    # Calculate strategy returns based on predictions
    results['Strategy_Returns'] = np.where(results['Predicted'] == 1, historical_data['Returns'][train_size + 1:], 0)
    results['Cumulative_Returns'] = (1 + results['Strategy_Returns']).cumprod() - 1
    
    # Generate buy/sell/hold signals
    latest_prediction = results['Predicted'].iloc[-1]
    if latest_prediction == 1:
        signal = "Buy"
    elif latest_prediction == 0:
        signal = "Sell"
    else:
        signal = "Hold"

    return results, signal

# Dark mode settings
plt.style.use('dark_background')

# Analyze multiple ETFs
plt.figure(figsize=(12, 6))
colors = ['cyan', 'magenta', 'yellow', 'orange', 'green']  # Updated colors for each ETF
signals = []  # To store signals for legend
for etf, color in zip(etf_list, colors):
    results, signal = analyze_etf(etf)
    plt.plot(results['Cumulative_Returns'], label=etf, color=color)
    signals.append((etf, signal))  # Store ETF and its signal

# Customize chart appearance
plt.title('Cumulative Returns of Multiple ETFs', color='white')
plt.xlabel('Date', color='white')
plt.ylabel('Cumulative Returns', color='white')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tick_params(axis='both', colors='white')
plt.gca().set_facecolor('black')  # Set axes background color

# Create custom legend outside of the plot
plt.subplots_adjust(right=0.85)  # Adjust subplot to make space for the text
for i, (etf, signal) in enumerate(signals):
    plt.text(1.02, 0.9 - i * 0.05, f"{etf}: {signal}", color=colors[i], transform=plt.gca().transAxes, fontsize=12, ha='left')

# Show the plot
plt.show()
