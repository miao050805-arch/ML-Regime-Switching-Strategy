# Machine Learning Regime Switching Strategy
# This project implements a simple ML-based asset allocation strategy
# that dynamically switches between equities (QQQ) and bonds (TLT)
# based on predicted market volatility regimes.
# Author: Yachen Miao

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set visual style
plt.style.use("seaborn-v0_8")

class MLRegimeStrategy:

    def __init__(self, ticker_eq="QQQ", ticker_bond="TLT", fee=0.0005):

        # Setting up trading pair: one for "attack" (Stocks) and one for "defense" (Bonds)
        self.ticker_eq = ticker_eq
        self.ticker_bond = ticker_bond

        # Simple assumption for trading cost (0.05%)
        # We subtract this cost whenever the strategy switches position
        self.fee = fee

        # Unified feature names
        self.feature_names = ["rsi", "vol_ratio", "corr"]

        # Random Forest model for regime prediction
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=40
        )

        # Standardize features before training the model
        self.scaler = StandardScaler()

    def fetch_data(self):

        # Download historical price data from Yahoo Finance
        raw_data= yf.download(
            [self.ticker_eq, self.ticker_bond],
            start="2020-01-01",
            auto_adjust=True
        )

        # Only keep closing prices and remove missing values
        data = raw_data["Close"].dropna()

        return data

    def build_features(self, data):

        # Create a dataframe to store features
        df = pd.DataFrame(index=data.index)

        # Log Returns for mathematical consistency
        df["eq_ret"] = np.log(data[self.ticker_eq] / data[self.ticker_eq].shift(1))
        df["bond_ret"] = np.log(data[self.ticker_bond] / data[self.ticker_bond].shift(1))

        # Feature 1: RSI (Relative Strength Index)
        # Add 1e-9 to prevent the loss from being 0
        delta = data[self.ticker_eq].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # Feature 2: Volatility Ratio (Short-term vs Long-term Volatility)
        df["vol"] = df["eq_ret"].rolling(21).std() * np.sqrt(252)
        df["vol_ratio"] = df["vol"] / df["vol"].shift(1).rolling(63).mean()

        # Feature 3: Rolling Correlation between Stocks and Bonds
        df["corr"] = df["eq_ret"].rolling(21).corr(df["bond_ret"])

        # Target variable
        # Use global 80% volatility threshold as a simple regime definition
        vol_threshold = df["vol"].quantile(0.8)

        # 1 = Safe Vol (Stocks), 0 = Risky Vol (Bonds)
        df["target"] = np.where(df["vol"].shift(-5) > vol_threshold,0,1)

        # Remove rows with missing values
        return df.dropna()

    def run_backtest(self):

        # Download price data
        data = self.fetch_data()

        # Create ML features
        df = self.build_features(data)
        features = ["rsi", "vol_ratio", "corr"]

        # Train/Test Split (80% training, 20% out-of-sample)
        split = int(len(df) * 0.8)
        train = df.iloc[:split].copy()
        test = df.iloc[split:].copy()

        # Scale features before training
        M_train = self.scaler.fit_transform(train[features])
        M_test = self.scaler.transform(test[features])

        # Train the Random Forest model
        self.model.fit(M_train, train["target"])

        # Predict probability of high-volatility regime
        test["prob"] = self.model.predict_proba(M_test)[:, 1]

        # Signal Smoothing
        test["smooth_prob"] = test["prob"].rolling(window=5).mean().fillna(0.5)

        # Execution Logic: Binary switch based on smoothed signals
        # Shift(1) is critical to avoid look-ahead bias
        test["eq_weight"] = np.where(test["smooth_prob"] < 0.5, 1.0, 0.0)
        test["eq_weight"] = test["eq_weight"].shift(1).fillna(0)
        test["bond_weight"] = 1 - test["eq_weight"]
        test["raw_ret"] = (test["eq_weight"] * test["eq_ret"] + test["bond_weight"] * test["bond_ret"])

        # Estimate trading turnover and subtract fees
        test["trades"] = test["eq_weight"].diff().abs().fillna(0)
        test["net_ret"] = test["raw_ret"] - test["trades"] * self.fee

        # Create a table showing portfolio allocation
        allocation = pd.DataFrame(index=test.index)
        allocation["QQQ_weight"] = test["eq_weight"]
        allocation["TLT_weight"] = test["bond_weight"]

        # Print the last few allocations
        print("Latest portfolio allocation:")
        print(allocation.tail(3))

        return test

    def evaluate(self, results):

        # Convert log returns back to simple returns
        returns = np.exp(results["net_ret"]) - 1

        # Calculate cumulative return
        cum_ret = (1 + returns).cumprod()


        # Sharpe ratio (simple risk-adjusted return metric)
        sharpe = (returns.mean() / returns.std() + 1e-9) * np.sqrt(252)

        # Maximum drawdown (largest peak-to-trough loss)
        drawdown = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
        max_dd = drawdown.min()

        # Total return
        total_return = cum_ret.iloc[-1] - 1

        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Max Drawdown: {max_dd:.3%}")
        print(f"Total Return: {total_return:.3%}")

        # Figure 1
        # Plot the strategy equity curve
        plt.figure(figsize=(10, 5))
        plt.plot(cum_ret, label="Strategy")
        plt.title("Regime Switching Strategy")
        plt.legend()
        plt.show()

        # Figure 2
        # Strategy Equity Curve vs Benchmarks
        qqq_returns = np.exp(results["eq_ret"]) - 1
        qqq_curve = (1 + qqq_returns).cumprod()

        portfolio_60_40 = 0.6 * (np.exp(results["eq_ret"]) - 1) + 0.4 * (np.exp(results["bond_ret"]) - 1)
        portfolio_curve = (1 + portfolio_60_40).cumprod()

        # Normalize all curves to start at 1.0
        cum_ret = cum_ret / cum_ret.iloc[0]
        qqq_curve = qqq_curve / qqq_curve.iloc[0]
        portfolio_curve = portfolio_curve / portfolio_curve.iloc[0]

        plt.figure(figsize=(10, 5))
        plt.plot(cum_ret, label="ML Strategy (Dynamic)", linewidth = 2)
        plt.plot(qqq_curve, label="QQQ Benchmark", linestyle = "--")
        plt.plot(portfolio_curve, label="60/40 Portfolio", linestyle = ":")
        plt.title("Strategy vs Benchmark")
        plt.legend()
        plt.show()

        # Figure 3
        # Feature importance
        importance = self.model.feature_importances_
        features = ["RSI", "Volatility Ratio", "Correlation"]
        plt.figure(figsize=(10, 5))
        plt.bar(features, importance)
        plt.title("Feature Importance")
        plt.ylabel("Importance")
        plt.show()

        # Figure 4
        # Predicted volatility regime
        plt.figure(figsize=(10, 5))
        plt.plot(results["smooth_prob"], label="Smoothed Risk Probability")
        plt.axhline(0.5, linestyle="--")
        plt.title("Predicted Market Risk Regime")
        plt.ylabel("Probability Score")
        plt.show()


# Run the strategy
if __name__ == "__main__":
    strategy = MLRegimeStrategy()
    results = strategy.run_backtest()
    strategy.evaluate(results)

    last_row = results.iloc[-1]
    last_date = results.index[-1].strftime('%Y-%m-%d')
    decision = "QQQ (STOCKS)" if last_row["eq_weight"] == 1.0 else "TLT (BONDS)"

    print(f"Trading Signal For Tomorrow ({last_date})")
    print(f"Recommended Asset: {decision}")
