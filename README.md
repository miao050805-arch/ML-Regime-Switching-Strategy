# ML-Regime-Switching-Strategy 

## 👨‍🎓 Project Abstract
This project was developed during my **sophomore year** to explore how Machine Learning can identify market risk regimes. It dynamically allocates assets between **QQQ (Nasdaq 100)** and **TLT (Treasury Bonds)** based on volatility predictions from a Random Forest model.

## 💡 Key Improvements (Innovation)
I focused on moving beyond theoretical backtesting to address real-world trading challenges:
* **Noise Reduction**: Implemented a **5-day signal smoothing filter** to minimize over-trading and slippage costs. This optimization improved the Sharpe Ratio from **0.61 to 1.04**.
* **Integrity First**: Enforced a strict **T+1 execution lag** and accounted for **5bps transaction fees** to eliminate look-ahead bias and ensure realistic performance.

## 📊 Backtest Performance
| Metric | Result |
| :--- | :--- |
| **Sharpe Ratio** | **1.04** |
| **Max Drawdown** | **-9.70%** |
| **Out-of-Sample Return** | **16.48%** |

### Performance Visualization
![Strategy Results](Results.png)

## 🛠️ Tech Stack
* **Language**: Python
* **Library**: Scikit-Learn (Random Forest), Pandas, Yfinance
* **Metrics**: Sharpe Ratio, Max Drawdown, Regime Probability

---
*Disclaimer: For academic purposes only. Not financial advice.*
