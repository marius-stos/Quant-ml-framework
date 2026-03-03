# ⬡ Quant ML Framework

A modular Python framework for systematic quantitative research, combining classical systematic strategies with modern ML signal processing and comprehensive risk analytics.

## Features

- **15+ Systematic Strategies** — momentum, mean-reversion, volatility, ML-enhanced
- **5 ML Models** — PCA, Kalman Filter, HMM, GARCH(1,1), Reinforcement Learning
- **50+ Performance & Risk Metrics** — Sharpe, Sortino, Calmar, Omega, CVaR, Hurst
- **Vectorized Backtester** — transaction costs, slippage, kill switch, leverage
- **Interactive Dashboard** — 5-tab HTML dashboard with Chart.js

## Project Structure
```
quant-ml-framework/
├── core.py                      # Data generation, base classes
├── main.py                      # Full pipeline runner
├── models/ml_models.py          # PCA, Kalman, HMM, GARCH, RL
├── strategies/strategies.py     # 15 systematic strategies
├── backtesting/backtester.py    # Vectorized backtester + 50+ metrics
└── dashboard/quant_ml_dashboard.html
```

## Quickstart
```bash
pip install -r requirements.txt
python main.py
```

## Tech Stack

`Python` · `NumPy` · `Pandas` · `SciPy` · `scikit-learn` · `Chart.js`
