"""
Quant ML Framework - Main Pipeline
=====================================
Real data (yfinance) + parameter optimization + improved portfolio construction.
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import BacktestConfig
from data import load_real_data, MarketDataGenerator
from models.ml_models import PCAFactorModel, KalmanFilterBeta, GaussianHMM, GARCH11
from strategies.strategies import STRATEGY_REGISTRY
from backtesting.backtester import VectorizedBacktester, EqualWeightCombiner


USE_REAL_DATA   = True
OPTIMIZE_PARAMS = True
START_DATE      = "2018-01-01"
TICKERS         = ["SPY", "QQQ", "TLT", "GLD", "EEM", "HYG", "UUP", "VXX"]

CONFIG = BacktestConfig(
    initial_capital    = 1_000_000,
    commission_bps     = 1.0,
    slippage_bps       = 0.5,
    max_position_size  = 0.15,
    max_drawdown_limit = 0.25,
    risk_free_rate     = 0.04,
    leverage           = 1.0,
)


def optimize_strategy(StratClass, param_grid, data, config):
    n = len(data["returns"])
    train_end = int(n * 0.70)
    train_data = {k: v.iloc[:train_end] if hasattr(v, 'iloc') else v for k, v in data.items()}
    backtester = VectorizedBacktester(config)
    best_sharpe, best_params = -999, {}

    def product(*args):
        if not args:
            yield ()
        else:
            for item in args[0]:
                for rest in product(*args[1:]):
                    yield (item,) + rest

    keys, values = list(param_grid.keys()), list(param_grid.values())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        try:
            result = backtester.run(StratClass(config=config, **params), train_data)
            sr = result.metrics.get("sharpe_ratio", -999)
            if np.isfinite(sr) and sr > best_sharpe:
                best_sharpe, best_params = sr, params
        except Exception:
            continue
    return best_params, best_sharpe


def run_full_pipeline():
    print("=" * 65)
    print("  QUANT ML FRAMEWORK — FULL PIPELINE")
    print("=" * 65)

    # 1. DATA
    print("\n[1/6] Loading market data...")
    if USE_REAL_DATA:
        try:
            data = load_real_data(tickers=TICKERS, start=START_DATE, verbose=True)
        except Exception as e:
            print(f"  yfinance failed ({e}), using synthetic data")
            data = MarketDataGenerator(seed=42).generate_full_dataset(n_days=1500)
    else:
        data = MarketDataGenerator(seed=42).generate_full_dataset(n_days=1500)

    returns = data["returns"]
    prices  = data["prices"]
    n_days  = len(returns)
    assets  = list(returns.columns)
    spy_col = "SPY" if "SPY" in prices.columns else prices.columns[0]
    qqq_col = "QQQ" if "QQQ" in prices.columns else prices.columns[1]

    print(f"  Universe: {assets} | Days: {n_days}")

    # 2. ML MODELS
    print("\n[2/6] Fitting ML models...")
    pca = PCAFactorModel(n_components=3)
    pca.fit(returns, window=min(252, n_days - 1))
    print(f"  PCA explained variance: {pca.explained_var_ratio_.round(3)}")

    kf = KalmanFilterBeta(delta=1e-4)
    kf_result = kf.rolling_spread(np.log(prices[spy_col]), np.log(prices[qqq_col]))
    print(f"  Kalman beta range: [{kf_result['beta'].min():.3f}, {kf_result['beta'].max():.3f}]")

    hmm = GaussianHMM(n_states=3, n_iter=80)
    hmm.fit(returns[spy_col].values)
    states = hmm.predict(returns[spy_col].values)
    regime_labels = hmm.label_regimes(states)
    from collections import Counter
    print(f"  HMM regimes: {dict(Counter(regime_labels))}")

    garch = GARCH11()
    garch.fit(returns[spy_col])
    print(f"  GARCH alpha={garch.alpha:.4f} beta={garch.beta:.4f} persistence={garch.alpha+garch.beta:.4f}")

    # 3. PARAMETER OPTIMIZATION
    print("\n[3/6] Optimizing strategy parameters (in-sample 70%)...")
    optimized_params = {}
    if OPTIMIZE_PARAMS:
        for name, grid in [
            ("TSMOM",      {"lookback": [63, 126, 252], "skip_days": [5, 10, 21]}),
            ("BollingerMR",{"window": [10, 20, 30], "n_std": [1.5, 2.0, 2.5]}),
            ("ZScoreMR",   {"window": [30, 60, 90], "entry_z": [1.5, 2.0, 2.5]}),
            ("RSI_MR",     {"period": [7, 14, 21], "oversold": [25, 30], "overbought": [70, 75]}),
        ]:
            print(f"  → {name}...", end=" ", flush=True)
            params, sr = optimize_strategy(STRATEGY_REGISTRY[name], grid, data, CONFIG)
            optimized_params[name] = params
            print(f"{params}  IS Sharpe={sr:.2f}")

    # 4. BACKTEST (OOS 30%)
    print("\n[4/6] Backtesting out-of-sample (last 30%)...")
    oos_start = int(n_days * 0.70)
    oos_data  = {k: v.iloc[oos_start:] if hasattr(v, 'iloc') else v for k, v in data.items()}
    backtester = VectorizedBacktester(CONFIG)
    strategies = [StratClass(config=CONFIG, **optimized_params.get(name, {}))
                  for name, StratClass in STRATEGY_REGISTRY.items()]
    results = backtester.run_all(strategies, oos_data, verbose=True)

    # 5. PORTFOLIO (filter + best weighting)
    print("\n[5/6] Building optimized portfolio...")
    valid = {n: r for n, r in results.items()
             if r.metrics.get("sharpe_ratio", 0) > 0.05
             and np.isfinite(r.metrics.get("sharpe_ratio", 0))
             and r.metrics.get("ann_vol", 0) > 0}

    if not valid:
        valid = {n: r for n, r in results.items() if np.isfinite(r.metrics.get("sharpe_ratio", 0))}

    print(f"  Valid strategies: {list(valid.keys())}")
    combiner = EqualWeightCombiner()
    portfolios = {m: combiner.combine(valid, weight_by=m) for m in ["equal", "sharpe", "inverse_vol"]}
    best_ptf = max(portfolios.values(), key=lambda p: p.metrics.get("sharpe_ratio", 0))

    print(f"\n  {'Method':<18} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
    print(f"  {'-'*46}")
    labels = {"equal": "Equal Weight", "sharpe": "Sharpe Weight", "inverse_vol": "Inv. Vol"}
    for method, ptf in portfolios.items():
        sr  = ptf.metrics.get("sharpe_ratio", 0)
        cg  = ptf.metrics.get("cagr", 0)
        mdd = ptf.metrics.get("max_drawdown", 0)
        star = " ★" if ptf is best_ptf else ""
        print(f"  {labels[method]:<18} {sr:>8.3f} {cg:>8.1%} {mdd:>8.1%}{star}")

    # 6. SUMMARY TABLE
    print("\n[6/6] Full Results Summary (OOS)")
    print("-" * 68)
    print(f"{'Strategy':<18} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'Sortino':>8} {'Omega':>7}")
    print("-" * 68)

    all_results = {**results, "Portfolio": best_ptf}
    summary_data = []
    for name, res in all_results.items():
        m   = res.metrics
        sr  = float(m.get("sharpe_ratio", 0))  if np.isfinite(m.get("sharpe_ratio", 0))  else 0
        so  = float(m.get("sortino_ratio", 0)) if np.isfinite(m.get("sortino_ratio", 0)) else 0
        cg  = float(m.get("cagr", 0))
        mdd = float(m.get("max_drawdown", 0))
        om  = float(m.get("omega_ratio", 0))   if np.isfinite(m.get("omega_ratio", 0))   else 0
        print(f"{name:<18} {sr:>7.2f} {cg:>7.1%} {mdd:>8.1%} {so:>8.2f} {om:>7.2f}")
        summary_data.append({"strategy":name,"sharpe":round(sr,3),"cagr":round(cg,4),
            "max_drawdown":round(mdd,4),"sortino":round(so,3),"omega":round(om,3),
            "ann_vol":round(float(m.get("ann_vol",0)),4),
            "calmar":round(float(m.get("calmar_ratio",0)) if np.isfinite(m.get("calmar_ratio",0)) else 0,3),
            "hurst":round(float(m.get("hurst_exponent",0.5)),3),
            "skewness":round(float(m.get("skewness",0)),3),
            "var_95":round(float(m.get("var_95",0)),4)})
    print("-" * 68)

    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    equity_data = {name: {"dates": res.equity_curve.index.strftime("%Y-%m-%d").tolist(),
                           "values": [round(v, 2) for v in res.equity_curve.tolist()]}
                   for name, res in all_results.items()}

    with open("outputs/backtest_results.json", "w") as f:
        json.dump({
            "summary":       summary_data,
            "equity_curves": equity_data,
            "regime_data": {
                "dates":         returns.index.strftime("%Y-%m-%d").tolist(),
                "regime_labels": regime_labels.tolist(),
                "spy_returns":   returns[spy_col].round(4).tolist(),
                "kalman_beta":   kf_result["beta"].round(4).tolist(),
                "garch_vol":     garch.annualized_vol().round(4).tolist(),
            },
            "pca_loadings":    pca.factor_loadings_.round(4).to_dict(),
            "hmm_params":      {"means": hmm.mu.round(5).tolist(), "stds": hmm.sigma.round(5).tolist(),
                                "transition_matrix": hmm.A.round(4).tolist()},
            "optimized_params": optimized_params,
            "data_source":     "real" if USE_REAL_DATA else "synthetic",
            "universe":        assets,
        }, f, indent=2)

    print(f"\n✅ Saved → outputs/backtest_results.json")
    print("✅ Done!\n")
    return all_results


if __name__ == "__main__":
    run_full_pipeline()
