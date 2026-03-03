"""
Quant ML Framework - Main Pipeline
====================================
Runs full simulation: data generation → ML models → backtesting → metrics.
"""

import numpy as np
import pandas as pd
import json
import sys
import os
sys.path.insert(0, "/home/claude/quant_ml_framework")

from core import MarketDataGenerator, BacktestConfig
from models.ml_models import PCAFactorModel, KalmanFilterBeta, GaussianHMM, GARCH11, QLearningTrader
from strategies.strategies import STRATEGY_REGISTRY, get_all_strategies
from backtesting.backtester import VectorizedBacktester, EqualWeightCombiner, PerformanceMetrics

import warnings
warnings.filterwarnings("ignore")


def run_full_pipeline():
    print("=" * 65)
    print("  QUANT ML FRAMEWORK — FULL PIPELINE")
    print("=" * 65)

    # ── 1. DATA GENERATION ──────────────────────────────────────
    print("\n[1/5] Generating synthetic market data...")
    generator = MarketDataGenerator(seed=42)
    data = generator.generate_full_dataset(n_days=1500)
    returns = data["returns"]
    prices = data["prices"]
    print(f"      Assets: {list(returns.columns)}")
    print(f"      Period: {returns.index[0].date()} → {returns.index[-1].date()} ({len(returns)} days)")

    # ── 2. ML MODELS ────────────────────────────────────────────
    print("\n[2/5] Fitting ML models...")

    # PCA
    print("  → PCA Factor Model...")
    pca = PCAFactorModel(n_components=3)
    pca.fit(returns, window=252)
    print(f"     Explained variance: {pca.explained_var_ratio_.round(3)}")

    # Kalman Filter
    print("  → Kalman Filter (SPY-QQQ beta)...")
    kf = KalmanFilterBeta(delta=1e-4)
    kf_result = kf.rolling_spread(
        np.log(prices["SPY"]),
        np.log(prices["QQQ"])
    )
    print(f"     Beta range: [{kf_result['beta'].min():.3f}, {kf_result['beta'].max():.3f}]")

    # HMM
    print("  → HMM Regime Detection (3 states)...")
    spy_ret = returns["SPY"].values
    hmm = GaussianHMM(n_states=3, n_iter=50)
    hmm.fit(spy_ret)
    states = hmm.predict(spy_ret)
    regime_labels = hmm.label_regimes(states)
    regime_counts = regime_labels.value_counts()
    print(f"     Regimes: {dict(regime_counts)}")
    print(f"     Bull μ={hmm.mu[np.argmax(hmm.mu)]:.4f}, Bear μ={hmm.mu[np.argmin(hmm.mu)]:.4f}")

    # GARCH
    print("  → GARCH(1,1) Volatility (SPY)...")
    garch = GARCH11()
    garch.fit(returns["SPY"])
    print(f"     ω={garch.omega:.2e}, α={garch.alpha:.4f}, β={garch.beta:.4f}")
    ann_vol = garch.annualized_vol()
    print(f"     Current ann. vol: {ann_vol.iloc[-1]:.1%}")

    # ── 3. STRATEGIES (fast subset) ─────────────────────────────
    print("\n[3/5] Running strategy backtests...")
    config = BacktestConfig(
        initial_capital=1_000_000,
        commission_bps=2.0,
        slippage_bps=1.0,
        max_position_size=0.10,
        max_drawdown_limit=0.25,
    )
    backtester = VectorizedBacktester(config=config)

    # Use fast strategies for demo
    fast_strategies = [
        STRATEGY_REGISTRY["TSMOM"](config=config),
        STRATEGY_REGISTRY["CSMom"](config=config),
        STRATEGY_REGISTRY["BollingerMR"](config=config),
        STRATEGY_REGISTRY["ZScoreMR"](config=config),
        STRATEGY_REGISTRY["VolTarget"](config=config),
        STRATEGY_REGISTRY["VIXRegime"](config=config),
        STRATEGY_REGISTRY["RSI_MR"](config=config),
        STRATEGY_REGISTRY["PCA_StatArb"](config=config),
        STRATEGY_REGISTRY["HMM_Regime"](config=config),
        STRATEGY_REGISTRY["Breakout"](config=config),
        STRATEGY_REGISTRY["DualMom"](config=config),
        STRATEGY_REGISTRY["GARCH_Vol"](config=config),
    ]

    results = backtester.run_all(fast_strategies, data, verbose=True)

    # ── 4. PORTFOLIO COMBINATION ─────────────────────────────────
    print("\n[4/5] Building combined portfolio...")
    combiner = EqualWeightCombiner()
    portfolio = combiner.combine(results, weight_by="sharpe")
    print(f"  Portfolio Sharpe: {portfolio.metrics['sharpe_ratio']:.2f}")
    print(f"  Portfolio CAGR:   {portfolio.metrics['cagr']:.1%}")
    print(f"  Max Drawdown:     {portfolio.metrics['max_drawdown']:.1%}")
    print(f"  Calmar Ratio:     {portfolio.metrics['calmar_ratio']:.2f}")

    # ── 5. METRICS SUMMARY ──────────────────────────────────────
    print("\n[5/5] Metrics Summary Table")
    print("-" * 65)
    header = f"{'Strategy':<18} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'Omega':>7}"
    print(header)
    print("-" * 65)

    all_results = {**results, "Portfolio": portfolio}
    summary_data = []

    for name, res in all_results.items():
        m = res.metrics
        sr = m.get("sharpe_ratio", 0)
        cagr = m.get("cagr", 0)
        mdd = m.get("max_drawdown", 0)
        calmar = m.get("calmar_ratio", 0)
        omega = m.get("omega_ratio", 0)
        print(f"{name:<18} {sr:>7.2f} {cagr:>7.1%} {mdd:>7.1%} {calmar:>7.2f} {omega:>7.2f}")
        summary_data.append({
            "strategy": name,
            "sharpe": round(sr, 3),
            "cagr": round(cagr, 4),
            "max_drawdown": round(mdd, 4),
            "calmar": round(calmar, 3),
            "omega": round(omega, 3),
            "sortino": round(m.get("sortino_ratio", 0), 3),
            "ann_vol": round(m.get("ann_vol", 0), 4),
            "var_95": round(m.get("var_95", 0), 4),
            "hurst": round(m.get("hurst_exponent", 0.5), 3),
            "skewness": round(m.get("skewness", 0), 3),
            "kurtosis": round(m.get("excess_kurtosis", 0), 3),
        })

    print("-" * 65)

    # Save results for dashboard
    os.makedirs("/home/claude/quant_ml_framework/outputs", exist_ok=True)

    # Equity curves
    equity_data = {}
    for name, res in all_results.items():
        equity_data[name] = {
            "dates": res.equity_curve.index.strftime("%Y-%m-%d").tolist(),
            "values": res.equity_curve.round(2).tolist()
        }

    # Regime data
    regime_data = {
        "dates": returns.index.strftime("%Y-%m-%d").tolist(),
        "regime_labels": regime_labels.tolist(),
        "spy_returns": returns["SPY"].round(4).tolist(),
        "kalman_beta": kf_result["beta"].round(4).tolist(),
        "kalman_zscore": kf_result["z_score"].fillna(0).round(4).tolist(),
        "garch_vol": ann_vol.round(4).tolist(),
        "pca_pc1": pca.get_factor_returns(returns)["PC1"].round(4).tolist(),
    }

    # Save to JSON
    with open("/home/claude/quant_ml_framework/outputs/backtest_results.json", "w") as f:
        json.dump({
            "summary": summary_data,
            "equity_curves": equity_data,
            "regime_data": regime_data,
            "pca_loadings": pca.factor_loadings_.round(4).to_dict(),
            "hmm_params": {
                "means": hmm.mu.round(5).tolist(),
                "stds": hmm.sigma.round(5).tolist(),
                "transition_matrix": hmm.A.round(4).tolist(),
            }
        }, f, indent=2)

    print(f"\n Results saved to outputs/backtest_results.json")
    print(" Pipeline complete!\n")
    return all_results, regime_data, pca, hmm, garch, kf_result


if __name__ == "__main__":
    run_full_pipeline()
