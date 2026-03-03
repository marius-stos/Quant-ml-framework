"""
Quant ML Framework - Data Module
==================================
Fetches real market data via yfinance.
Falls back to synthetic data if no internet.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# REAL DATA LOADER (yfinance)
# ─────────────────────────────────────────────

UNIVERSE = {
    "SPY":  "S&P 500 ETF",
    "QQQ":  "Nasdaq 100 ETF",
    "TLT":  "20Y Treasury ETF",
    "GLD":  "Gold ETF",
    "EEM":  "Emerging Markets ETF",
    "HYG":  "High Yield Bond ETF",
    "DXY":  "US Dollar (UUP proxy)",
    "VXX":  "VIX Short-Term Futures",
}

def load_real_data(
    tickers: list = None,
    start: str = "2018-01-01",
    end: str = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Load real OHLCV data from Yahoo Finance.

    Returns dict with keys: returns, prices, volumes
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    if tickers is None:
        tickers = list(UNIVERSE.keys())

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    if verbose:
        print(f"  Downloading {len(tickers)} assets from {start} to {end}...")

    # Download all tickers
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True
    )

    # Extract Adj Close
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(how="all")
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]}).dropna()

    # Keep only business days with enough data (>50% non-NaN)
    prices = prices.dropna(thresh=int(len(tickers) * 0.5))
    prices = prices.fillna(method="ffill").fillna(method="bfill")

    returns = prices.pct_change().dropna()
    prices  = prices.loc[returns.index]

    # Volumes
    if isinstance(raw.columns, pd.MultiIndex):
        volumes = raw["Volume"].reindex(returns.index).fillna(0)
    else:
        volumes = pd.DataFrame(index=returns.index, columns=prices.columns, data=1e6)

    if verbose:
        print(f"  ✅ Loaded {len(prices)} trading days × {len(prices.columns)} assets")
        print(f"  Period: {prices.index[0].date()} → {prices.index[-1].date()}")
        ann_rets = (1 + returns).prod() ** (252 / len(returns)) - 1
        ann_vols = returns.std() * np.sqrt(252)
        for col in prices.columns:
            sr = ann_rets[col] / ann_vols[col] if ann_vols[col] > 0 else 0
            print(f"    {col:<8} CAGR={ann_rets[col]:+.1%}  Vol={ann_vols[col]:.1%}  Sharpe={sr:.2f}")

    return {
        "returns": returns,
        "prices":  prices,
        "volumes": volumes,
        "regimes": pd.DataFrame({"regime": "Unknown"}, index=returns.index),
    }


# ─────────────────────────────────────────────
# SYNTHETIC FALLBACK
# ─────────────────────────────────────────────

class MarketDataGenerator:
    """Synthetic data generator (fallback / testing)."""

    ASSETS = ["SPY", "QQQ", "TLT", "GLD", "EEM", "VXX", "HYG", "DXY"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_correlated_returns(self, n_days: int = 1500, n_assets: int = 8) -> pd.DataFrame:
        corr = np.array([
            [1.00, 0.92,-0.35, 0.05, 0.75,-0.80, 0.70,-0.30],
            [0.92, 1.00,-0.38, 0.02, 0.70,-0.82, 0.65,-0.28],
            [-0.35,-0.38, 1.00, 0.20,-0.25, 0.45,-0.40, 0.10],
            [0.05, 0.02, 0.20, 1.00, 0.10,-0.05, 0.08,-0.60],
            [0.75, 0.70,-0.25, 0.10, 1.00,-0.72, 0.60,-0.25],
            [-0.80,-0.82, 0.45,-0.05,-0.72, 1.00,-0.65, 0.20],
            [0.70, 0.65,-0.40, 0.08, 0.60,-0.65, 1.00,-0.20],
            [-0.30,-0.28, 0.10,-0.60,-0.25, 0.20,-0.20, 1.00],
        ])
        L = np.linalg.cholesky(corr)
        regimes = {"bull":(0.15,0.12,0.95),"bear":(-0.25,0.28,0.80),"sideways":(0.02,0.08,0.90)}
        regime_names = list(regimes.keys())
        regime_seq = ["bull"]
        trans = {"bull":[0.95,0.03,0.02],"bear":[0.15,0.80,0.05],"sideways":[0.10,0.05,0.85]}
        for _ in range(n_days-1):
            regime_seq.append(self.rng.choice(regime_names, p=trans[regime_seq[-1]]))
        all_returns = []
        vol_state = np.ones(n_assets)*0.12/np.sqrt(252)
        for t, regime in enumerate(regime_seq):
            mu, sigma, _ = regimes[regime]
            if t > 0:
                vol_state = 0.92*vol_state + 0.08*np.abs(all_returns[-1])
                vol_state = np.clip(vol_state, 0.003, 0.06)
            z = self.rng.standard_t(df=5, size=n_assets)/np.sqrt(5/3)
            ret = mu/252 + vol_state*(sigma/(0.15/np.sqrt(252)))*(L@z)
            all_returns.append(ret)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
        df = pd.DataFrame(all_returns, index=dates, columns=self.ASSETS[:n_assets])
        df.attrs["regime_seq"] = regime_seq
        return df

    def returns_to_prices(self, returns, start_price=100.0):
        return (1+returns).cumprod()*start_price

    def generate_full_dataset(self, n_days=1500):
        returns = self.generate_correlated_returns(n_days)
        prices  = self.returns_to_prices(returns)
        base_vol = self.rng.lognormal(15, 0.5, (n_days, len(returns.columns)))
        volumes  = pd.DataFrame(base_vol*(1+5*np.abs(returns.values)), index=returns.index, columns=returns.columns)
        regime_df = pd.DataFrame({"regime": returns.attrs["regime_seq"]}, index=returns.index)
        return {"returns":returns,"prices":prices,"volumes":volumes,"regimes":regime_df}
