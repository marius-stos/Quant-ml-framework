"""
Quant ML Framework - Core Module
=================================
Base classes and data generation utilities for systematic backtesting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Signal:
    """Represents a trading signal with metadata."""
    timestamp: pd.Timestamp
    asset: str
    direction: float          # -1 (short), 0 (flat), +1 (long)
    strength: float           # [0, 1] conviction
    strategy_id: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Represents an open position."""
    asset: str
    entry_price: float
    entry_time: pd.Timestamp
    size: float               # in units
    direction: float          # +1 long / -1 short
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 1_000_000.0
    commission_bps: float = 2.0           # basis points per trade
    slippage_bps: float = 1.0
    max_position_size: float = 0.10       # % of capital per trade
    max_drawdown_limit: float = 0.20      # kill switch at 20% drawdown
    risk_free_rate: float = 0.04          # annual
    rebalance_freq: str = "D"             # D / W / M
    leverage: float = 1.0


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────

class MarketDataGenerator:
    """
    Generates synthetic multi-asset OHLCV data with realistic properties:
    - Fat tails (Student-t returns)
    - Volatility clustering (GARCH-like)
    - Regime switches (bull/bear/sideways)
    - Cross-asset correlations
    """

    ASSETS = ["SPY", "QQQ", "TLT", "GLD", "EEM", "VIX_proxy", "HYG", "DXY"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_correlated_returns(
        self,
        n_days: int = 1500,
        n_assets: int = 8
    ) -> pd.DataFrame:
        """Generate correlated returns with regime switching."""

        # Correlation matrix (equity-heavy portfolio)
        corr = np.array([
            # SPY  QQQ  TLT  GLD  EEM  VIX  HYG  DXY
            [1.00, 0.92, -0.35, 0.05, 0.75, -0.80, 0.70, -0.30],
            [0.92, 1.00, -0.38, 0.02, 0.70, -0.82, 0.65, -0.28],
            [-0.35, -0.38, 1.00, 0.20, -0.25, 0.45, -0.40, 0.10],
            [0.05, 0.02, 0.20, 1.00, 0.10, -0.05, 0.08, -0.60],
            [0.75, 0.70, -0.25, 0.10, 1.00, -0.72, 0.60, -0.25],
            [-0.80, -0.82, 0.45, -0.05, -0.72, 1.00, -0.65, 0.20],
            [0.70, 0.65, -0.40, 0.08, 0.60, -0.65, 1.00, -0.20],
            [-0.30, -0.28, 0.10, -0.60, -0.25, 0.20, -0.20, 1.00],
        ])
        L = np.linalg.cholesky(corr)

        # Regime parameters: (mu_annual, vol_annual, persistence)
        regimes = {
            "bull":     (0.15, 0.12, 0.95),
            "bear":     (-0.25, 0.28, 0.80),
            "sideways": (0.02, 0.08, 0.90),
        }
        regime_names = list(regimes.keys())
        regime_seq = ["bull"]
        trans_probs = {"bull": [0.95, 0.03, 0.02],
                       "bear": [0.15, 0.80, 0.05],
                       "sideways": [0.10, 0.05, 0.85]}

        # Simulate regime path
        for _ in range(n_days - 1):
            current = regime_seq[-1]
            probs = trans_probs[current]
            next_r = self.rng.choice(regime_names, p=probs)
            regime_seq.append(next_r)

        # Generate returns per regime
        all_returns = []
        vol_state = np.ones(n_assets) * 0.12 / np.sqrt(252)

        for t, regime in enumerate(regime_seq):
            mu, sigma, _ = regimes[regime]
            # GARCH-like vol update
            if t > 0:
                prev_ret = all_returns[-1]
                vol_state = 0.92 * vol_state + 0.08 * np.abs(prev_ret)
                vol_state = np.clip(vol_state, 0.003, 0.06)

            # Student-t correlated shocks (fat tails, df=5)
            z = self.rng.standard_t(df=5, size=n_assets) / np.sqrt(5 / 3)
            corr_z = (L @ z)
            daily_vol = vol_state * (sigma / (0.15 / np.sqrt(252)))
            daily_mu = mu / 252
            ret = daily_mu + daily_vol * corr_z
            all_returns.append(ret)

        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
        df = pd.DataFrame(all_returns, index=dates, columns=self.ASSETS[:n_assets])
        df.attrs["regime_seq"] = regime_seq
        return df

    def returns_to_prices(self, returns: pd.DataFrame, start_price: float = 100.0) -> pd.DataFrame:
        """Convert log returns to price series."""
        prices = (1 + returns).cumprod() * start_price
        return prices

    def generate_full_dataset(self, n_days: int = 1500) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset: returns, prices, volumes, regimes."""
        returns = self.generate_correlated_returns(n_days)
        prices = self.returns_to_prices(returns)

        # Synthetic volumes (mean-reverting + correlated with abs(returns))
        base_vol = self.rng.lognormal(mean=15, sigma=0.5, size=(n_days, len(returns.columns)))
        vol_bump = 1 + 5 * np.abs(returns.values)
        volumes = pd.DataFrame(base_vol * vol_bump, index=returns.index, columns=returns.columns)

        regime_df = pd.DataFrame(
            {"regime": returns.attrs["regime_seq"]},
            index=returns.index
        )

        return {
            "returns": returns,
            "prices": prices,
            "volumes": volumes,
            "regimes": regime_df
        }


# ─────────────────────────────────────────────
# BASE STRATEGY CLASS
# ─────────────────────────────────────────────

class BaseStrategy(ABC):
    """Abstract base for all systematic strategies."""

    def __init__(self, name: str, config: Optional[BacktestConfig] = None):
        self.name = name
        self.config = config or BacktestConfig()
        self._signals: List[Signal] = []

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Returns a DataFrame: index=dates, columns=assets, values=position [-1, 0, 1].
        """
        pass

    def preprocess(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Optional preprocessing hook."""
        return data

    def __repr__(self):
        return f"<Strategy: {self.name}>"


print("core.py loaded")
