"""
Quant ML Framework - Systematic Strategies Module
===================================================
15+ alpha-generating strategies across momentum, mean-reversion,
carry, volatility, and ML-enhanced approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import BaseStrategy, BacktestConfig


# ═══════════════════════════════════════════════
# MOMENTUM STRATEGIES
# ═══════════════════════════════════════════════

class TimeSeriesMomentum(BaseStrategy):
    """TSMOM: Long assets with positive trailing return, short negative."""

    def __init__(self, lookback: int = 252, skip_days: int = 21, **kwargs):
        super().__init__("TSMOM", **kwargs)
        self.lookback = lookback
        self.skip_days = skip_days

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        prices = data["prices"]
        past_return = prices.shift(self.skip_days) / prices.shift(self.lookback) - 1
        signals = np.sign(past_return)
        return signals.fillna(0)


class CrossSectionalMomentum(BaseStrategy):
    """CSMom: Long top quintile, short bottom quintile cross-sectionally."""

    def __init__(self, lookback: int = 126, n_long: int = 2, n_short: int = 2, **kwargs):
        super().__init__("CSMom", **kwargs)
        self.lookback = lookback
        self.n_long = n_long
        self.n_short = n_short

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        returns = data["returns"]
        momentum = returns.rolling(self.lookback).sum()
        signals = pd.DataFrame(0.0, index=momentum.index, columns=momentum.columns)
        for date in momentum.index:
            row = momentum.loc[date].dropna()
            if len(row) < self.n_long + self.n_short:
                continue
            sorted_assets = row.sort_values()
            short_assets = sorted_assets.index[:self.n_short]
            long_assets = sorted_assets.index[-self.n_long:]
            signals.loc[date, long_assets] = 1.0 / self.n_long
            signals.loc[date, short_assets] = -1.0 / self.n_short
        return signals


class DualMomentum(BaseStrategy):
    """Antonacci Dual Momentum: absolute + relative momentum filter."""

    def __init__(self, lookback: int = 252, **kwargs):
        super().__init__("DualMom", **kwargs)
        self.lookback = lookback

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        prices = data["prices"]
        abs_mom = prices / prices.shift(self.lookback) - 1
        # Relative: rank within universe
        rel_rank = abs_mom.rank(axis=1, pct=True)
        # Both conditions must hold
        signals = ((abs_mom > 0) & (rel_rank > 0.6)).astype(float)
        signals[abs_mom < 0] = -0.5
        return signals.fillna(0)


class BreakoutMomentum(BaseStrategy):
    """Donchian channel breakout strategy."""

    def __init__(self, window: int = 55, exit_window: int = 20, **kwargs):
        super().__init__("Breakout", **kwargs)
        self.window = window
        self.exit_window = exit_window

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        prices = data["prices"]
        high_channel = prices.rolling(self.window).max().shift(1)
        low_channel = prices.rolling(self.window).min().shift(1)
        exit_high = prices.rolling(self.exit_window).max().shift(1)
        exit_low = prices.rolling(self.exit_window).min().shift(1)

        long_entry = (prices >= high_channel).astype(float)
        short_entry = (prices <= low_channel).astype(float) * -1
        signals = long_entry + short_entry
        return signals.fillna(0)


# ═══════════════════════════════════════════════
# MEAN REVERSION STRATEGIES
# ═══════════════════════════════════════════════

class BollingerBandReversion(BaseStrategy):
    """Mean-reversion using Bollinger Bands."""

    def __init__(self, window: int = 20, n_std: float = 2.0, **kwargs):
        super().__init__("BollingerMR", **kwargs)
        self.window = window
        self.n_std = n_std

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        prices = data["prices"]
        ma = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        upper = ma + self.n_std * std
        lower = ma - self.n_std * std

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[prices > upper] = -1.0    # short at upper band
        signals[prices < lower] = 1.0     # long at lower band
        return signals.fillna(0)


class ZScoreMeanReversion(BaseStrategy):
    """Z-score based statistical mean-reversion."""

    def __init__(self, window: int = 60, entry_z: float = 2.0, exit_z: float = 0.5, **kwargs):
        super().__init__("ZScoreMR", **kwargs)
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        prices = data["prices"]
        log_p = np.log(prices)
        ma = log_p.rolling(self.window).mean()
        std = log_p.rolling(self.window).std() + 1e-10
        z = (log_p - ma) / std

        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signals[z > self.entry_z] = -1.0
        signals[z < -self.entry_z] = 1.0
        signals[z.abs() < self.exit_z] = 0.0
        return signals.fillna(0)


class KalmanPairsTrading(BaseStrategy):
    """Pairs trading using Kalman Filter dynamic hedge ratio."""

    def __init__(self, delta: float = 1e-4, entry_z: float = 2.0, **kwargs):
        super().__init__("KalmanPairs", **kwargs)
        self.delta = delta
        self.entry_z = entry_z

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        from models.ml_models import KalmanFilterBeta
        prices = data["prices"]
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        kf = KalmanFilterBeta(delta=self.delta)
        # Use SPY-QQQ as primary pair
        if "SPY" in prices.columns and "QQQ" in prices.columns:
            result = kf.rolling_spread(
                np.log(prices["SPY"]),
                np.log(prices["QQQ"])
            )
            signals["SPY"] = np.where(result["z_score"] > self.entry_z, -1,
                             np.where(result["z_score"] < -self.entry_z, 1, 0))
            signals["QQQ"] = -signals["SPY"]  # hedge leg
        return signals.fillna(0)


class RSIMeanReversion(BaseStrategy):
    """RSI-based overbought/oversold strategy."""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, **kwargs):
        super().__init__("RSI_MR", **kwargs)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean() + 1e-10
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        prices = data["prices"]
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            rsi = self._compute_rsi(prices[col])
            signals[col] = np.where(rsi < self.oversold, 1.0,
                           np.where(rsi > self.overbought, -1.0, 0.0))
        return signals.fillna(0)


# ═══════════════════════════════════════════════
# VOLATILITY STRATEGIES
# ═══════════════════════════════════════════════

class VolatilityTargeting(BaseStrategy):
    """
    Scale positions to target a fixed annualized volatility.
    Base signals from momentum, rescaled by GARCH vol forecast.
    """

    def __init__(self, target_vol: float = 0.10, vol_lookback: int = 60, **kwargs):
        super().__init__("VolTarget", **kwargs)
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        returns = data["returns"]
        realized_vol = returns.rolling(self.vol_lookback).std() * np.sqrt(252) + 1e-6
        # Naive direction: sign of 1-month momentum
        direction = np.sign(returns.rolling(21).sum())
        # Scale by vol
        position_size = self.target_vol / realized_vol
        position_size = position_size.clip(0, 2.0)  # max 2x leverage
        return (direction * position_size).fillna(0)


class VIXRegimeSwitching(BaseStrategy):
    """Switch between risk-on and risk-off based on VIX proxy level."""

    def __init__(self, vol_threshold: float = 0.20, **kwargs):
        super().__init__("VIXRegime", **kwargs)
        self.vol_threshold = vol_threshold

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        returns = data["returns"]
        prices = data["prices"]
        realized_vol = returns["SPY"].rolling(21).std() * np.sqrt(252) if "SPY" in returns.columns \
                       else returns.iloc[:, 0].rolling(21).std() * np.sqrt(252)

        risk_on = (realized_vol < self.vol_threshold).astype(float)
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Risk-on: long equities, risk-off: long bonds/gold
        risk_on_assets = [c for c in ["SPY", "QQQ", "EEM"] if c in prices.columns]
        risk_off_assets = [c for c in ["TLT", "GLD"] if c in prices.columns]

        for a in risk_on_assets:
            signals[a] = risk_on
        for a in risk_off_assets:
            signals[a] = 1 - risk_on

        return signals.fillna(0)


class ShortVolatility(BaseStrategy):
    """Short realized volatility strategy: sell variance when vol is elevated."""

    def __init__(self, vol_window: int = 21, percentile_threshold: float = 75, **kwargs):
        super().__init__("ShortVol", **kwargs)
        self.vol_window = vol_window
        self.percentile_threshold = percentile_threshold

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        returns = data["returns"]
        realized_vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        roll_percentile = realized_vol.rolling(252).apply(
            lambda x: (x[-1] > np.percentile(x, self.percentile_threshold)) * 1.0
        )
        # Short when vol is high (collect vol premium)
        signals = -roll_percentile * 0.5
        return signals.fillna(0)


# ═══════════════════════════════════════════════
# ML-ENHANCED STRATEGIES
# ═══════════════════════════════════════════════

class PCAStatisticalArbitrage(BaseStrategy):
    """
    PCA-based statistical arbitrage.
    Trade mean-reversion of idiosyncratic residuals.
    """

    def __init__(self, n_components: int = 3, z_entry: float = 2.0, z_exit: float = 0.5,
                 fit_window: int = 252, **kwargs):
        super().__init__("PCA_StatArb", **kwargs)
        self.n_components = n_components
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.fit_window = fit_window

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        from models.ml_models import PCAFactorModel
        returns = data["returns"]
        signals = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

        pca = PCAFactorModel(n_components=self.n_components)
        pca.fit(returns, window=self.fit_window)
        residuals = pca.get_residuals(returns)
        z_scores = pca.get_z_scores(residuals)

        signals[z_scores > self.z_entry] = -1.0
        signals[z_scores < -self.z_entry] = 1.0
        signals[(z_scores.abs() < self.z_exit)] = 0.0

        return signals.fillna(0)


class HMMRegimeStrategy(BaseStrategy):
    """
    HMM-based regime switching strategy.
    Adjusts position sizing and direction based on detected regime.
    """

    def __init__(self, n_regimes: int = 3, fit_window: int = 500, **kwargs):
        super().__init__("HMM_Regime", **kwargs)
        self.n_regimes = n_regimes
        self.fit_window = fit_window

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        from models.ml_models import GaussianHMM
        returns = data["returns"]
        prices = data["prices"]
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Use SPY as regime indicator
        spy_ret = returns["SPY"].dropna() if "SPY" in returns.columns else returns.iloc[:, 0].dropna()

        if len(spy_ret) < self.fit_window:
            return signals

        hmm = GaussianHMM(n_states=self.n_regimes)
        train_data = spy_ret.values[:self.fit_window]
        hmm.fit(train_data)

        # Predict on full series
        all_states = hmm.predict(spy_ret.values)
        regime_labels = hmm.label_regimes(all_states)

        # Regime-conditional positioning
        regime_position = {"Bull": 1.0, "Sideways": 0.3, "Bear": -0.5}
        default_pos = {"Bull": 1.0, "Bear": -0.5}

        for i, date in enumerate(spy_ret.index):
            if i >= len(regime_labels):
                break
            regime = regime_labels.iloc[i] if i < len(regime_labels) else "Sideways"
            pos = regime_position.get(regime, default_pos.get(regime, 0.3))
            for col in prices.columns:
                if col in ["TLT", "GLD"]:  # defensive assets — inverse
                    signals.loc[date, col] = -pos * 0.5
                else:
                    signals.loc[date, col] = pos

        return signals.fillna(0)


class GARCHVolStrategy(BaseStrategy):
    """
    GARCH-based strategy: trade volatility forecasts.
    Long when vol expanding, scale down when vol contracting.
    """

    def __init__(self, vol_window: int = 60, **kwargs):
        super().__init__("GARCH_Vol", **kwargs)
        self.vol_window = vol_window

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        from models.ml_models import GARCH11
        returns = data["returns"]
        signals = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

        for col in returns.columns[:4]:  # limit to avoid long compute
            r = returns[col].dropna()
            if len(r) < 200:
                continue
            try:
                garch = GARCH11()
                garch.fit(r)
                ann_vol = garch.annualized_vol()
                # Signal: position inversely proportional to vol (vol targeting flavor)
                target_vol = 0.15
                pos_size = (target_vol / ann_vol).clip(0.1, 2.0)
                direction = np.sign(r.rolling(21).sum())
                sig = direction * pos_size
                signals[col] = sig.reindex(signals.index).fillna(0)
            except Exception:
                pass

        return signals


class RLEnhancedMomentum(BaseStrategy):
    """RL Q-Learning agent trained on momentum + vol + regime features."""

    def __init__(self, train_ratio: float = 0.7, n_episodes: int = 30, **kwargs):
        super().__init__("RL_Momentum", **kwargs)
        self.train_ratio = train_ratio
        self.n_episodes = n_episodes

    def generate_signals(self, data: Dict) -> pd.DataFrame:
        from models.ml_models import QLearningTrader, GaussianHMM
        returns = data["returns"]
        prices = data["prices"]
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Focus on SPY
        asset = "SPY" if "SPY" in returns.columns else returns.columns[0]
        r = returns[asset].dropna()

        if len(r) < 300:
            return signals

        # Features
        momentum = r.rolling(21).sum()
        volatility = r.rolling(21).std() * np.sqrt(252)

        # Quick HMM for regime
        hmm = GaussianHMM(n_states=3, n_iter=30)
        hmm.fit(r.values)
        regime_raw = hmm.predict(r.values)
        regimes = pd.Series(regime_raw, index=r.index)

        # Split train/test
        n_train = int(len(r) * self.train_ratio)

        agent = QLearningTrader(n_episodes=self.n_episodes)
        agent.train(
            r.iloc[:n_train],
            momentum.iloc[:n_train].fillna(0),
            volatility.iloc[:n_train].fillna(0.15),
            regimes.iloc[:n_train].fillna(0)
        )

        # Out-of-sample positions
        oos_pos = agent.predict(
            momentum.fillna(0),
            volatility.fillna(0.15),
            regimes.fillna(0)
        )
        signals[asset] = oos_pos.reindex(signals.index).fillna(0)
        return signals


# ═══════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════

STRATEGY_REGISTRY = {
    # Momentum
    "TSMOM":         TimeSeriesMomentum,
    "CSMom":         CrossSectionalMomentum,
    "DualMom":       DualMomentum,
    "Breakout":      BreakoutMomentum,
    # Mean Reversion
    "BollingerMR":   BollingerBandReversion,
    "ZScoreMR":      ZScoreMeanReversion,
    "KalmanPairs":   KalmanPairsTrading,
    "RSI_MR":        RSIMeanReversion,
    # Volatility
    "VolTarget":     VolatilityTargeting,
    "VIXRegime":     VIXRegimeSwitching,
    "ShortVol":      ShortVolatility,
    # ML-Enhanced
    "PCA_StatArb":   PCAStatisticalArbitrage,
    "HMM_Regime":    HMMRegimeStrategy,
    "GARCH_Vol":     GARCHVolStrategy,
    "RL_Momentum":   RLEnhancedMomentum,
}


def get_all_strategies(config: Optional[BacktestConfig] = None) -> list:
    """Instantiate all registered strategies."""
    return [cls(config=config) for cls in STRATEGY_REGISTRY.values()]


print(f"strategies.py loaded — {len(STRATEGY_REGISTRY)} strategies registered")
