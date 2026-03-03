"""
Quant ML Framework - ML Models Module
======================================
PCA factor model, Kalman Filter, Hidden Markov Model,
GARCH volatility, and Reinforcement Learning agent.
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional, List


# ─────────────────────────────────────────────
# 1. PCA FACTOR MODEL
# ─────────────────────────────────────────────

class PCAFactorModel:
    """
    Statistical factor decomposition via PCA.
    
    Decomposes cross-sectional returns into:
    - Market factor (PC1)
    - Sector / risk-on-off factors (PC2, PC3)
    - Idiosyncratic alpha (residuals)
    
    Applications:
    - Risk decomposition
    - Statistical arbitrage (mean-reversion on residuals)
    - Portfolio hedging
    """

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None   # [n_comp, n_assets]
        self.explained_var_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.factor_loadings_: Optional[pd.DataFrame] = None

    def fit(self, returns: pd.DataFrame, window: int = 252) -> "PCAFactorModel":
        """Fit PCA on rolling window of returns."""
        X = returns.tail(window).values
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-10
        X_std = (X - self.mean_) / self.std_

        # SVD decomposition
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        total_var = (S ** 2).sum()
        self.explained_var_ratio_ = (S[:self.n_components] ** 2) / total_var
        self.components_ = Vt[:self.n_components]

        self.factor_loadings_ = pd.DataFrame(
            self.components_.T,
            index=returns.columns,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )
        return self

    def get_factor_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Project returns onto factor space."""
        X = returns.values
        X_std = (X - self.mean_) / self.std_
        factor_ret = X_std @ self.components_.T
        return pd.DataFrame(
            factor_ret,
            index=returns.index,
            columns=[f"PC{i+1}" for i in range(self.n_components)]
        )

    def get_residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Extract idiosyncratic residuals (alpha signals)."""
        X = returns.values
        X_std = (X - self.mean_) / self.std_
        factor_ret = X_std @ self.components_.T
        reconstruction = factor_ret @ self.components_
        residuals_std = X_std - reconstruction
        residuals = residuals_std * self.std_
        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

    def get_z_scores(self, residuals: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """Compute rolling z-scores of residuals for mean-reversion signals."""
        roll_mean = residuals.rolling(lookback).mean()
        roll_std = residuals.rolling(lookback).std() + 1e-10
        return (residuals - roll_mean) / roll_std


# ─────────────────────────────────────────────
# 2. KALMAN FILTER (Dynamic Beta Estimation)
# ─────────────────────────────────────────────

class KalmanFilterBeta:
    """
    Kalman Filter for dynamic hedge ratio / beta estimation.
    
    State: β_t (time-varying regression coefficient)
    Observation: y_t = β_t * x_t + ε_t
    Transition:  β_t = β_{t-1} + η_t
    
    Applications:
    - Dynamic pairs trading
    - Time-varying factor exposure
    - Adaptive signal filtering
    """

    def __init__(
        self,
        delta: float = 1e-4,       # state noise (how fast beta evolves)
        R: float = 1e-2,           # observation noise variance
    ):
        self.delta = delta
        self.R = R

    def filter(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run Kalman filter.
        
        Args:
            y: dependent variable (e.g., asset A returns)
            X: independent variable (e.g., asset B returns)
        
        Returns:
            dict with beta estimates, variances, innovations
        """
        n = len(y)
        beta = np.zeros(n)          # filtered state
        P = np.zeros(n)             # state variance
        innovations = np.zeros(n)   # prediction errors
        innovation_var = np.zeros(n)

        # Initial state
        beta[0] = 0.0
        P[0] = 1.0
        Q = self.delta / (1 - self.delta)  # process noise

        for t in range(1, n):
            # Predict
            beta_pred = beta[t-1]
            P_pred = P[t-1] + Q

            # Update
            H = X[t]
            S = H * P_pred * H + self.R    # innovation variance
            K = P_pred * H / (S + 1e-12)   # Kalman gain

            innovations[t] = y[t] - H * beta_pred
            innovation_var[t] = S

            beta[t] = beta_pred + K * innovations[t]
            P[t] = (1 - K * H) * P_pred

        return {
            "beta": beta,
            "variance": P,
            "innovations": innovations,
            "innovation_var": innovation_var,
            "z_score": innovations / (np.sqrt(innovation_var) + 1e-12)
        }

    def rolling_spread(
        self,
        asset_a: pd.Series,
        asset_b: pd.Series
    ) -> pd.DataFrame:
        """Compute dynamic spread for pairs trading."""
        result = self.filter(asset_a.values, asset_b.values)
        spread = asset_a.values - result["beta"] * asset_b.values
        spread_mean = pd.Series(spread).rolling(30).mean().values
        spread_std = pd.Series(spread).rolling(30).std().values + 1e-10

        return pd.DataFrame({
            "spread": spread,
            "beta": result["beta"],
            "z_score": (spread - spread_mean) / spread_std,
            "innovation_z": result["z_score"]
        }, index=asset_a.index)


# ─────────────────────────────────────────────
# 3. HIDDEN MARKOV MODEL (Regime Detection)
# ─────────────────────────────────────────────

class GaussianHMM:
    """
    Gaussian Hidden Markov Model for market regime detection.
    
    States: Bull / Bear / Sideways (N_STATES regimes)
    Observation: daily returns (or vol-adjusted returns)
    
    Uses Baum-Welch (EM) for parameter estimation.
    Viterbi for most-likely state sequence.
    
    Applications:
    - Regime-conditional position sizing
    - Strategy rotation
    - Risk-on/risk-off switching
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol

        # Parameters (initialized randomly)
        self.pi = None          # initial state probs [n_states]
        self.A = None           # transition matrix [n_states, n_states]
        self.mu = None          # emission means [n_states]
        self.sigma = None       # emission stds [n_states]

    def _init_params(self, X: np.ndarray):
        """Initialize with K-means-like partitioning."""
        percentiles = np.linspace(0, 100, self.n_states + 1)
        cuts = np.percentile(X, percentiles)

        self.mu = np.array([(cuts[i] + cuts[i+1]) / 2 for i in range(self.n_states)])
        self.sigma = np.full(self.n_states, X.std())
        self.A = np.full((self.n_states, self.n_states), 1 / self.n_states)
        np.fill_diagonal(self.A, 0.7)
        self.A /= self.A.sum(axis=1, keepdims=True)
        self.pi = np.full(self.n_states, 1 / self.n_states)

    def _emission_log_prob(self, X: np.ndarray) -> np.ndarray:
        """Log emission probabilities [T, n_states]."""
        log_b = np.zeros((len(X), self.n_states))
        for k in range(self.n_states):
            log_b[:, k] = norm.logpdf(X, self.mu[k], self.sigma[k] + 1e-8)
        return log_b

    def _forward(self, log_b: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm in log space."""
        T = len(log_b)
        log_alpha = np.full((T, self.n_states), -np.inf)
        log_alpha[0] = np.log(self.pi + 1e-12) + log_b[0]
        log_A = np.log(self.A + 1e-12)

        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = np.logaddexp.reduce(
                    log_alpha[t-1] + log_A[:, j]
                ) + log_b[t, j]

        log_likelihood = np.logaddexp.reduce(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        """Backward algorithm in log space."""
        T = len(log_b)
        log_beta = np.zeros((T, self.n_states))
        log_A = np.log(self.A + 1e-12)

        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = np.logaddexp.reduce(
                    log_A[i] + log_b[t+1] + log_beta[t+1]
                )
        return log_beta

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Baum-Welch EM algorithm."""
        self._init_params(X)
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            log_b = self._emission_log_prob(X)
            log_alpha, log_ll = self._forward(log_b)
            log_beta = self._backward(log_b)
            log_A = np.log(self.A + 1e-12)

            # E-step: compute posteriors
            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # M-step: update parameters
            self.pi = gamma[0] / gamma[0].sum()

            # Transition matrix
            new_A = np.zeros((self.n_states, self.n_states))
            for t in range(len(X) - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        new_A[i, j] += np.exp(
                            log_alpha[t, i] + log_A[i, j] + log_b[t+1, j] + log_beta[t+1, j] - log_ll
                        )
            self.A = new_A / (new_A.sum(axis=1, keepdims=True) + 1e-12)

            # Emission parameters
            for k in range(self.n_states):
                denom = gamma[:, k].sum() + 1e-12
                self.mu[k] = (gamma[:, k] * X).sum() / denom
                self.sigma[k] = np.sqrt(
                    (gamma[:, k] * (X - self.mu[k]) ** 2).sum() / denom
                ) + 1e-6

            if abs(log_ll - prev_ll) < self.tol:
                break
            prev_ll = log_ll

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most-likely state sequence."""
        T = len(X)
        log_b = self._emission_log_prob(X)
        log_A = np.log(self.A + 1e-12)

        delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = np.log(self.pi + 1e-12) + log_b[0]

        for t in range(1, T):
            for j in range(self.n_states):
                scores = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + log_b[t, j]

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities."""
        log_b = self._emission_log_prob(X)
        log_alpha, log_ll = self._forward(log_b)
        log_beta = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def label_regimes(self, states: np.ndarray) -> pd.Series:
        """Label regimes by their mean return: bull/sideways/bear."""
        labels = {i: f"Regime_{i}" for i in range(self.n_states)}
        sorted_states = np.argsort(self.mu)
        name_map = {sorted_states[0]: "Bear", sorted_states[-1]: "Bull"}
        if self.n_states == 3:
            name_map[sorted_states[1]] = "Sideways"
        labels = {k: name_map.get(k, f"Regime_{k}") for k in range(self.n_states)}
        return pd.Series([labels[s] for s in states])


# ─────────────────────────────────────────────
# 4. GARCH(1,1) VOLATILITY MODEL
# ─────────────────────────────────────────────

class GARCH11:
    """
    GARCH(1,1) model for conditional volatility estimation.
    
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Parameters estimated via MLE (quasi-Newton).
    
    Applications:
    - Dynamic position sizing (vol targeting)
    - Option-adjusted signals
    - Risk-on/risk-off regime detection via vol level
    """

    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta  = None
        self.cond_vol_ = None

    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = returns.var()

        for t in range(1, T):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        sigma2 = np.maximum(sigma2, 1e-8)
        nll = 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        return nll

    def fit(self, returns: pd.Series) -> "GARCH11":
        """Estimate GARCH parameters via MLE."""
        r = returns.dropna().values
        var = r.var()

        x0 = [var * 0.05, 0.10, 0.85]
        bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]
        constraints = [{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}]

        result = minimize(
            self._neg_log_likelihood,
            x0, args=(r,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500}
        )

        self.omega, self.alpha, self.beta = result.x
        self.cond_vol_ = self._compute_vol(r)
        return self

    def _compute_vol(self, returns: np.ndarray) -> np.ndarray:
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = returns.var()
        for t in range(1, T):
            sigma2[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * sigma2[t-1]
        return np.sqrt(np.maximum(sigma2, 1e-8))

    def predict_vol(self, returns: pd.Series, horizon: int = 10) -> np.ndarray:
        """h-step ahead volatility forecast."""
        r = returns.values
        sigma2_last = self.cond_vol_[-1] ** 2
        long_run = self.omega / (1 - self.alpha - self.beta + 1e-12)
        forecasts = np.zeros(horizon)
        for h in range(horizon):
            if h == 0:
                sigma2_h = self.omega + (self.alpha + self.beta) * sigma2_last
            else:
                sigma2_h = self.omega + (self.alpha + self.beta) * forecasts[h-1]**2
                sigma2_h = long_run + (self.alpha + self.beta)**h * (sigma2_last - long_run)
            forecasts[h] = np.sqrt(max(sigma2_h, 1e-8))
        return forecasts

    def annualized_vol(self) -> pd.Series:
        """Return annualized conditional volatility series."""
        if self.cond_vol_ is None:
            raise RuntimeError("Model not fitted.")
        return pd.Series(self.cond_vol_ * np.sqrt(252))


# ─────────────────────────────────────────────
# 5. REINFORCEMENT LEARNING - Q-LEARNING AGENT
# ─────────────────────────────────────────────

class QLearningTrader:
    """
    Tabular Q-Learning agent for position management.
    
    State space: discretized (momentum, volatility, regime)
    Action space: {-1: short, 0: flat, +1: long}
    Reward: PnL - transaction_cost - risk_penalty
    
    Applications:
    - Adaptive position sizing
    - Entry/exit timing optimization
    - Multi-regime strategy switching
    """

    ACTIONS = [-1, 0, 1]   # short, flat, long

    def __init__(
        self,
        n_mom_bins: int = 5,
        n_vol_bins: int = 3,
        n_regimes: int = 3,
        alpha: float = 0.1,       # learning rate
        gamma: float = 0.95,      # discount factor
        epsilon: float = 0.20,    # exploration rate
        epsilon_decay: float = 0.995,
        risk_aversion: float = 0.5,
    ):
        self.n_mom_bins = n_mom_bins
        self.n_vol_bins = n_vol_bins
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.risk_aversion = risk_aversion

        # Q-table: state x action
        state_size = n_mom_bins * n_vol_bins * n_regimes
        self.Q = np.zeros((state_size, len(self.ACTIONS)))
        self.action_map = {a: i for i, a in enumerate(self.ACTIONS)}

        # Discretization bins (set during fit)
        self.mom_bins = None
        self.vol_bins = None

    def _encode_state(self, mom: float, vol: float, regime: int) -> int:
        mom_idx = min(np.searchsorted(self.mom_bins, mom), self.n_mom_bins - 1)
        vol_idx = min(np.searchsorted(self.vol_bins, vol), self.n_vol_bins - 1)
        return regime * (self.n_mom_bins * self.n_vol_bins) + vol_idx * self.n_mom_bins + mom_idx

    def _compute_reward(
        self, pnl: float, position: float, prev_position: float, vol: float
    ) -> float:
        transaction_cost = abs(position - prev_position) * 0.001
        risk_penalty = self.risk_aversion * vol * abs(position)
        return pnl - transaction_cost - risk_penalty

    def train(
        self,
        returns: pd.Series,
        momentum: pd.Series,
        volatility: pd.Series,
        regimes: pd.Series,
        n_episodes: int = 50,
    ) -> Dict[str, List]:
        """Train the Q-learning agent."""
        # Set discretization bins
        self.mom_bins = np.percentile(
            momentum.dropna(), np.linspace(0, 100, self.n_mom_bins + 1)[1:-1]
        )
        self.vol_bins = np.percentile(
            volatility.dropna(), np.linspace(0, 100, self.n_vol_bins + 1)[1:-1]
        )

        episode_rewards = []
        episode_sharpes = []

        for episode in range(n_episodes):
            position = 0
            episode_pnl = []
            n = len(returns)

            for t in range(1, n):
                try:
                    state = self._encode_state(
                        float(momentum.iloc[t]),
                        float(volatility.iloc[t]),
                        int(regimes.iloc[t])
                    )
                except (ValueError, IndexError):
                    continue

                # ε-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.choice(self.ACTIONS)
                else:
                    action = self.ACTIONS[np.argmax(self.Q[state])]

                # Step environment
                pnl = action * float(returns.iloc[t])
                reward = self._compute_reward(pnl, action, position, float(volatility.iloc[t]))
                episode_pnl.append(pnl)

                # Next state
                if t + 1 < n:
                    try:
                        next_state = self._encode_state(
                            float(momentum.iloc[t+1]),
                            float(volatility.iloc[t+1]),
                            int(regimes.iloc[t+1]) if t+1 < len(regimes) else 0
                        )
                        next_best = np.max(self.Q[next_state])
                    except:
                        next_best = 0.0
                else:
                    next_best = 0.0

                # Q-update (Bellman)
                a_idx = self.action_map[action]
                td_error = reward + self.gamma * next_best - self.Q[state, a_idx]
                self.Q[state, a_idx] += self.alpha * td_error
                position = action

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

            if episode_pnl:
                total_reward = sum(episode_pnl)
                pnl_arr = np.array(episode_pnl)
                sharpe = (pnl_arr.mean() / (pnl_arr.std() + 1e-8)) * np.sqrt(252)
                episode_rewards.append(total_reward)
                episode_sharpes.append(sharpe)

        return {"rewards": episode_rewards, "sharpes": episode_sharpes}

    def predict(
        self,
        momentum: pd.Series,
        volatility: pd.Series,
        regimes: pd.Series
    ) -> pd.Series:
        """Generate out-of-sample positions."""
        positions = []
        for t in range(len(momentum)):
            try:
                state = self._encode_state(
                    float(momentum.iloc[t]),
                    float(volatility.iloc[t]),
                    int(regimes.iloc[t])
                )
                action = self.ACTIONS[np.argmax(self.Q[state])]
            except:
                action = 0
            positions.append(action)
        return pd.Series(positions, index=momentum.index)


print("✅ models.py loaded")
