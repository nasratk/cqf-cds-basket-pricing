"""
CDS pricing and default time utilities.
"""
import numpy as np
import pandas as pd


def time_to_default(u: float, haz_rates: np.ndarray, tenors: np.ndarray) -> float:
    """
    Invert uniform to default time using piecewise-constant hazard rates.
    
    Parameters
    ----------
    u : float
        Uniform random variable in (0, 1)
    haz_rates : np.ndarray
        Piecewise hazard rates for each tenor interval
    tenors : np.ndarray
        Tenor points in years
        
    Returns
    -------
    float
        Default time (np.inf if no default by last tenor)
    """
    h = haz_rates
    t = tenors
    dt = np.diff(np.r_[0.0, t])
    H = np.cumsum(h * dt)  # Cumulative hazard
    y = -np.log(1.0 - u)

    j = int(np.searchsorted(H, y, side="left"))
    if j >= len(H):
        return np.inf

    H_prev = 0.0 if j == 0 else H[j - 1]
    t_prev = 0.0 if j == 0 else t[j - 1]
    return t_prev + (y - H_prev) / h[j]


def kth_to_default(ttd_array: np.ndarray, k: int) -> np.ndarray:
    """
    Extract k-th default time from each simulation row.
    
    Parameters
    ----------
    ttd_array : np.ndarray
        Shape (n_sims, n_entities) of time-to-default values
    k : int
        Which default to extract (1 = first, 2 = second, etc.)
        
    Returns
    -------
    np.ndarray
        Shape (n_sims,) of k-th default times
    """
    return np.sort(ttd_array, axis=1)[:, k - 1]


def kth_to_default_df(ttd_df: pd.DataFrame, tickers: list, k: int) -> np.ndarray:
    """
    Extract k-th default time from DataFrame (convenience wrapper).
    
    Parameters
    ----------
    ttd_df : pd.DataFrame
        DataFrame with time-to-default columns
    tickers : list
        Column names to use
    k : int
        Which default to extract (1 = first, 2 = second, etc.)
        
    Returns
    -------
    np.ndarray
        Shape (n_sims,) of k-th default times
    """
    X = ttd_df.loc[:, tickers].to_numpy()
    return np.sort(X, axis=1)[:, k - 1]


class CDSPricing:
    """Pricing for k-th to default CDS."""
    
    def __init__(self, time_to_default: np.ndarray, term: float, recovery_rate: float = 0.4):
        """
        Parameters
        ----------
        time_to_default : np.ndarray
            Array of default times from Monte Carlo simulation
        term : float
            CDS contract term in years
        recovery_rate : float
            Assumed recovery rate (default 0.4)
        """
        self.tau = np.asarray(time_to_default)
        self.T = float(term)
        self.R = float(recovery_rate)
        self.lgd = 1 - self.R

    def _default_leg(self) -> np.ndarray:
        """Protection leg: pays LGD if default occurs before maturity."""
        return self.lgd * (self.tau <= self.T).astype(float)
    
    def _unit_premium_leg(self) -> np.ndarray:
        """Premium leg: accrues until default or maturity."""
        return np.minimum(self.tau, self.T)
    
    def scenario_prices(self) -> np.ndarray:
        """
        Per-scenario breakeven spread (bps).
        
        Returns
        -------
        np.ndarray
            Breakeven spread for each simulation path
        """
        payout = self._default_leg()       
        unit_premium = self._unit_premium_leg()
        prices = np.where(
            unit_premium > 0.0,
            payout / unit_premium * 10000,  # in bps
            np.inf
        )
        return prices
    
    def price(self) -> float:
        """
        Fair spread (bps) as ratio of expected legs.
        
        Returns
        -------
        float
            Breakeven CDS spread in basis points
        """
        payout = self._default_leg()       
        unit_premium = self._unit_premium_leg()
        
        if np.any(unit_premium <= 0.0):
            return np.inf
        
        return float(payout.mean() / unit_premium.mean()) * 10000
    
    def se_price(self, is_antithetic: bool = False) -> float:
        """
        Standard error of fair spread estimate using delta method.
        
        The fair spread is a ratio of expectations: θ = E[X]/E[Y]
        Delta method gives: Var(θ̂) ≈ θ² × [Var(X)/μX² + Var(Y)/μY² − 2Cov(X,Y)/(μX×μY)] / N
        
        For antithetic variates, we compute pair averages first (pairs are 
        consecutive: (0,1), (2,3), ...), then apply delta method to N/2 
        independent pair averages.
        
        Parameters
        ----------
        is_antithetic : bool
            If True, account for antithetic pairing structure
        
        Returns
        -------
        float
            Standard error in basis points
        """
        X = self._default_leg()
        Y = self._unit_premium_leg()
        
        if is_antithetic:
            # Compute pair averages - pairs are consecutive (0,1), (2,3), ...
            X = (X[0::2] + X[1::2]) / 2
            Y = (Y[0::2] + Y[1::2]) / 2
        
        N = len(X)
        
        mu_X, mu_Y = X.mean(), Y.mean()
        var_X, var_Y = X.var(), Y.var()
        cov_XY = np.cov(X, Y)[0, 1]
        
        theta = mu_X / mu_Y
        
        var_ratio = (theta**2) * (var_X/mu_X**2 + var_Y/mu_Y**2 - 2*cov_XY/(mu_X*mu_Y)) / N
        
        return float(np.sqrt(var_ratio)) * 10000
