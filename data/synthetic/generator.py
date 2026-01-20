"""
Synthetic CDS Data Generator

Generates correlated CDS spread time series with regime switching,
and term structure snapshots for basket CDS pricing analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Regime:
    """Defines a volatility/spread regime period."""
    name: str
    start: str  # YYYY-MM format
    end: str    # YYYY-MM format
    vol_multiplier: float = 1.0
    spread_shift_pct: float = 0.0  # percentage shift to mean (e.g., 0.5 = 50% higher)
    correlation_multiplier: float = 1.0  # scales off-diagonal correlations toward 1


@dataclass
class EntityConfig:
    """Configuration for a single reference entity."""
    name: str
    ticker: str
    sector: str
    base_spread_5y: float        # long-run mean 5Y spread (bps)
    volatility_pct: float        # annualised volatility as decimal (e.g., 0.30 = 30%)
    mean_reversion_speed: float = 1.0  # kappa (annualised), 1.0 = half-life ~8 months


class CDSDataGenerator:
    """
    Generates synthetic CDS spread data with:
    - Correlated movements across entities
    - Regime-dependent volatility and spread levels
    - Log-normal mean-reverting dynamics (ensures positive spreads)
    
    Model: Ornstein-Uhlenbeck on log(spread)
        d(log S) = κ * (log θ - log S) * dt + σ * dW
    
    This gives:
        - Positive spreads always
        - Percentage-based volatility (more realistic)
        - Mean reversion to base level
    
    Usage:
        gen = CDSDataGenerator(seed=42)
        gen.set_entities([...])
        gen.set_correlation_matrix(corr)
        gen.set_regimes([...])
        ts_df = gen.generate_time_series(...)
        curve_df = gen.generate_curve_snapshot(...)
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.entities: List[EntityConfig] = []
        self.correlation_matrix: Optional[np.ndarray] = None
        self.regimes: List[Regime] = []
        
        # Populated after generation
        self._time_series: Optional[pd.DataFrame] = None
        self._realised_correlation: Optional[np.ndarray] = None
    
    def set_entities(self, entities: List[EntityConfig]) -> 'CDSDataGenerator':
        """
        Set the reference entities.
        
        Parameters
        ----------
        entities : list of EntityConfig
            Configuration for each reference entity
        
        Returns
        -------
        self : for method chaining
        """
        self.entities = entities
        return self
    
    def set_correlation_matrix(self, corr_matrix: np.ndarray) -> 'CDSDataGenerator':
        """
        Set the target correlation matrix for spread changes.
        
        Parameters
        ----------
        corr_matrix : np.ndarray
            n x n correlation matrix (must be positive semi-definite)
        
        Returns
        -------
        self : for method chaining
        
        Raises
        ------
        ValueError
            If matrix is not valid (not square, not symmetric, not PSD)
        """
        n = len(self.entities)
        if corr_matrix.shape != (n, n):
            raise ValueError(f"Correlation matrix must be {n}x{n}, got {corr_matrix.shape}")
        
        if not np.allclose(corr_matrix, corr_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(f"Correlation matrix must be positive semi-definite. "
                           f"Min eigenvalue: {eigenvalues.min():.6f}")
        
        self.correlation_matrix = corr_matrix
        return self
    
    def set_regimes(self, regimes: List[Regime]) -> 'CDSDataGenerator':
        """
        Set the regime structure.
        
        Parameters
        ----------
        regimes : list of Regime
            Regime definitions (should cover entire simulation period)
        
        Returns
        -------
        self : for method chaining
        """
        # Sort by start date
        self.regimes = sorted(regimes, key=lambda r: r.start)
        return self
    
    def _get_regime_for_date(self, date: pd.Timestamp) -> Optional[Regime]:
        """Get the regime active at a given date."""
        date_str = date.strftime("%Y-%m")
        for regime in self.regimes:
            if regime.start <= date_str <= regime.end:
                return regime
        return None
    
    def _apply_correlation_stress(self, base_corr: np.ndarray, multiplier: float) -> np.ndarray:
        """
        Scale correlations toward 1 during stress (contagion effect).
        
        multiplier > 1: correlations increase toward 1
        multiplier = 1: no change
        """
        if multiplier == 1.0:
            return base_corr
        
        # Scale off-diagonal elements toward 1
        stressed = base_corr.copy()
        n = len(base_corr)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Interpolate toward 1
                    stressed[i, j] = base_corr[i, j] + (1 - base_corr[i, j]) * (multiplier - 1) / multiplier
                    stressed[i, j] = min(stressed[i, j], 0.99)  # Cap at 0.99
        
        # Ensure PSD
        stressed = self._nearest_psd(stressed)
        return stressed
    
    def _nearest_psd(self, matrix: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Ensure diagonal is 1
        d = np.sqrt(np.diag(result))
        result = result / np.outer(d, d)
        return result
    
    def _cholesky_decompose(self, corr_matrix: np.ndarray) -> np.ndarray:
        """Compute Cholesky decomposition for correlation matrix."""
        try:
            return np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Fall back to nearest PSD
            psd = self._nearest_psd(corr_matrix)
            return np.linalg.cholesky(psd)
    
    def generate_time_series(
        self,
        start: str,
        end: str,
        freq: str = "M"
    ) -> pd.DataFrame:
        """
        Generate correlated CDS spread time series.
        
        Uses log-OU dynamics:
            d(log S) = κ * (log θ - log S) * dt + σ * √dt * dW
        
        Parameters
        ----------
        start : str
            Start date (YYYY-MM-DD)
        end : str
            End date (YYYY-MM-DD)
        freq : str
            Frequency ('M' for monthly, 'W' for weekly)
        
        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            [date, reference_entity, ticker, sector, cds_5y_spread_bps]
        """
        if not self.entities:
            raise ValueError("No entities configured. Call set_entities() first.")
        if self.correlation_matrix is None:
            raise ValueError("No correlation matrix set. Call set_correlation_matrix() first.")
        
        # Generate date index
        dates = pd.date_range(start=start, end=end, freq=freq)
        n_periods = len(dates)
        n_entities = len(self.entities)
        
        # Time step (in years)
        if freq == "M":
            dt = 1/12
        elif freq == "W":
            dt = 1/52
        else:
            dt = 1/12  # default to monthly
        
        # Initialize spread paths (in log space)
        log_spreads = np.zeros((n_periods, n_entities))
        
        # Set initial spreads at base levels
        for i, entity in enumerate(self.entities):
            log_spreads[0, i] = np.log(entity.base_spread_5y)
        
        # Store changes for correlation calculation
        spread_changes = []
        
        # Simulate path
        for t in range(1, n_periods):
            date = dates[t]
            regime = self._get_regime_for_date(date)
            
            # Get regime parameters (default to normal if no regime defined)
            vol_mult = regime.vol_multiplier if regime else 1.0
            spread_shift_pct = regime.spread_shift_pct if regime else 0.0
            corr_mult = regime.correlation_multiplier if regime else 1.0
            
            # Apply correlation stress if needed
            corr = self._apply_correlation_stress(self.correlation_matrix, corr_mult)
            chol = self._cholesky_decompose(corr)
            
            # Generate correlated standard normals
            z_indep = self.rng.standard_normal(n_entities)
            z_corr = chol @ z_indep
            
            # Update each entity's spread (log-OU dynamics)
            for i, entity in enumerate(self.entities):
                log_S_prev = log_spreads[t-1, i]
                kappa = entity.mean_reversion_speed
                
                # Shifted target during stress
                theta = entity.base_spread_5y * (1 + spread_shift_pct)
                log_theta = np.log(theta)
                
                # Volatility (annualised, scaled by regime)
                sigma = entity.volatility_pct * vol_mult
                
                # Log-OU: d(log S) = κ * (log θ - log S) * dt + σ * √dt * dW
                drift = kappa * (log_theta - log_S_prev) * dt
                diffusion = sigma * np.sqrt(dt) * z_corr[i]
                
                log_spreads[t, i] = log_S_prev + drift + diffusion
            
            # Track changes (in level space) for realised correlation
            spreads_prev = np.exp(log_spreads[t-1, :])
            spreads_curr = np.exp(log_spreads[t, :])
            spread_changes.append(spreads_curr - spreads_prev)
        
        # Convert to levels
        spreads = np.exp(log_spreads)
        
        # Calculate realised correlation from changes
        if spread_changes:
            changes_arr = np.array(spread_changes)
            self._realised_correlation = np.corrcoef(changes_arr.T)
        
        # Build long-format DataFrame
        records = []
        for t, date in enumerate(dates):
            for i, entity in enumerate(self.entities):
                records.append({
                    "date": date,
                    "reference_entity": entity.name,
                    "ticker": entity.ticker,
                    "sector": entity.sector,
                    "cds_5y_spread_bps": round(spreads[t, i], 1)
                })
        
        self._time_series = pd.DataFrame(records)
        return self._time_series
    
    def generate_curve_snapshot(
        self,
        as_of: str,
        tenors: List[int] = [1, 2, 3, 4, 5],
        recovery_rate: float = 0.4,
        base_k: float = 0.30,
        spread_sensitivity: float = 0.002
    ) -> pd.DataFrame:
        """
        Generate CDS curve snapshot, anchored to terminal 5Y spreads from time series.
        
        Uses exponential saturation form for realistic term structure:
            spread(T) = spread_5Y × (1 - e^(-k×T)) / (1 - e^(-k×5))
        
        This gives:
        - Steep rise at short tenors, flattening toward 5Y
        - Decreasing deltas between successive tenors
        - 5Y typically 2-3x of 1Y spread
        
        Curve shape varies by credit quality:
        - Low spread (high quality) → steeper curve (lower k)
        - High spread (lower quality) → flatter curve (higher k)
        
        Parameters
        ----------
        as_of : str
            Snapshot date (YYYY-MM-DD). Should match end of time series.
        tenors : list of int
            Tenor points in years
        recovery_rate : float
            Assumed recovery rate for hazard rate calculation
        base_k : float
            Base steepness parameter. Default 0.30.
            Lower k = steeper curve (more curvature).
        spread_sensitivity : float
            How much spread level flattens the curve. Default 0.002.
            Each 100bps adds 0.2 to k (flattens curve).
        
        Returns
        -------
        pd.DataFrame
            Curve data with columns:
            [date, reference_entity, ticker, currency, seniority, tenor_years,
             cds_spread_bps, cds_5y_spread_bps, curve_k, hazard_rate_approx, 
             pd_to_tenor_approx]
        
        Examples
        --------
        Resulting curve shapes (1Y as % of 5Y):
        - k = 0.30 → 1Y = 33% of 5Y (3.0x ratio)
        - k = 0.40 → 1Y = 38% of 5Y (2.6x ratio)
        - k = 0.50 → 1Y = 43% of 5Y (2.3x ratio)
        - k = 0.70 → 1Y = 51% of 5Y (2.0x ratio)
        
        With defaults (base_k=0.30, spread_sensitivity=0.002):
        - 45 bps spread → k = 0.39 → 1Y = 37% of 5Y (2.7x)
        - 75 bps spread → k = 0.45 → 1Y = 41% of 5Y (2.5x)
        - 100 bps spread → k = 0.50 → 1Y = 43% of 5Y (2.3x)
        """
        if self._time_series is None:
            raise ValueError("No time series generated. Call generate_time_series() first.")
        
        as_of_date = pd.to_datetime(as_of)
        max_tenor = max(tenors)
        
        # Get terminal 5Y spreads from time series
        terminal_spreads = (
            self._time_series
            .loc[self._time_series["date"] == self._time_series["date"].max()]
            .set_index("ticker")["cds_5y_spread_bps"]
            .to_dict()
        )
        
        # Build curve for each entity (shape depends on spread level)
        records = []
        for entity in self.entities:
            spread_5y = terminal_spreads[entity.ticker]
            
            # Spread-dependent shape: higher spread → higher k → flatter curve
            k = base_k + spread_sensitivity * spread_5y
            k = np.clip(k, 0.15, 1.0)  # bounds for sanity
            
            # Normalisation factor for 5Y = 100%
            norm = 1 - np.exp(-k * max_tenor)
            
            for tenor in tenors:
                # Exponential saturation: steep rise then flattening
                # spread(T) = spread_5Y * (1 - exp(-k*T)) / (1 - exp(-k*5))
                ratio = (1 - np.exp(-k * tenor)) / norm
                spread = spread_5y * ratio
                
                # Approximate hazard rate: lambda ≈ spread / (1 - R)
                hazard_rate = (spread / 10000) / (1 - recovery_rate)
                
                # Approximate cumulative PD: PD(T) = 1 - exp(-lambda * T)
                pd_to_tenor = 1 - np.exp(-hazard_rate * tenor)
                
                records.append({
                    "date": as_of_date,
                    "reference_entity": entity.name,
                    "ticker": entity.ticker,
                    "currency": "USD",
                    "seniority": "Senior Unsecured",
                    "tenor_years": tenor,
                    "cds_spread_bps": round(spread, 1),
                    "cds_5y_spread_bps": round(spread_5y, 1),
                    "curve_k": round(k, 3),
                    "hazard_rate_approx": hazard_rate,
                    "pd_to_tenor_approx": pd_to_tenor
                })
        
        return pd.DataFrame(records)
    
    def get_realised_correlation(self) -> Optional[np.ndarray]:
        """
        Get realised correlation matrix from generated spread changes.
        
        Returns
        -------
        np.ndarray or None
            Realised correlation matrix, or None if not yet generated
        """
        return self._realised_correlation
    
    def get_correlation_comparison(self) -> Optional[pd.DataFrame]:
        """
        Compare target vs realised correlation.
        
        Returns
        -------
        pd.DataFrame or None
            Comparison table with target, realised, and difference
        """
        if self._realised_correlation is None or self.correlation_matrix is None:
            return None
        
        tickers = [e.ticker for e in self.entities]
        n = len(tickers)
        
        records = []
        for i in range(n):
            for j in range(i+1, n):
                records.append({
                    "pair": f"{tickers[i]}-{tickers[j]}",
                    "target": self.correlation_matrix[i, j],
                    "realised": self._realised_correlation[i, j],
                    "diff": self._realised_correlation[i, j] - self.correlation_matrix[i, j]
                })
        
        return pd.DataFrame(records)
    
    def save_time_series(self, path: str) -> None:
        """Save time series to CSV."""
        if self._time_series is None:
            raise ValueError("No time series to save. Call generate_time_series() first.")
        self._time_series.to_csv(path, index=False)
    
    def save_curve(self, df: pd.DataFrame, path: str) -> None:
        """Save curve snapshot to CSV."""
        df.to_csv(path, index=False)
