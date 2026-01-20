"""
Copula models for dependence structure.
"""
import numpy as np
from scipy.stats import norm, t, spearmanr
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

from .rng import RNGEngine


@dataclass
class TCopulaFitResult:
    """Results from t-copula MLE fitting."""
    df: float                    # Optimal degrees of freedom
    log_likelihood: float        # Log-likelihood at optimum
    df_bounds: tuple             # Search bounds used
    success: bool                # Optimization converged
    n_obs: int                   # Number of observations
    n_dims: int                  # Number of dimensions


def _multivariate_t_logpdf(X: np.ndarray, sigma: np.ndarray, df: float) -> np.ndarray:
    """
    Log-density of multivariate Student-t distribution.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (n, d) observations in t-space
    sigma : np.ndarray
        Shape (d, d) correlation matrix
    df : float
        Degrees of freedom
        
    Returns
    -------
    np.ndarray
        Shape (n,) log-density values
    """
    n, d = X.shape
    
    # Cholesky for stable inverse and determinant
    L = np.linalg.cholesky(sigma)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    
    # Mahalanobis distance: x' Σ^{-1} x
    # Solve L @ z = x.T, then mahal = sum(z^2)
    Z = np.linalg.solve(L, X.T)  # (d, n)
    mahal = np.sum(Z**2, axis=0)  # (n,)
    
    # Log-density components
    log_norm = (
        gammaln((df + d) / 2)
        - gammaln(df / 2)
        - (d / 2) * np.log(df * np.pi)
        - 0.5 * log_det
    )
    log_kernel = -((df + d) / 2) * np.log(1 + mahal / df)
    
    return log_norm + log_kernel


def _t_copula_loglikelihood(
    df: float, 
    U: np.ndarray, 
    sigma: np.ndarray
) -> float:
    """
    Log-likelihood of t-copula.
    
    L(df) = Σ_i [ log f_{df,Σ}(T_i) - Σ_j log f_{df}(T_{ij}) ]
    
    where T_i = t_df^{-1}(U_i) are the t-transformed observations.
    
    Parameters
    ----------
    df : float
        Degrees of freedom (must be > 2)
    U : np.ndarray
        Shape (n, d) uniform observations in (0, 1)
    sigma : np.ndarray
        Shape (d, d) correlation matrix
        
    Returns
    -------
    float
        Total log-likelihood
    """
    # Transform uniforms to t-space
    T = t.ppf(U, df=df)
    
    # Multivariate t log-density (joint)
    log_joint = _multivariate_t_logpdf(T, sigma, df)
    
    # Univariate t log-densities (marginals)
    log_marginals = t.logpdf(T, df=df).sum(axis=1)
    
    # Copula log-likelihood = joint - marginals
    log_lik = np.sum(log_joint - log_marginals)
    
    return log_lik


def _estimate_correlation_matrix(U: np.ndarray) -> np.ndarray:
    """
    Estimate correlation matrix from uniform observations using rank correlation.
    
    Works for all elliptical copulas (Gaussian, t).
    
    Method: Spearman's rho -> convert to Pearson
        rho = 2 * sin(pi * rho_S / 6)
    
    Parameters
    ----------
    U : np.ndarray
        Shape (n_obs, n_dims) of uniform marginals in (0, 1)
        
    Returns
    -------
    np.ndarray
        Estimated correlation matrix
    """
    rho_S, _ = spearmanr(U)
    
    # Handle single dimension case
    if U.shape[1] == 2:
        # spearmanr returns scalar for 2D
        rho_S = np.array([[1.0, rho_S], [rho_S, 1.0]])
    
    # Convert Spearman to Pearson
    sigma = 2 * np.sin(np.pi * rho_S / 6)
    
    return sigma


class BaseCopula(ABC):
    """Abstract base class for copula models."""
    
    @abstractmethod
    def fit(self, U: np.ndarray) -> 'BaseCopula':
        """Calibrate copula from uniform observations."""
        pass
    
    @abstractmethod
    def simulate(self, n_sims: int, rng: RNGEngine = None) -> np.ndarray:
        """Generate correlated uniform samples."""
        pass


class GaussianCopula(BaseCopula):
    """
    Gaussian copula for modelling joint default dependence.
    
    Calibration: U → Z = Φ⁻¹(U) → Σ = corr(Z)
    Simulation:  Z ~ N(0, Σ) → U = Φ(Z)
    """
    
    def __init__(self, correlation_matrix: np.ndarray = None):
        """
        Parameters
        ----------
        correlation_matrix : np.ndarray, optional
            If provided, use this correlation matrix directly (skip calibration).
            If None, must call fit() before simulate().
        """
        self.sigma = correlation_matrix
        self._cholesky = None
        self._fitted = correlation_matrix is not None
        
        if self._fitted:
            self.n_dims = self.sigma.shape[0]
    
    def fit(self, U: np.ndarray) -> 'GaussianCopula':
        """
        Calibrate copula from uniform observations.
        
        Uses rank correlation (Spearman -> Pearson conversion).
        
        Parameters
        ----------
        U : np.ndarray
            Shape (n_obs, n_dims) of uniform marginals in (0, 1)
            
        Returns
        -------
        self
        """
        U = np.asarray(U)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        
        # Estimate correlation matrix via rank correlation
        self.sigma = _estimate_correlation_matrix(U)
        self.n_dims = self.sigma.shape[0]
        
        # Pre-compute Cholesky decomposition
        self._cholesky = np.linalg.cholesky(self.sigma)
        self._fitted = True
        
        return self
    
    @property
    def cholesky(self) -> np.ndarray:
        """Cholesky decomposition of correlation matrix."""
        if not self._fitted:
            raise ValueError("Copula not fitted. Call fit() first.")
        if self._cholesky is None:
            self._cholesky = np.linalg.cholesky(self.sigma)
        return self._cholesky
    
    @property
    def correlation_matrix(self) -> np.ndarray:
        """Calibrated correlation matrix."""
        if not self._fitted:
            raise ValueError("Copula not fitted. Call fit() first.")
        return self.sigma
    
    def simulate(self, n_sims: int, rng: RNGEngine = None) -> np.ndarray:
        """
        Generate correlated uniform samples via Gaussian copula.
        
        Parameters
        ----------
        n_sims : int
            Number of simulation paths
        rng : RNGEngine, optional
            Random number generator. If None, uses default pseudo-random.
            
        Returns
        -------
        np.ndarray
            Shape (n_sims, n_dims) of correlated uniforms in (0, 1)
        """
        if not self._fitted:
            raise ValueError("Copula not fitted. Call fit() first.")
        
        if rng is None:
            rng = RNGEngine()
        
        # Generate independent standard normals
        Z_indep = rng.normal(n_sims, self.n_dims)
        
        # Apply Cholesky to induce correlation
        Z_corr = Z_indep @ self.cholesky.T
        
        # Transform to uniform via standard normal CDF
        U = norm.cdf(Z_corr)
        
        return U


class TCopula(BaseCopula):
    """
    Student-t copula for modeling joint default dependence with tail dependence.
    
    Calibration:
        - Σ estimated via rank correlation (Spearman → Pearson)
        - ν (df) estimated via profile MLE, or user-specified
    
    Simulation:  Z ~ N(0,Σ), W ~ χ²(ν), T = Z/√(W/ν), U = t_cdf(T)
    
    Key difference from Gaussian: heavier tails = more joint extreme events.
    
    Attributes
    ----------
    df : float
        Degrees of freedom (after fitting)
    sigma : np.ndarray
        Correlation matrix (after fitting)
    fit_result_ : TCopulaFitResult
        Detailed MLE results (only if df was estimated)
    """
    
    # Default bounds for df optimization
    DF_BOUNDS = (2.01, 50.0)
    
    def __init__(self, df: float = None, correlation_matrix: np.ndarray = None):
        """
        Parameters
        ----------
        df : float, optional
            Degrees of freedom (lower = heavier tails). 
            If None, will be estimated via MLE in fit().
        correlation_matrix : np.ndarray, optional
            If provided with df, skip calibration.
        """
        self.df = df
        self.sigma = correlation_matrix
        self._cholesky = None
        self._fitted = (df is not None) and (correlation_matrix is not None)
        self._U_fit = None  # Store data for log_likelihood() calls
        self.fit_result_: Optional[TCopulaFitResult] = None
        
        if self._fitted:
            self.n_dims = self.sigma.shape[0]
    
    @property
    def cholesky(self) -> np.ndarray:
        """Cholesky decomposition of correlation matrix."""
        if not self._fitted:
            raise ValueError("Copula not fitted. Call fit() first.")
        if self._cholesky is None:
            self._cholesky = np.linalg.cholesky(self.sigma)
        return self._cholesky
    
    @property
    def correlation_matrix(self) -> np.ndarray:
        """Calibrated correlation matrix."""
        if not self._fitted:
            raise ValueError("Copula not fitted. Call fit() first.")
        return self.sigma
    
    def log_likelihood(self, df: float, U: np.ndarray = None) -> float:
        """
        Compute t-copula log-likelihood at given df.
        
        Useful for plotting the likelihood surface.
        
        Parameters
        ----------
        df : float
            Degrees of freedom to evaluate
        U : np.ndarray, optional
            Uniform observations. If None, uses data from fit().
            
        Returns
        -------
        float
            Log-likelihood value
        """
        if U is None:
            if self._U_fit is None:
                raise ValueError("No data available. Provide U or call fit() first.")
            U = self._U_fit
        
        if self.sigma is None:
            raise ValueError("Correlation matrix not set. Call fit() first.")
        
        return _t_copula_loglikelihood(df, U, self.sigma)
    
    def fit(
        self, 
        U: np.ndarray, 
        df_bounds: tuple = None,
        estimate_df: bool = None
    ) -> 'TCopula':
        """
        Calibrate t-copula from uniform observations.
        
        Parameters
        ----------
        U : np.ndarray
            Shape (n_obs, n_dims) of uniform marginals in (0, 1)
        df_bounds : tuple, optional
            (lower, upper) bounds for df optimization. Default (2.01, 50).
        estimate_df : bool, optional
            If True, estimate df via MLE even if df was provided.
            If False, use provided df (error if None).
            If None (default), estimate only if self.df is None.
            
        Returns
        -------
        self
        """
        U = np.asarray(U)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        
        # Store for later log_likelihood() calls
        self._U_fit = U.copy()
        
        # Step 1: Estimate correlation matrix via rank correlation
        self.sigma = _estimate_correlation_matrix(U)
        self.n_dims = self.sigma.shape[0]
        
        # Step 2: Determine whether to estimate df
        if estimate_df is None:
            estimate_df = (self.df is None)
        
        if estimate_df:
            # Profile MLE for df
            bounds = df_bounds or self.DF_BOUNDS
            
            # Minimize negative log-likelihood
            def neg_ll(df):
                return -_t_copula_loglikelihood(df, U, self.sigma)
            
            result = minimize_scalar(
                neg_ll,
                bounds=bounds,
                method='bounded'
            )
            
            self.df = result.x
            
            # Store fit results
            self.fit_result_ = TCopulaFitResult(
                df=result.x,
                log_likelihood=-result.fun,
                df_bounds=bounds,
                success=result.success,
                n_obs=U.shape[0],
                n_dims=U.shape[1]
            )
        
        elif self.df is None:
            raise ValueError(
                "Degrees of freedom (df) must be provided at initialization "
                "or set estimate_df=True.\n"
                "Example: TCopula(df=5).fit(U) or TCopula().fit(U, estimate_df=True)"
            )
        
        # Pre-compute Cholesky decomposition
        self._cholesky = np.linalg.cholesky(self.sigma)
        self._fitted = True
        
        return self
    
    def simulate(self, n_sims: int, rng: RNGEngine = None) -> np.ndarray:
        """
        Generate correlated uniform samples via t-copula.
        
        Parameters
        ----------
        n_sims : int
            Number of simulation paths
        rng : RNGEngine, optional
            Random number generator. If None, uses default pseudo-random.
            
        Returns
        -------
        np.ndarray
            Shape (n_sims, n_dims) of correlated uniforms in (0, 1)
        """
        if not self._fitted:
            raise ValueError("Copula not fitted. Call fit() first.")
        
        if rng is None:
            rng = RNGEngine()
        
        # 1. Generate correlated normals (same as Gaussian copula)
        Z_indep = rng.normal(n_sims, self.n_dims)
        Z_corr = Z_indep @ self.cholesky.T
        
        # 2. Generate chi-square for t-distribution scaling
        W = rng.chisquare(n_sims, df=self.df)
        
        # 3. Construct multivariate t: T = Z / sqrt(W/df)
        T = Z_corr / np.sqrt(W / self.df).reshape(-1, 1)
        
        # 4. Transform to uniform via t CDF
        U = t.cdf(T, df=self.df)
        
        return U
