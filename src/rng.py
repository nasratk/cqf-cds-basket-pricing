"""
Random number generation for Monte Carlo simulation.

Supports:
- Pseudo-random (standard numpy)
- Pseudo-random with antithetic variates (variance reduction)
- Halton sequence (quasi-random)
- Sobol sequence (quasi-random)
"""
import numpy as np
from scipy.stats import qmc, norm
from typing import Literal

from . import config


Method = Literal['pseudo', 'pseudo_antithetic', 'halton', 'sobol']


class RNGEngine:
    """
    Unified random number generator for Monte Carlo simulation.
    
    Methods
    -------
    pseudo : Standard pseudo-random (numpy)
    pseudo_antithetic : Pseudo-random with antithetic variates
    halton : Halton quasi-random sequence
    sobol : Sobol quasi-random sequence
    
    Note: Antithetic variates only available with pseudo-random.
    Quasi-random sequences already provide balanced coverage.
    """
    
    VALID_METHODS = ('pseudo', 'pseudo_antithetic', 'halton', 'sobol')
    
    def __init__(self, method: Method = 'pseudo', seed: int = None):
        """
        Parameters
        ----------
        method : str
            One of 'pseudo', 'pseudo_antithetic', 'halton', 'sobol'
        seed : int, optional
            Random seed. Defaults to config.DEFAULT_SEED
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}")
        
        self.method = method
        self.seed = seed if seed is not None else config.DEFAULT_SEED
        self._rng = np.random.default_rng(self.seed)
    
    @property
    def is_antithetic(self) -> bool:
        return self.method == 'pseudo_antithetic'
    
    @property
    def is_quasi(self) -> bool:
        return self.method in ('halton', 'sobol')
    
    def uniform(self, n: int, d: int) -> np.ndarray:
        """
        Generate uniform random numbers in (0, 1).
        
        Parameters
        ----------
        n : int
            Number of samples
        d : int
            Number of dimensions
            
        Returns
        -------
        np.ndarray
            Shape (n, d) of uniforms in (0, 1)
        """
        if self.method == 'pseudo':
            return self._rng.uniform(size=(n, d))
        
        elif self.method == 'pseudo_antithetic':
            n_half = n // 2
            U_half = self._rng.uniform(size=(n_half, d))
            # Interleave pairs: [U0, 1-U0, U1, 1-U1, ...]
            U = np.empty((n_half * 2, d))
            U[0::2] = U_half
            U[1::2] = 1 - U_half
            return U[:n]  # Handle odd n
        
        elif self.method == 'halton':
            sampler = qmc.Halton(d=d, scramble=True, seed=self.seed)
            return sampler.random(n)
        
        elif self.method == 'sobol':
            sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
            return sampler.random(n)
    
    def normal(self, n: int, d: int) -> np.ndarray:
        """
        Generate standard normal random numbers.
        
        Parameters
        ----------
        n : int
            Number of samples
        d : int
            Number of dimensions
            
        Returns
        -------
        np.ndarray
            Shape (n, d) of standard normals
        """
        if self.method == 'pseudo':
            return self._rng.standard_normal(size=(n, d))
        
        elif self.method == 'pseudo_antithetic':
            n_half = n // 2
            Z_half = self._rng.standard_normal(size=(n_half, d))
            # Interleave pairs: [Z0, -Z0, Z1, -Z1, ...]
            Z = np.empty((n_half * 2, d))
            Z[0::2] = Z_half
            Z[1::2] = -Z_half
            return Z[:n]
        
        else:
            # Quasi-random: transform uniforms via inverse CDF
            U = self.uniform(n, d)
            return norm.ppf(U)
    
    def chisquare(self, n: int, df: float) -> np.ndarray:
        """
        Generate chi-square random numbers.
        
        Used for t-copula simulation.
        
        Parameters
        ----------
        n : int
            Number of samples
        df : float
            Degrees of freedom
            
        Returns
        -------
        np.ndarray
            Shape (n,) of chi-square samples
        """
        if self.method in ('pseudo', 'pseudo_antithetic'):
            # Note: antithetic doesn't apply naturally to chi-square
            # (chi-square is non-negative, no symmetric reflection)
            return self._rng.chisquare(df=df, size=n)
        
        else:
            # Quasi-random: use inverse CDF
            from scipy.stats import chi2
            U = self.uniform(n, 1).flatten()
            return chi2.ppf(U, df=df)
    
    def __repr__(self) -> str:
        return f"RNGEngine(method='{self.method}', seed={self.seed})"
