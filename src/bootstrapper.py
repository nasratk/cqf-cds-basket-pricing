"""
Hazard rate bootstrapping from CDS spreads.
"""
import numpy as np


class BootStrapper:
    """Simple piecewise-constant hazard bootstrap from CDS spreads (approx)."""
    
    def __init__(self, tenors: np.ndarray, spreads: np.ndarray, recovery_rate: float = 0.4):
        """
        Parameters
        ----------
        tenors : np.ndarray
            Tenor points in years (e.g., [1, 2, 3, 5, 7, 10])
        spreads : np.ndarray
            CDS spreads in basis points at each tenor
        recovery_rate : float
            Assumed recovery rate (default 0.4)
        """
        self.tenors = np.asarray(tenors, dtype=float)
        self.spreads = np.asarray(spreads, dtype=float) / 10000  # bps to decimal
        self.R = recovery_rate
        self.lgd = 1 - self.R
    
    def bootstrap(self) -> np.ndarray:
        """
        Perform bootstrapping to get piecewise-constant hazard rates.
        
        Returns
        -------
        np.ndarray
            Hazard rates for each tenor interval
        """
        tenors = self.tenors
        spreads = self.spreads
        lgd = self.lgd
        
        avg_haz = spreads / lgd
        haz = np.zeros(len(tenors))
        haz[0] = spreads[0] / lgd  # Initial hazard rate
        
        for i in range(1, len(tenors)):
            delta_t = tenors[i] - tenors[i - 1]
            hazard_rate = (
                (avg_haz[i] * tenors[i]) - (avg_haz[i-1] * tenors[i - 1])
            ) / delta_t
            haz[i] = hazard_rate

        return haz
