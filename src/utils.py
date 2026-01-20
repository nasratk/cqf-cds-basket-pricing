"""
Utility functions for data processing and visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def calc_ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate empirical CDF of a 1D array.
    
    Parameters
    ----------
    x : np.ndarray or pd.Series
        Input data (NaN values are dropped)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (sorted values, CDF values)
    """
    if hasattr(x, 'dropna'):
        x = x.dropna()
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    
    n = len(x)
    x_sorted = np.sort(x)
    y_vals = np.arange(1, n + 1) / (n + 1)  # Avoid reaching exactly 0 or 1
    return x_sorted, y_vals


def plot_scatter_matrix(df: pd.DataFrame, title: str = None, color: str = 'darkgreen'):
    """
    Plot scatter matrix for pairwise relationships.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    title : str, optional
        Plot title
    color : str
        Color for plots (default 'darkgreen')
    """
    scatter_matrix(
        df,
        figsize=(10, 10),
        diagonal='hist',
        color=color,
        hist_kwds={"color": color}
    )
    if title:
        plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def rank_to_uniform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame columns to uniform marginals via rank transformation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
        
    Returns
    -------
    pd.DataFrame
        Uniform-transformed data in (0, 1)
    """
    from scipy.stats import rankdata
    
    u_df = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        x = df[col]
        mask = x.notna()
        n = mask.sum()
        
        ranks = rankdata(x[mask], method='average')
        u_df.loc[mask, col] = ranks / (n + 1)
    
    return u_df
