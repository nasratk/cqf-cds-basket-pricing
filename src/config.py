"""
Configuration for CDS pricing project.
"""

# =============================================================================
# Entity Definitions
# =============================================================================
TICKERS = {
    'real': {
        'low': {
            "JPMorgan Chase & Co.": "JPM",
            "Exxon Mobil": "XOM",
            "AT&T": "T",
            "Boeing": "BA",
            "IBM": "IBM"
        },
        'high': {
            "JPMorgan Chase & Co.": "JPM",
            "Bank of America": "BAC",
            "Citigroup": "C",
            "Goldman Sachs": "GS",
            "Morgan Stanley": "MS"
        }
    },
    'synthetic': {    
        'low': {
            "Alpha Bank Corp": "ALPHA",
            "Beta Energy Inc": "BETA",
            "Gamma Tech Ltd": "GAMMA",
            "Delta Industrial Co": "DELTA",
            "Epsilon Telecom": "EPSILON"
        },
        'high': {
            "Alpha Bank Corp": "ALPHA",
            "Beta Energy Inc": "BETA",
            "Gamma Tech Ltd": "GAMMA",
            "Delta Industrial Co": "DELTA",
            "Epsilon Telecom": "EPSILON"
        }
    }
}

# =============================================================================
# Data Paths
# =============================================================================
DATA_PATHS = {
    'real': {
        'historical': 'data/real/equity_prices.csv',
        'current': 'data/real/synthetic_cds_curves.csv'
    },
    'synthetic': {
        'historical': 'data/synthetic/synthetic_cds_5y_monthly.csv',
        'current': 'data/synthetic/synthetic_cds_curve.csv'
    }
}

# =============================================================================
# Field Mappings
# =============================================================================
FIELD_MAP = {
    'real': {
        'delta': 'log_return',
        'level': 'adjusted_close'  # Using equity prices as proxy for CDS spreads
    },
    'synthetic': {
        'delta': 'cds_delta',
        'level': 'cds_5y_spread_bps'
    }
}

# =============================================================================
# Pricing Defaults
# =============================================================================
DEFAULT_RECOVERY_RATE = 0.4   # Industry standard
DEFAULT_TERM = 5.0            # Standard CDS tenor (years)
DEFAULT_N_SIMS = 100_000      # Monte Carlo paths

# =============================================================================
# Random Number Generation
# =============================================================================
DEFAULT_SEED = 42
