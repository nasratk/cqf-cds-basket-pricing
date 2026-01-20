# Synthetic CDS Data Generation

This folder contains the data generator module and synthetic CDS data for basket CDS pricing analysis.

## Files

| File | Description |
|------|-------------|
| `generator.py` | Data generation module |
| `data_generation.ipynb` | Notebook to configure and generate data |
| `synthetic_cds_5y_monthly.csv` | Generated 5Y CDS spread time series |
| `synthetic_cds_curve.csv` | Generated CDS term structure snapshot |

## Models

### Time Series: Log-OU Process

Spreads evolve as an Ornstein-Uhlenbeck process on log-spreads:

$$d(\log S) = \kappa(\log \theta - \log S)\,dt + \sigma\sqrt{dt}\,dW$$

**Properties:**
- Positive spreads guaranteed (log-normal)
- Mean-reverting to base level $\theta$
- Percentage-based volatility $\sigma$
- Correlated across entities via Cholesky decomposition

**Regime switching:**
- Volatility multiplier (stress periods have higher vol)
- Spread shift (mean level increases during stress)
- Correlation multiplier (correlations increase toward 1 during stress)

### Term Structure: Exponential Saturation

CDS curve anchored to terminal 5Y spread:

$$S(T) = S_{5Y} \times \frac{1 - e^{-kT}}{1 - e^{-5k}}$$

**Properties:**
- Steep rise at short tenors, flattening toward 5Y
- Decreasing deltas between successive tenors
- $S_{5Y}$ typically 2-3× of $S_{1Y}$

**Spread-dependent shape:**

$$k = k_{\text{base}} + \lambda \times S_{5Y}$$

Where $\lambda$ is the spread sensitivity. This gives:
- Lower $k$ → steeper curve (more curvature)
- Higher $k$ → flatter curve

## Parameters

### Entity Configuration (`EntityConfig`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `name` | Full entity name | "Alpha Bank Corp" |
| `ticker` | Short identifier | "ALPHA" |
| `sector` | Industry sector | "financials" |
| `base_spread_5y` | Long-run mean 5Y spread (bps) | 75 |
| `volatility_pct` | Annualised volatility (decimal) | 0.35 (35%) |
| `mean_reversion_speed` | $\kappa$, annualised | 1.0 (half-life ~8 months) |

### Regime Configuration (`Regime`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `name` | Regime identifier | "covid_stress" |
| `start` | Start date (YYYY-MM) | "2020-03" |
| `end` | End date (YYYY-MM) | "2020-06" |
| `vol_multiplier` | Volatility scaling | 2.5 (2.5× normal vol) |
| `spread_shift_pct` | Mean level shift | 0.8 (80% higher) |
| `correlation_multiplier` | Correlation stress | 1.5 (pushes toward 1) |

### Curve Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_k` | Base steepness $k_{\text{base}}$ | 0.30 |
| `spread_sensitivity` | Spread effect $\lambda$ | 0.002 |
| `recovery_rate` | Assumed recovery $R$ | 0.40 |

**Resulting shapes (approximate):**

| $k$ | $S_{1Y}$ as % of $S_{5Y}$ | Ratio $S_{5Y}/S_{1Y}$ |
|-----|---------------------------|------------------------|
| 0.30 | 33% | 3.0× |
| 0.40 | 38% | 2.6× |
| 0.50 | 43% | 2.3× |

## Output Formats

### `synthetic_cds_5y_monthly.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Month-end date |
| `reference_entity` | string | Full entity name |
| `ticker` | string | Short identifier |
| `sector` | string | Industry sector |
| `cds_5y_spread_bps` | float | 5Y CDS spread in basis points |

### `synthetic_cds_curve.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Snapshot date |
| `reference_entity` | string | Full entity name |
| `ticker` | string | Short identifier |
| `currency` | string | Always "USD" |
| `seniority` | string | Always "Senior Unsecured" |
| `tenor_years` | int | Tenor (1, 2, 3, 4, 5) |
| `cds_spread_bps` | float | CDS spread at tenor |
| `cds_5y_spread_bps` | float | 5Y spread (anchor) |
| `curve_k` | float | Shape parameter used |
| `hazard_rate_approx` | float | $\lambda \approx S / (1-R)$ |
| `pd_to_tenor_approx` | float | $PD(T) \approx 1 - e^{-\lambda T}$ |

## Usage

```python
from generator import CDSDataGenerator, EntityConfig, Regime

# Initialize
gen = CDSDataGenerator(seed=42)

# Configure entities
entities = [
    EntityConfig(
        name="Alpha Bank Corp",
        ticker="ALPHA",
        sector="financials",
        base_spread_5y=75,
        volatility_pct=0.35,
        mean_reversion_speed=1.0
    ),
    # ... more entities
]
gen.set_entities(entities)

# Set correlation matrix
gen.set_correlation_matrix(correlation_matrix)  # n×n numpy array

# Set regimes
regimes = [
    Regime("normal", "2016-01", "2020-02", 1.0, 0.0, 1.0),
    Regime("stress", "2020-03", "2020-06", 2.5, 0.8, 1.5),
    # ... more regimes
]
gen.set_regimes(regimes)

# Generate
ts_df = gen.generate_time_series(start="2016-01-31", end="2025-12-31", freq="M")
curve_df = gen.generate_curve_snapshot(
    as_of="2025-12-31",
    tenors=[1, 2, 3, 4, 5],
    recovery_rate=0.40,
    base_k=0.30,
    spread_sensitivity=0.002
)

# Save
gen.save_time_series("synthetic_cds_5y_monthly.csv")
gen.save_curve(curve_df, "synthetic_cds_curve.csv")

# Validation
print(gen.get_correlation_comparison())  # Target vs realised correlation
```

## Regenerating Data

1. Open `data_generation.ipynb`
2. Modify parameters in Section 1 as needed
3. Run all cells
4. New CSVs will overwrite existing files

To create a new dataset with different seed:
```python
SEED = 123  # Change this
```

## Current Dataset

Generated with `SEED=42`:
- **Period:** 2016-01 to 2025-12 (monthly)
- **Entities:** ALPHA, BETA, GAMMA, DELTA, EPSILON
- **Regimes:** Pre-COVID, COVID stress, recovery, post-COVID, banking stress, final
- **Curve as-of:** 2025-12-31
