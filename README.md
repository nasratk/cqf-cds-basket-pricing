# CQF Final Project: k-th to Default Basket CDS Pricing

## Overview

This project implements a Monte Carlo simulation framework for pricing k-th to default basket Credit Default Swaps (CDS) using copula models. The implementation compares Gaussian and Student-t copulas for modelling default dependence, with support for multiple random number generation methods including quasi-random sequences (Sobol, Halton) and variance reduction techniques.

The methodology and results are documented in `REPORT.md` (and `REPORT.html` for formatted viewing).

---

## Project Structure

```
cqf-cds-pricing/
├── src/                              # Core implementation modules
│   ├── __init__.py                   # Package exports
│   ├── config.py                     # Configuration and defaults
│   ├── bootstrapper.py               # Hazard rate bootstrapping
│   ├── copula.py                     # Gaussian and t-copula implementations
│   ├── rng.py                        # Random number generation
│   ├── pricer.py                     # Basket CDS pricing logic
│   └── utils.py                      # Helper functions
│
├── data/
│   ├── synthetic/                    # Generated test data
│   │   ├── historical_spreads.csv
│   │   └── current_curves.csv
│   ├── real/                         # Real market data
│   └── README_Data.md                # Data documentation
│
├── output/
│   ├── runs/                         # Executed notebooks from orchestration
│   │   ├── sensitivity/              # Sensitivity analysis runs
│   │   └── pricing/                  # Copula comparison runs
│   ├── convergence/                  # Convergence analysis runs
│   ├── figs_for_report/              # Figures used in report
│   ├── tables_for_report/            # CSV tables for report
│   └── discard_figs/                 # Temporary figures (not for report)
│
├── cds_pricing.ipynb                 # Core pricing notebook (parameterised)
├── 1_convergence_analysis.ipynb      # RNG convergence study
├── 2_pricing_orchestration.ipynb     # Copula comparison runner
├── 3_sensitivity_orchestration.ipynb # Sensitivity analysis runner
│
├── REPORT.md                         # Full methodology and results
├── REPORT.html                       # Formatted report (view in browser)
├── style.css                         # Report styling
└── README.md                         # This file
```

---

## Notebooks

The project uses a parameterised notebook workflow. The core pricing logic lives in `cds_pricing.ipynb`, which is executed programmatically by orchestration notebooks using `papermill`. Results are collected using `scrapbook`.

### Notebook Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         cds_pricing.ipynb                               │
│                    (Core parameterised notebook)                        │
│  • Loads data, calibrates copula, simulates defaults, computes spread   │
│  • Tagged 'parameters' cell allows injection via papermill              │
│  • Glues results (fair_spread, se_bps, etc.) via scrapbook              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ convergence_        │  │ pricing_            │  │ sensitivity_        │
│ analysis.ipynb      │  │ orchestration.ipynb │  │ orchestration.ipynb │
│                     │  │                     │  │                     │
│ Varies: N_SIMS,     │  │ Varies: COPULA      │  │ Varies: K, CORR,    │
│ RNG_METHOD          │  │ (gaussian vs t)     │  │ DF_OVERRIDE, R      │
│                     │  │                     │  │                     │
│ Output: convergence │  │ Output: copula      │  │ Output: sensitivity │
│ plots, SE analysis  │  │ comparison charts   │  │ charts and tables   │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### `cds_pricing.ipynb`

**Purpose:** Core end-to-end pricing notebook. Designed to be run standalone or executed programmatically with parameter injection.

**Sections:**
1. Setup & Imports
2. Parameters (papermill-injectable via tagged cell)
3. Data Loading (equity prices, CDS term structures)
4. Exploratory Data Analysis
5. Copula Calibration (correlation matrix, t-copula df estimation)
6. Hazard Rate Bootstrapping
7. Monte Carlo Simulation
8. CDS Pricing
9. Results Summary

**Key Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DATA_MODE` | `'synthetic'` or `'real'` | `'real'` |
| `SPREAD_MEASURE` | `'delta'` or `'level'` for correlation estimation | `'delta'` |
| `K` | k-th to default (1 to 5) | `5` |
| `COPULA` | `'gaussian'` or `'t'` | `'t'` |
| `RNG_METHOD` | `'pseudo'`, `'antithetic'`, `'halton'`, `'sobol'` | `'sobol'` |
| `N_SIMS` | Number of simulations | `100000` |
| `CORR_LEVEL` | `'high'` or `'low'` correlation regime | `'high'` |
| `RECOVERY_RATE` | Recovery rate assumption | `0.4` |
| `DF_OVERRIDE` | Override t-copula df (None = estimate via MLE) | `None` |
| `FIG_OUTPUT_DIR` | Subdirectory for figures | `'figs_for_report'` |

**Scrapbook outputs** (glued for programmatic retrieval):
- `fair_spread` — Fair spread in basis points
- `se_bps` — Standard error in basis points
- `n_defaults_in_term` — Count of simulations with k-th default before maturity
- `kth_default_times` — Array of k-th default times

---

### `convergence_analysis.ipynb`

**Purpose:** Study convergence of fair spread estimates vs number of simulations. Compares RNG methods.

**Methodology reference:** Section 4.1 of REPORT.md

**Parameters swept:**
- `RNG_METHOD`: pseudo, antithetic, halton, sobol
- `N_SIMS`: 1,000 to 250,000

**Fixed:** K=3, COPULA='t', CORR_LEVEL='high'

**Outputs:**
- `output/convergence/*.ipynb` — Individual executed notebooks
- `output/figs_for_report/convergence_plot.png` — Spread vs N
- `output/figs_for_report/se_convergence_plot.png` — SE convergence (log-log)
- `output/figs_for_report/convergence_results.csv` — Results table

---

### `pricing_orchestration.ipynb`

**Purpose:** Compare Gaussian and t-copula pricing results. Generates the main copula comparison charts.

**Methodology reference:** Section 4.4 of REPORT.md

**Parameters swept:**
- `COPULA`: gaussian, t

**Fixed:** K=5, N_SIMS=100,000, RNG_METHOD='sobol', CORR_LEVEL='high'

**Outputs:**
- `output/runs/pricing/*.ipynb` — Executed notebooks
- `output/figs_for_report/spread_comparison.png` — Bar chart comparing spreads
- `output/figs_for_report/default_time_comparison.png` — Overlapping histograms
- `output/figs_for_report/default_time_ecdf_comparison.png` — ECDF comparison

---

### `sensitivity_orchestration.ipynb`

**Purpose:** Run sensitivity analyses on key model parameters.

**Methodology reference:** Section 4.5 of REPORT.md

**Analyses performed:**

| Sensitivity | Parameter | Values |
|-------------|-----------|--------|
| Tranche seniority | `K` | 1, 2, 3, 4, 5 |
| Correlation level | `CORR_LEVEL` | 'high', 'low' |
| Degrees of freedom | `DF_OVERRIDE` | 3, 3.9, 6, 10, 15, 30, plus Gaussian |
| Recovery rate | `RECOVERY_RATE` | 0%, 30%, 40%, 50% |

**Base case:** K=5, COPULA='t', CORR_LEVEL='high', DF_OVERRIDE=None (MLE), RECOVERY_RATE=0.4

**Outputs:**
- `output/runs/sensitivity/*.ipynb` — Individual executed notebooks
- `output/figs_for_report/sensitivity_k.png`
- `output/figs_for_report/sensitivity_correlation_times.png`
- `output/figs_for_report/sensitivity_correlation_spread.png`
- `output/figs_for_report/sensitivity_df.png`
- `output/figs_for_report/sensitivity_recovery.png`
- `output/tables_for_report/sensitivity_*.csv` — Data tables

---

## Modules

### `src/bootstrapper.py`

**Class:** `HazardRateBootstrapper`

Derives piecewise-constant hazard rates from CDS spread term structures using the approximation $h = s / (1-R)$.

**Methodology reference:** Section 2.2 of REPORT.md

```python
from src import HazardRateBootstrapper

bootstrapper = HazardRateBootstrapper(tenors, spreads, recovery_rate=0.4)
hazard_rates = bootstrapper.hazard_rates
survival_probs = bootstrapper.survival_probability(times)
```

### `src/copula.py`

**Classes:** `GaussianCopula`, `TCopula`

Copula fitting and simulation. Both use rank correlation (Spearman → Pearson conversion) for correlation matrix estimation. The t-copula estimates degrees of freedom via profile maximum likelihood.

**Methodology reference:** Section 2.3 of REPORT.md

```python
from src import GaussianCopula, TCopula

# Gaussian copula
copula = GaussianCopula().fit(uniform_data)
samples = copula.simulate(n_sims, rng=rng_engine)

# t-copula (df estimated via MLE)
copula = TCopula().fit(uniform_data)
print(f"Estimated df: {copula.df:.2f}")

# t-copula with specified df (for sensitivity analysis)
copula = TCopula(df=6).fit(uniform_data)
```

### `src/rng.py`

**Class:** `RNGEngine`

Unified interface for random number generation supporting multiple methods.

**Methodology reference:** Section 2.5 of REPORT.md

| Method | Description |
|--------|-------------|
| `'pseudo'` | Standard pseudo-random (NumPy Mersenne Twister) |
| `'antithetic'` | Antithetic variates for variance reduction |
| `'halton'` | Halton quasi-random sequence (scrambled) |
| `'sobol'` | Sobol quasi-random sequence (scrambled) |

```python
from src import RNGEngine

rng = RNGEngine(method='sobol', seed=42)
uniforms = rng.uniform(n_samples, n_dims)
normals = rng.normal(n_samples, n_dims)
```

### `src/pricer.py`

**Class:** `BasketCDSPricer`

Computes fair spread from simulated default times. The fair spread equates the expected protection leg to the expected premium leg.

**Methodology reference:** Section 2.6 of REPORT.md

```python
from src import BasketCDSPricer

pricer = BasketCDSPricer(
    kth_default_times=kth_default_times,
    term=5,
    recovery_rate=0.4
)
result = pricer.price()
print(f"Fair spread: {result['fair_spread_bps']:.2f} bps")
print(f"Standard error: {result['se_bps']:.2f} bps")
```

### `src/utils.py`

Helper functions for data transformation and visualisation:
- `rank_to_uniform()` — Transform data to uniform marginals via empirical CDF
- `spearman_to_pearson()` — Convert Spearman ρ to Pearson ρ for elliptical copulas
- `time_to_default()` — Invert uniform to default time given hazard rates

---

## Code-Methodology Mapping

The table below maps methodology sections in `REPORT.md` to their implementation.

| Methodology (REPORT.md) | Implementation |
|-------------------------|----------------|
| §2.2 Hazard Rate Bootstrapping | `src/bootstrapper.py` |
| §2.3 Gaussian Copula | `src/copula.py` → `GaussianCopula` |
| §2.3 Student-t Copula | `src/copula.py` → `TCopula` |
| §2.3 Correlation Calibration | `src/copula.py` → `fit()` with rank correlation |
| §2.3 Degrees of Freedom MLE | `src/copula.py` → `TCopula.fit()`, `log_likelihood()` |
| §2.4 Monte Carlo Simulation | `src/copula.py` → `simulate()` methods |
| §2.5 Pseudo-random | `src/rng.py` → `RNGEngine(method='pseudo')` |
| §2.5 Quasi-random (Sobol) | `src/rng.py` → `RNGEngine(method='sobol')` |
| §2.5 Quasi-random (Halton) | `src/rng.py` → `RNGEngine(method='halton')` |
| §2.5 Antithetic Variates | `src/rng.py` → `RNGEngine(method='antithetic')` |
| §2.6 Fair Spread Calculation | `src/pricer.py` → `BasketCDSPricer.price()` |
| §2.7 Implementation Table | See table in report |

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: numpy, pandas, scipy, matplotlib, papermill, scrapbook

### Quick Start

1. **Single pricing run:**
   Open `cds_pricing.ipynb` in Jupyter, adjust parameters in the tagged cell, run all cells.

2. **Copula comparison:**
   Run `pricing_orchestration.ipynb` to compare Gaussian vs t-copula.

3. **Sensitivity analysis:**
   Run `sensitivity_orchestration.ipynb` to generate all sensitivity charts.

4. **Convergence study:**
   Run `convergence_analysis.ipynb` to analyse RNG method performance.

### Reproducing Report Results

To regenerate all figures and tables used in the report:

```bash
# 1. Convergence analysis (Section 4.1)
jupyter nbconvert --execute convergence_analysis.ipynb

# 2. Copula comparison (Section 4.4)
jupyter nbconvert --execute pricing_orchestration.ipynb

# 3. Sensitivity analysis (Section 4.5)
jupyter nbconvert --execute sensitivity_orchestration.ipynb
```

Outputs will be saved to `output/figs_for_report/` and `output/tables_for_report/`.

---

## Papermill & Scrapbook

The orchestration notebooks use two libraries for automated execution:

**Papermill** executes `cds_pricing.ipynb` with different parameters:
```python
import papermill as pm

pm.execute_notebook(
    'cds_pricing.ipynb',                      # Source notebook (unchanged)
    'output/runs/sensitivity/run_k_5.ipynb',  # Output notebook (new file)
    parameters={'K': 5, 'COPULA': 't'}        # Injected parameters
)
```

**Scrapbook** retrieves results from executed notebooks:
```python
import scrapbook as sb

# In cds_pricing.ipynb — save results to notebook metadata
sb.glue('fair_spread', fair_spread_bps)

# In orchestration notebook — retrieve results
nb = sb.read_notebook('output/runs/sensitivity/run_k_5.ipynb')
spread = nb.scraps['fair_spread'].data
```

This workflow keeps the source notebook clean while enabling systematic parameter sweeps.

---

## Notes

- **Data modes:** Set `DATA_MODE='real'` for actual market data or `'synthetic'` for generated test data.
- **Quasi-random sequences** (Sobol) are recommended for production runs—they achieve target precision with fewer simulations.
- **Antithetic variates** provide no benefit for k-th to default pricing due to the order statistic payoff structure (see Section 4.1 of report).
- The t-copula with MLE-estimated degrees of freedom (ν ≈ 4) produces significantly higher spreads than the Gaussian copula due to tail dependence.
