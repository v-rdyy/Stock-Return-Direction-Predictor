# Probabilistic Market Bias Estimation and Risk-Aware Trading System

A quantitative machine learning system that estimates daily directional market bias with calibrated confidence and implements a complete trading system. This project focuses on **probabilistic bias estimation** rather than binary predictions, using calibration, uncertainty quantification, expected return modeling, and a full backtesting framework.

## TL;DR

This project builds a **probabilistic market bias estimation system** with calibrated confidence and uncertainty awareness, then connects it to a **risk-aware trading framework** using expected return modeling, volatility-adjusted position sizing, and realistic backtesting with walk-forward validation.

The focus is **decision-making under uncertainty**, not binary prediction.

## Project Overview

**Goal:** Estimate the probability that tomorrow's stock return will be positive (P(up)) with calibrated confidence, then use this for risk-aware trading decisions.

**Key Points:**
- Probabilistic output: Each day gets P(up) ∈ [0,1] (not just 0 or 1)
- Calibrated probabilities: Probabilities are reliable (0.65 means ~65% historically)
- Uncertainty quantification: Measures prediction confidence through ensemble variance
- Expected return modeling: Predicts return values conditionally on high-confidence days
- Complete trading system: Decision logic, position sizing, backtesting, risk management

## Key Limitations (Read First)

- **Daily data only**: No intraday execution or intraday stop-loss
- **Simplified transaction cost model**: Volatility-scaled approximation; real trading includes bid-ask spreads, market impact, slippage
- **Threshold optimization may overfit**: Grid search used for sensitivity analysis, not as a claim of deployable alpha
- **Kelly-style sizing is heuristic**: Fractional Kelly approximation, not optimal solution
- **Sharpe ratios are for relative comparison**: High values (3-4) reflect single-stock, short-sample, optimized conditions; not deployable performance claims
- **Negative R² is expected**: Return models evaluated conditionally on high-confidence days, not globally

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd firsttrading
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline on multiple stocks:

```bash
python3 src/main.py
```

This executes the full pipeline:
1. Downloads 2 years of data for 12 companies: AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, GS, BAC, WMT, DIS, and CAT
2. Creates features and labels
3. **Phase 1**: Trains probability models, calibrates probabilities, estimates uncertainty
4. **Phase 2**: Trains expected return models (evaluated conditionally)
5. **Phase 3**: Optimizes thresholds, generates trade signals, sizes positions, backtests strategies
6. Generates visualizations (equity curves, drawdown plots) saved to `plots/` directory
7. Saves models and predictions to `outputs/` directory (for Streamlit demo)

**Note**: By default, `main.py` uses walk-forward validation. To use single split, change `validation_method = "single"` in `main.py`.

### Custom Usage

```python
from src.train_model import train_and_evaluate, walk_forward_validation

# Single train/test split
prob_model_rf, prob_model_lr, return_model_lr, return_model_gb = train_and_evaluate("AAPL", "2y")

# Walk-forward validation
walk_forward_validation("AAPL", "2y", train_size=0.7, test_size=0.1, step=0.1, verbose=True)
```

### Streamlit Demo App

A lightweight tool for inspecting calibrated market bias probabilities and uncertainty. The app demonstrates probabilistic reasoning and uncertainty estimation—not trading signals or performance claims.

**To run the demo**:

1. First, generate artifacts by running the training pipeline:
   ```bash
   python3 src/main.py
   ```
   This saves models and predictions to the `outputs/` directory.

2. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. In the app:
   - Select a ticker (dropdown of available trained stocks)
   - Select a date (dropdown of available dates)
   - View probabilistic bias estimates (P(up), uncertainty, bias regime)
   - Inspect feature values for context
   - See historical context plot (probability vs actual returns)

The app loads pre-computed models and predictions and does not retrain models or simulate trading.

## Features

### Data Pipeline
- Downloads stock data from Yahoo Finance (OHLCV)
- Handles MultiIndex columns automatically
- Works with any stock ticker and time period

### Feature Engineering
Creates 8 features from raw price data:
- **Returns**: Daily percentage change
- **Moving Averages**: 5-day and 20-day (short/medium-term trends)
- **Volatility**: 5-day rolling standard deviation of returns
- **Momentum**: 5-day price change
- **RSI**: Relative Strength Index (14-period momentum oscillator)
- **Price to MA Ratio**: Close price relative to 20-day moving average
- **Volume Ratio**: Current volume relative to 20-day average

All features use only past data (no look-ahead bias).

### Models

**Phase 1: Probability Models (Bias Estimation)**
- **Random Forest Classifier**: Ensemble of 100 decision trees (probabilistic output)
- **Logistic Regression**: Linear model for comparison (probabilistic output)
- **Calibration**: Isotonic regression with 5-fold CV
- **Uncertainty**: Ensemble of 5 models with variance-based confidence

**Phase 2: Expected Return Models**
- **Linear Regression**: Predicts expected return values
- **Gradient Boosting Regressor**: Non-linear regression for expected returns
- **Evaluation**: Conditionally on high-confidence days from Phase 1 (not globally)

**Phase 3: Trading System**
- **Decision Logic**: Bias filter (P(up) threshold) + EV filter (expected return > cost + buffer)
- **Position Sizing**: Fractional Kelly-style heuristic with volatility adjustment (`w = α * E[return] / σ²`, α=0.25, capped at 1.0)
- **Backtesting**: Simulates 5 strategies (Buy & Hold, Always Long, Long Filtered, Long+EV Filtered, Long-Short Optimized)
- **Risk Management**: Stop-loss (2% threshold), volatility-scaled transaction costs
- **Validation**: Walk-forward validation (rolling windows)

### Evaluation

**Phase 1**: Win rate by probability bucket, calibration metrics (Brier score), uncertainty metrics, coverage rate

**Phase 2**: MSE, MAE, R² evaluated conditionally on high-confidence days. Global regression accuracy is not the objective; downstream trading performance is the evaluation target.

**Phase 3**: Total return, Sharpe ratio, max drawdown, hit rate, turnover. Strategy comparison across 5 approaches.

## Project Structure

```
firsttrading/
├── src/
│   ├── data_loader.py          # Downloads stock data from Yahoo Finance
│   ├── feature_engineering.py  # Creates features and labels
│   ├── train_model.py          # Full pipeline: Phase 1-3, backtesting, visualization
│   └── main.py                 # Entry point - runs pipeline on multiple stocks
├── notebooks/
│   └── test_system.ipynb       # Jupyter notebook for interactive testing
├── outputs/                    # Saved models and predictions for Streamlit demo
├── plots/                      # Generated visualizations (equity curves, drawdown plots)
├── data/                       # Directory for data files (empty - data downloaded on-demand)
├── streamlit_app.py            # Streamlit demo app (probabilistic inspection only)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Results and Performance

### What This System Does

**Phase 1: Probabilistic Bias Estimation**
- Estimates P(up) for each day with calibrated confidence
- Quantifies uncertainty (knows when predictions are uncertain)
- Filters out days without edge (neutral zone 0.45-0.55)

**Phase 2: Expected Return Models**
- Predicts expected return values (not just direction)
- Evaluates conditionally on high-confidence days from Phase 1
- Provides inputs for position sizing

**Phase 3: Trading System**
- Generates trading signals (long and short) based on bias + EV filters
- Sizes positions using Kelly-style heuristic with volatility adjustment
- Backtests multiple strategies with comprehensive metrics
- Implements stop-loss risk management
- Generates visualizations (equity curves, drawdown plots)

### Typical Performance

- **Calibration**: Brier scores typically improve after calibration (especially for Random Forest)
- **Win Rate by Bucket**: Higher probability buckets show higher win rates (monotonic behavior)
- **Coverage**: Typically 30-50% of days flagged (model is selective)
- **Uncertainty**: Low variance means models agree (higher confidence)
- **Return Models**: R² values often near 0 or negative (expected for financial data; models used conditionally, not globally)
- **Trading Performance**: Strategy comparison shows relative performance; Sharpe ratios used for comparison across strategies/windows, not as deployable performance claims

### What This System Does NOT Do

- Intended for educational/research purposes, not live trading
- Uses daily data only (no intraday execution)
- Uses simplified cost model (does not account for market impact or slippage)
- Threshold optimization may overfit (not optimized for production deployment)
- Historical backtesting only (does not provide real-time trading signals)

## Critical Caveats and Limitations

### Threshold Optimization

Threshold optimization is performed on a fixed historical window and may overfit. **Threshold optimization is used for sensitivity analysis, not as a claim of deployable alpha.** For production use, optimize on one window and evaluate on a later, untouched window (walk-forward validation).

### Position Sizing

We use a capped, fractional Kelly-style heuristic for risk-aware sizing, not a true Kelly solution. The formula `w = α * E[return] / σ²` is a simplified approximation. True Kelly criterion requires assumptions (known edge, independent bets) that don't hold in practice. We use `α = 0.25` (25% of full Kelly) as a conservative risk parameter, with positions capped at `w_max = 1.0`.

### Transaction Costs

Transaction costs are incorporated as a return threshold buffer. A base cost of 0.1% is used with volatility scaling. This is a simplified model; real trading includes bid-ask spreads, market impact, slippage, and broker-specific fees.

### Expected Return Models (R² Interpretation)

For financial return prediction, low or even negative R² values are expected due to high noise. **We use expected return estimates conditionally after bias filtering (only on high-confidence days from Phase 1), not globally.** The regression models provide expected return estimates that are used conditionally—only after the bias filter identifies high-confidence days. The results demonstrate this conditional usefulness through selective trading performance.

### Performance Metrics Interpretation

**Sharpe ratios are used for relative comparison, not as claims of deployable performance.** High Sharpe ratios (e.g., 3-4) on daily data are not realistic for live trading. Sharpe is used here as a relative comparison metric across strategies and windows.

**Why Sharpe ratios may appear high:**
- Single stock focus (no diversification drag)
- Short sample periods (limited historical data)
- Threshold optimization (may overfit to test period)
- Daily resolution (no intraday volatility/friction)
- No market impact modeling (assumes perfect execution)

Compare Sharpe ratios across strategies or walk-forward windows to understand relative performance, not as absolute performance guarantees.

### Why This Matters

Acknowledging these limitations:
- **Increases credibility**: Shows understanding of real-world trading constraints
- **Sets expectations**: Results are illustrative, not guarantees
- **Demonstrates maturity**: Recognizes the gap between backtesting and live trading

The system is designed for **educational and research purposes** to demonstrate probabilistic bias estimation, calibration, and backtesting methodology.

## Key Technical Details

### Data Leakage Prevention
- All features use only historical data (no future information)
- Train/test split preserves temporal order (shuffle=False)
- Labels are computed correctly (shifted forward properly)

### Probability Calibration
- Isotonic regression maps raw probabilities to calibrated probabilities
- Uses cross-validation to fit calibrator
- Brier score measures calibration quality

### Uncertainty Estimation
- Ensemble of 5 Random Forest models with different random seeds
- Variance across ensemble measures prediction uncertainty
- Lower variance = higher confidence

### Evaluation Strategy
- Time-series appropriate split (80/20, no shuffling)
- Calibration-focused metrics (win rate by bucket, coverage rate)
- Walk-forward validation for realistic evaluation

## Technologies Used

- Python 3.14
- pandas: Data manipulation
- numpy: Numerical operations
- yfinance: Stock data download
- scikit-learn: Machine learning (models, metrics, preprocessing, calibration)
- matplotlib: Visualization (equity curves, drawdown plots)
- streamlit: Interactive demo app (probabilistic inspection tool)
- jupyter: Interactive notebook for testing

## Future Improvements

Potential extensions:
- Regime analysis (performance by market conditions)
- Multi-asset portfolio optimization
- More sophisticated position sizing (e.g., full Kelly optimization)
- Real-time data integration
- Additional risk metrics (VaR, CVaR)
- Machine learning model improvements (neural networks, time-series models)

## License

This is a personal project for learning and portfolio purposes.

## Acknowledgments

This project demonstrates probabilistic bias estimation with calibration and uncertainty quantification. Stock prediction is used as a domain example to showcase quantitative ML techniques for bias estimation and proper evaluation methodology.
