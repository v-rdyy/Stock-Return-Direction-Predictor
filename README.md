# Probabilistic Stock Market Bias Estimator

A quantitative machine learning system that estimates daily directional market bias with calibrated confidence. This project focuses on **probabilistic bias estimation** rather than binary predictions, using calibration, uncertainty quantification, and proper evaluation metrics.

## Project Overview

**Goal:** Estimate the probability that tomorrow's stock return will be positive (P(up)) with calibrated confidence, not just predict up/down.

**Key Points:**
- Probabilistic output: Each day gets P(up) ∈ [0,1] (not just 0 or 1)
- Calibrated probabilities: Probabilities are reliable (0.65 means ~65% historically)
- Uncertainty quantification: Measures prediction confidence through ensemble variance
- Calibration-focused evaluation: Win rate by bucket, coverage rate (not just accuracy)
- Uses only historical data (no future information)

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
- **Random Forest Classifier**: Ensemble of 100 decision trees (probabilistic output)
- **Logistic Regression**: Linear model for comparison (probabilistic output)

### Probability Calibration
- **Isotonic Regression**: Calibrates raw probabilities to reliable probabilities
- Uses 5-fold cross-validation to fit calibrator
- **Brier Score**: Evaluates calibration quality (lower is better)

### Uncertainty Estimation
- **Ensemble Method**: 5 Random Forest models with different random seeds
- **Variance/Std**: Measures prediction variance across ensemble
- Higher variance = more uncertainty (models disagree)
- Lower variance = less uncertainty (models agree)

### Evaluation
- **Win Rate by Probability Bucket**: Groups predictions by probability ranges and calculates actual win rate
  - Low (<0.45): Low probability predictions
  - Neutral (0.45-0.55): No clear edge
  - Weak (0.55-0.60): Weak edge
  - Moderate (0.60-0.70): Moderate edge
  - Strong (>0.70): Strong edge
- **Coverage Rate**: Percent of days flagged as having edge (outside neutral zone)
- **Calibration Metrics**: Brier scores (raw vs calibrated)
- **Uncertainty Metrics**: Variance and std across ensemble
- **Basic Accuracy**: For reference (simplified)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd firsttrading
```

2. Create a virtual environment (recommended):
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

This will:
1. Download 2 years of data for AAPL, MSFT, and AMZN
2. Create features and labels
3. Train probability models (Random Forest, Logistic Regression)
4. Calibrate probabilities
5. Estimate uncertainty
6. Evaluate with calibration-focused metrics
7. Print comprehensive results

### Custom Usage

To test on different stocks, modify `src/main.py`:

```python
stocks = ['AAPL', 'MSFT', 'GOOGL']  # Change stocks here
```

To test a single stock programmatically:

```python
from src.train_model import train_and_evaluate

prob_model_rf, prob_model_lr = train_and_evaluate("AAPL", "2y")
```

## Project Structure

```
firsttrading/
├── src/
│   ├── data_loader.py          # Downloads stock data from Yahoo Finance
│   ├── feature_engineering.py  # Creates features and labels
│   ├── train_model.py          # Trains models, calibrates probabilities, estimates uncertainty
│   └── main.py                 # Entry point - runs pipeline on multiple stocks
├── data/                       # Directory for data files (empty - data downloaded on-demand)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Results and Limitations

### What This System Does

- Estimates P(up) for each day (probabilistic bias)
- Provides calibrated probabilities (reliable confidence levels)
- Quantifies uncertainty (knows when predictions are uncertain)
- Filters out days without edge (neutral zone 0.45-0.55)

### What This System Does NOT Do

- Does not predict exact returns (only direction probability)
- Does not size positions
- Does not force trades
- Does not optimize profit

### Typical Performance

- Calibration: Brier scores typically improve after calibration (especially for Random Forest)
- Win Rate by Bucket: Higher probability buckets should have higher win rates (monotonic behavior)
- Coverage: Typically 30-50% of days flagged (model is selective)
- Uncertainty: Low variance means models agree (higher confidence)

### Expected Return Model Performance (Phase 2)

**Important:** For financial return prediction, low or even negative R² values are expected due to high noise. Model usefulness is judged downstream via selective trading performance, not raw regression scores.

- **R² values** are often near 0 or negative (this is expected and normal)
- **MSE/MAE** are meaningful for comparison between models, but absolute values may appear high
- **Model evaluation** should focus on conditional performance (e.g., performance on high-confidence days) rather than global metrics

### Why Results Are Modest

Stock markets are:
- **Noisy**: Random events affect prices
- **Efficient**: Information is quickly reflected in prices
- **Complex**: Many factors influence returns

This system focuses on **probabilistic bias estimation** - knowing when there's edge, not trying to predict exact outcomes.

## Critical Caveats and Limitations

### A. Threshold Optimization and Overfitting Risk

**Threshold optimization is performed on a fixed historical window and may overfit.**

- Results are used to study sensitivity and relative performance, not to claim deployable alpha.
- The grid search finds thresholds that maximize Sharpe ratio on the test set, which may not generalize to future data.
- **For production use:** Optimize on one window and evaluate on a later, untouched window (walk-forward validation).

**What this means:** The optimized thresholds shown are best-case results for the specific historical period tested. Real-world performance may differ.

### B. Position Sizing: Kelly-Style Heuristic

**We use a capped, fractional Kelly-style heuristic for risk-aware sizing, not a true Kelly solution.**

- The formula `w = α * E[return] / σ²` is a simplified approximation.
- True Kelly criterion requires assumptions (known edge, independent bets) that don't hold in practice.
- We use `α = 0.25` (25% of full Kelly) as a conservative risk parameter.
- Position sizes are capped at `w_max = 1.0` (100% of capital).

**What this means:** Position sizing is a risk-aware heuristic, not an optimal solution. It provides reasonable risk management but is not theoretically optimal.

### C. Transaction Costs

**Results incorporate transaction costs as a return threshold buffer.**

- A fixed transaction cost (0.1%) is incorporated via the EV threshold (expected return must exceed cost + buffer).
- This is a simplified model; real trading has:
  - Bid-ask spreads (variable by stock and volatility)
  - Market impact (larger positions move prices)
  - Slippage (execution price vs expected price)
  - Brokerage fees (varies by broker)

**What this means:** Backtested returns are more optimistic than real-world results. Actual trading would have higher costs, especially for larger positions or volatile stocks.

### Summary

These caveats are important for:
- **Credibility**: Acknowledging limitations shows domain understanding
- **Realistic expectations**: Results are illustrative, not guarantees
- **Interview readiness**: Quant interviewers will ask about these limitations

The system is designed for **educational and research purposes** to demonstrate probabilistic bias estimation, calibration, and backtesting methodology.

### Project Framing

**This is a probabilistic bias estimation system, not a trading system.**

The value is in:
- Probabilistic output (P(up), not just up/down)
- Calibrated probabilities (reliable confidence)
- Uncertainty quantification (knows when uncertain)
- Proper evaluation (calibration-focused, not just accuracy)

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
- Basic accuracy for reference

### Models Used
- **Random Forest**: Non-linear, ensemble method (probabilistic output)
- **Logistic Regression**: Linear, interpretable (probabilistic output)

## Technologies Used

- Python 3.14
- pandas: Data manipulation
- numpy: Numerical operations
- yfinance: Stock data download
- scikit-learn: Machine learning (models, metrics, preprocessing, calibration)

## Future Improvements (Optional)

Potential extensions:
- Expected return models (Phase 2)
- Position sizing based on probability and uncertainty
- Trading decision logic (filter by bias and expected return)
- Backtesting engine
- Regime analysis (performance by market conditions)
- Walk-forward validation (more realistic time-series evaluation)

## License

This is a personal project for learning and portfolio purposes.

## Acknowledgments

This project demonstrates probabilistic bias estimation with calibration and uncertainty quantification. Stock prediction is used as a domain example to showcase quantitative ML techniques for bias estimation and proper evaluation methodology.
