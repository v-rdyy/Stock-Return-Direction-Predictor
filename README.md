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

### Why Results Are Modest

Stock markets are:
- **Noisy**: Random events affect prices
- **Efficient**: Information is quickly reflected in prices
- **Complex**: Many factors influence returns

This system focuses on **probabilistic bias estimation** - knowing when there's edge, not trying to predict exact outcomes.

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
