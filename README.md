# Stock Return Direction Predictor

A machine learning project that predicts whether a stock's next-day return will be positive or negative using historical price data. This is a **binary classification problem** focusing on proper ML practices, evaluation, and analysis - not a trading system.

## Project Overview

**Goal:** Predict if tomorrow's stock return will be positive (1) or negative (0) using only past information.

**Key Points:**
- Binary classification problem (up/down prediction)
- Uses only historical data (no future information)
- Proper train/test split with temporal ordering
- Comprehensive evaluation with baselines and multiple metrics
- Error analysis to understand model failures

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
- **Random Forest Classifier**: Ensemble of 100 decision trees
- **Logistic Regression**: Linear model for comparison and interpretability

### Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Baselines**: 
  - Always predict "up"
  - Predict based on yesterday's return
- **Feature Importance**: Analysis of which features matter most
- **Error Analysis**: 
  - False positives vs false negatives
  - Performance by volatility (high vs low periods)
  - Error pattern analysis

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
3. Train both models
4. Evaluate performance
5. Print comprehensive results including error analysis

### Custom Usage

To test on different stocks, modify `src/main.py`:

```python
stocks = ['AAPL', 'MSFT', 'GOOGL']  # Change stocks here
```

To test a single stock programmatically:

```python
from src.train_model import train_and_evaluate

model, lr_model = train_and_evaluate("AAPL", "2y")
```

## Project Structure

```
firsttrading/
├── src/
│   ├── data_loader.py          # Downloads stock data from Yahoo Finance
│   ├── feature_engineering.py  # Creates features and labels
│   ├── train_model.py          # Trains models and evaluates performance
│   └── main.py                 # Entry point - runs pipeline on multiple stocks
├── data/                       # Directory for data files (empty - data downloaded on-demand)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Results and Limitations

### Typical Performance
- Model accuracy: ~50-55% (slightly better than random 50%)
- Baseline accuracy: ~51-56% (simple rules often perform similarly)
- This is **expected and realistic** - stock direction prediction is inherently difficult

### Why Results Are Modest
Stock markets are:
- **Noisy**: Random events affect prices
- **Efficient**: Information is quickly reflected in prices
- **Complex**: Many factors influence returns

Academic research shows similar challenges - beating simple baselines consistently is difficult.

### Project Framing
**This is a modeling and evaluation project, not a trading system.**

The value is in:
- Demonstrating proper ML practices
- Comprehensive evaluation methodology
- Understanding model limitations
- Building a complete, professional pipeline

## Key Technical Details

### Data Leakage Prevention
- All features use only historical data (no future information)
- Train/test split preserves temporal order (shuffle=False)
- Labels are computed correctly (shifted forward properly)

### Evaluation Strategy
- Time-series appropriate split (80/20, no shuffling)
- Multiple metrics (not just accuracy)
- Baseline comparisons
- Error analysis by market conditions

### Models Used
- **Random Forest**: Non-linear, can capture complex patterns
- **Logistic Regression**: Linear, more interpretable (coefficients show feature impact)

## Technologies Used

- Python 3.14
- pandas: Data manipulation
- numpy: Numerical operations
- yfinance: Stock data download
- scikit-learn: Machine learning (models, metrics, preprocessing)

## Future Improvements (Optional)

Potential extensions:
- Walk-forward validation (more realistic time-series evaluation)
- Cross-sectional prediction (predict across multiple stocks simultaneously)
- Additional features (technical indicators, market sentiment)
- Model hyperparameter tuning
- Visualization dashboard

## License

This is a personal project for learning and portfolio purposes.

## Acknowledgments

This project follows best practices for machine learning evaluation and time-series data handling. Stock prediction is used as a domain example to demonstrate ML pipeline development and evaluation skills.

