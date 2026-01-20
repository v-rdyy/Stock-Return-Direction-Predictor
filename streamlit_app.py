"""
Probabilistic Market Bias Inspection Tool

Research demo for inspecting calibrated market bias probabilities and uncertainty.
This tool is for analysis only and does not provide trading signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Page config
st.set_page_config(page_title="Market Bias Inspection", layout="wide")

# Title and disclaimer
st.title("Probabilistic Market Bias Inspection Tool")

st.caption(
    "Research demo for inspecting calibrated market bias probabilities and uncertainty. "
    "This tool is for analysis only and does not provide trading signals."
)

# Sidebar: inputs
st.sidebar.header("Configuration")

# Available stocks (from outputs directory)
outputs_dir = Path("outputs")
available_stocks = []
if outputs_dir.exists():
    # Find all stocks with saved models
    for file in outputs_dir.glob("*_calibrated_model.pkl"):
        stock = file.stem.replace("_calibrated_model", "")
        available_stocks.append(stock)
    available_stocks = sorted(set(available_stocks))

if not available_stocks:
    st.error(
        "No saved models found. Please run the training pipeline first to generate outputs in the `outputs/` directory."
    )
    st.stop()

ticker = st.sidebar.selectbox(
    "Select Ticker",
    available_stocks,
    help="Choose a stock ticker to inspect. Models must be trained for the ticker first."
)

# Load predictions dataframe to get available dates
predictions_path = outputs_dir / f"{ticker}_predictions.csv"
if not predictions_path.exists():
    st.error(f"Predictions file not found for {ticker}. Please run training first.")
    st.stop()

predictions_df = pd.read_csv(predictions_path)
predictions_df['date'] = pd.to_datetime(predictions_df['date'])

# Get available dates (most recent first)
available_dates = sorted(predictions_df['date'].dt.date.unique(), reverse=True)
if not available_dates:
    st.error(f"No dates found in predictions for {ticker}.")
    st.stop()

date = st.sidebar.selectbox(
    "Select Date",
    available_dates,
    index=0,
    help="Select a date to inspect. Most recent dates shown first."
)

# Load models
@st.cache_data
def load_models(ticker):
    """Load saved models and scaler for the given ticker."""
    model_path = outputs_dir / f"{ticker}_calibrated_model.pkl"
    scaler_path = outputs_dir / f"{ticker}_scaler.pkl"
    ensemble_path = outputs_dir / f"{ticker}_ensemble_models.pkl"
    
    with open(model_path, 'rb') as f:
        calibrated_model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(ensemble_path, 'rb') as f:
        ensemble_models = pickle.load(f)
    
    return calibrated_model, scaler, ensemble_models

try:
    calibrated_model, scaler, ensemble_models = load_models(ticker)
except FileNotFoundError as e:
    st.error(f"Model files not found for {ticker}. Please run training first.")
    st.stop()

# Main content: Get data for selected date
date_mask = predictions_df['date'].dt.date == date
if not date_mask.any():
    st.error(f"No data found for {date}.")
    st.stop()

row = predictions_df[date_mask].iloc[0]

# Extract key variables
p_calibrated = row['p_up']
uncertainty = row['uncertainty']
actual_return = row['actual_return']
actual_label = row['actual_label']

# Summary info box
next_day_direction = "📈 Up" if actual_label == 1 else "📉 Down" if actual_label == 0 else "—"
actual_return_formatted = f"{actual_return:+.2%}" if pd.notna(actual_return) else "N/A"

st.info(
    f"**📅 Selected Date**: {date.strftime('%B %d, %Y')} | "
    f"**📊 Ticker**: {ticker} | "
    f"**📈 Next-Day Return**: {actual_return_formatted} ({next_day_direction}) | "
    f"**🎯 Model P(up)**: {p_calibrated:.1%}"
)

# Core outputs: Probabilistic bias
st.header("Probabilistic Bias Estimate")

col1, col2, col3 = st.columns(3)

# Determine bias regime with strength
if p_calibrated > 0.65:
    bias_label = "Strong Bullish"
    bias_color = "green"
elif p_calibrated > 0.55:
    bias_label = "Bullish Bias"
    bias_color = "normal"
elif p_calibrated < 0.35:
    bias_label = "Strong Bearish"
    bias_color = "inverse"
elif p_calibrated < 0.45:
    bias_label = "Bearish Bias"
    bias_color = "off"
else:
    bias_label = "Neutral / No Edge"
    bias_color = "off"

with col1:
    # Determine strength of probability signal
    if p_calibrated > 0.65:
        prob_delta = "Very Bullish"
        prob_color = "green"
    elif p_calibrated > 0.55:
        prob_delta = "Bullish"
        prob_color = "normal"
    elif p_calibrated < 0.35:
        prob_delta = "Very Bearish"
        prob_color = "inverse"
    elif p_calibrated < 0.45:
        prob_delta = "Bearish"
        prob_color = "off"
    else:
        prob_delta = "Neutral"
        prob_color = "off"
    
    # Calculate distance from neutral (0.5)
    distance_from_neutral = abs(p_calibrated - 0.5)
    
    st.metric(
        label="P(up) — Calibrated Probability",
        value=f"{p_calibrated:.2%}",
        delta=f"{prob_delta} ({distance_from_neutral:.1%} from neutral)",
        delta_color=prob_color,
        help="Probability of positive return tomorrow. Calibrated to reflect historical frequencies. >0.55 = Bullish, <0.45 = Bearish, 0.45-0.55 = Neutral"
    )

with col2:
    # Determine uncertainty level for context
    if uncertainty < 0.001:
        uncertainty_level = "Very Low"
        uncertainty_color = "green"
    elif uncertainty < 0.005:
        uncertainty_level = "Low"
        uncertainty_color = "normal"
    elif uncertainty < 0.01:
        uncertainty_level = "Moderate"
        uncertainty_color = "off"
    else:
        uncertainty_level = "High"
        uncertainty_color = "inverse"
    
    st.metric(
        label="Model Uncertainty (Ensemble Variance)",
        value=f"{uncertainty:.4f}",
        delta=f"{uncertainty_level}",
        delta_color=uncertainty_color,
        help="Variance across 5 ensemble models. Lower = more agreement (higher confidence). Typical range: 0.0001-0.01"
    )

with col3:
    st.metric(
        label="Bias Regime",
        value=bias_label,
        delta_color=bias_color,
        help="Market bias interpretation: Bullish (>0.55) = upward bias, Bearish (<0.45) = downward bias, Neutral (0.45-0.55) = no clear edge"
    )

# Feature context
st.header("Feature Snapshot")

feature_names = {
    'returns': '5d Return',
    'ma5': '5d Moving Average',
    'ma20': '20d Moving Average',
    'volatility': '5d Volatility',
    'momentum': 'Momentum (5d)',
    'rsi': 'RSI (14)',
    'price_to_ma': '20d Moving Avg Ratio',
    'volume_ratio': 'Volume Ratio'
}

feature_descriptions = {
    'returns': '5-day cumulative return (percentage change)',
    'ma5': '5-day simple moving average of closing price',
    'ma20': '20-day simple moving average of closing price',
    'volatility': '5-day rolling standard deviation of returns (risk measure)',
    'momentum': '5-day price change (trend indicator)',
    'rsi': 'Relative Strength Index (0-100, >70 overbought, <30 oversold)',
    'price_to_ma': 'Current price relative to 20-day MA (>1 = above MA, <1 = below MA)',
    'volume_ratio': 'Current volume relative to 20-day average (>1 = above average)'
}

feature_data = []
for key, label in feature_names.items():
    if key in row:
        value = row[key]
        # Format value based on feature type
        if key in ['rsi']:
            formatted_value = f"{value:.2f}"
            # Add interpretation for RSI
            if value > 70:
                interpretation = "Overbought"
            elif value < 30:
                interpretation = "Oversold"
            else:
                interpretation = "Neutral"
        elif key in ['price_to_ma']:
            formatted_value = f"{value:.4f}"
            interpretation = "Above MA" if value > 1 else "Below MA"
        elif key in ['volume_ratio']:
            formatted_value = f"{value:.2f}"
            interpretation = "Above Avg" if value > 1 else "Below Avg"
        elif key in ['returns', 'momentum']:
            formatted_value = f"{value:.4f}"
            interpretation = "Positive" if value > 0 else "Negative"
        elif key in ['volatility']:
            formatted_value = f"{value:.4f}"
            interpretation = "High" if value > 0.02 else "Low" if value < 0.01 else "Moderate"
        else:
            formatted_value = f"{value:.4f}"
            interpretation = ""
        
        feature_data.append({
            "Feature": label,
            "Value": formatted_value,
            "Interpretation": interpretation,
            "Description": feature_descriptions.get(key, "")
        })

feature_df = pd.DataFrame(feature_data)

# Display with better formatting
st.dataframe(
    feature_df,
    width='stretch',
    hide_index=True,
    column_config={
        "Feature": st.column_config.TextColumn(
            "Feature",
            help="Technical indicator or market metric"
        ),
        "Value": st.column_config.TextColumn(
            "Value",
            help="Current feature value for the selected date"
        ),
        "Interpretation": st.column_config.TextColumn(
            "Interpretation",
            help="Quick interpretation of the feature value"
        ),
        "Description": st.column_config.TextColumn(
            "Description",
            help="Detailed explanation of what this feature measures",
            width="large"
        )
    }
)

# Add summary stats
st.caption(
    f"💡 **Feature Context**: "
    f"RSI: {'📈 Overbought' if row.get('rsi', 50) > 70 else '📉 Oversold' if row.get('rsi', 50) < 30 else '⚖️ Neutral'}, "
    f"Price vs MA: {'📊 Above' if row.get('price_to_ma', 1) > 1 else '📊 Below'}, "
    f"Volume: {'🔊 High' if row.get('volume_ratio', 1) > 1.5 else '🔇 Normal' if row.get('volume_ratio', 1) > 0.5 else '🔇 Low'}, "
    f"Volatility: {'⚠️ High' if row.get('volatility', 0) > 0.02 else '✅ Low'}"
)

# Historical context plot
st.header("Historical Context")

st.caption(
    "📊 **Historical Context Plot**: Shows how calibrated probabilities (P(up)) aligned with actual returns over the last 60 days. "
    "The blue line shows P(up) over time, with green/red dots showing positive/negative actual returns. "
    "Vertical orange line highlights the selected date."
)

# Get last 60 days of data for context
recent_df = predictions_df[predictions_df['date'] <= pd.Timestamp(date)].tail(60)

if len(recent_df) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot calibrated probabilities
    ax.plot(recent_df['date'], recent_df['p_up'], 
            label='P(up) — Calibrated Probability', color='blue', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    ax.axhline(y=0.55, color='green', linestyle='--', alpha=0.3, label='Bullish Threshold (0.55)')
    ax.axhline(y=0.45, color='red', linestyle='--', alpha=0.3, label='Bearish Threshold (0.45)')
    
    # Scatter plot: actual returns (color by sign)
    colors = ['green' if r > 0 else 'red' for r in recent_df['actual_return']]
    ax2 = ax.twinx()
    ax2.scatter(recent_df['date'], recent_df['actual_return'], 
               alpha=0.6, s=30, c=colors, label='Next-Day Return')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('P(up) — Calibrated Probability', color='blue')
    ax2.set_ylabel('Actual Return', color='black')
    ax.set_title(f'{ticker}: Probability vs Actual Returns (Last 60 Days)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Highlight selected date
    selected_row = recent_df[recent_df['date'].dt.date == date]
    if not selected_row.empty:
        selected_date = selected_row['date'].iloc[0]
        ax.axvline(x=selected_date, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Not enough historical data for visualization.")

# Interpretation notes
st.info(
    "**Interpretation Notes:**\n\n"
    "- **Probabilities** reflect historical frequencies after calibration. "
    "A predicted 65% chance of up means that historically, 65% of such predictions resulted in an up move.\n\n"
    "- **Bias Regime**: Market bias interpretation based on P(up). "
    "**Bullish** (>0.55) = upward bias, **Bearish** (<0.45) = downward bias, **Neutral** (0.45-0.55) = no clear edge. "
    "**Strong Bullish** (>0.65) and **Strong Bearish** (<0.35) indicate stronger signals.\n\n"
    "- **Uncertainty (Ensemble Variance)**: Measures disagreement across 5 Random Forest models. "
    "Lower variance = models agree more = higher confidence. "
    "Typical values: <0.001 (very low), 0.001-0.005 (low), 0.005-0.01 (moderate), >0.01 (high).\n\n"
    "- **This tool is for research inspection only**, not trading. "
    "Probabilistic estimates do not guarantee future outcomes."
)
