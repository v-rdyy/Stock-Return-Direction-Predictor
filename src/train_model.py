from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import CalibratedClassifierCV

import numpy as np

import pandas as pd 

from src.data_loader import download_stock_data
from src.feature_engineering import create_features, create_labels

def analyze_errors(y_test, y_pred, X_test):
    """
    Analyzes prediction errors to understand when/why the model fails.
    Looks at false positives, false negatives, and performance by volatility.
    """
    # Identify different types of predictions
    fp_mask = (y_pred == 1) & (y_test == 0)  # Predicted up, but was down
    fn_mask = (y_pred == 0) & (y_test == 1)  # Predicted down, but was up
    tp_mask = (y_pred == 1) & (y_test == 1)  # Predicted up, and was up
    tn_mask = (y_pred == 0) & (y_test == 0)  # Predicted down, and was down

    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    tp_count = tp_mask.sum()
    tn_count = tn_mask.sum()

    # Split into high/low volatility periods to see when model performs better
    test_volatility = X_test['volatility'].values
    median_vol = np.median(test_volatility)
    high_vol_mask = test_volatility > median_vol
    low_vol_mask = test_volatility <= median_vol

    high_vol_correct = (y_pred[high_vol_mask] == y_test[high_vol_mask]).sum()
    high_vol_total = high_vol_mask.sum()
    high_vol_accuracy = high_vol_correct / high_vol_total if high_vol_total > 0 else 0

    low_vol_correct = (y_pred[low_vol_mask] == y_test[low_vol_mask]).sum()
    low_vol_total = low_vol_mask.sum()
    low_vol_accuracy = low_vol_correct / low_vol_total if low_vol_total > 0 else 0

    if fp_count > 0:
        fp_features = X_test[fp_mask].mean()
    else:
        fp_features = None

    if fn_count > 0:
        fn_features = X_test[fn_mask].mean()
    else:
        fn_features = None

    print("\n=== Error Analysis ===")
    print(f"False Positives (predicted up, was down): {fp_count}")
    print(f"False Negatives (predicted down, was up): {fn_count}")
    print(f"True Positives: {tp_count}")
    print(f"True Negatives: {tn_count}")

    print(f"\nPerformance by Volatility:")
    print(f"High Volatility Accuracy: {high_vol_accuracy:.2%}")
    print(f"Low Volatility Accuracy: {low_vol_accuracy:.2%}")

    if fp_count > 0:
        print(f"\nFalse Positive Patterns (avg feature values):")
        for feature, value in fp_features.items():
            print(f"  {feature:15}: {value:8.4f}")
    if fn_count > 0:
        print(f"\nFalse Negative Patterns (avg feature values):")
        for feature, value in fn_features.items():
            print(f"  {feature:15}: {value:8.4f}")


def backtest_strategy(returns, trade_signals, position_sizes=None):
    """
    Backtests a trading strategy.
    
    Args:
        returns: Actual returns (array or Series)
        trade_signals: Boolean array (True = trade, False = no trade)
        position_sizes: Position sizes (array, optional). If None, uses fixed size 1.0
    
    Returns:
        dict with performance metrics (total_return, sharpe_ratio, max_drawdown, hit_rate, turnover)
    """
    # Convert to numpy arrays to avoid pandas deprecation warnings
    returns = np.array(returns)
    trade_signals = np.array(trade_signals)
    
    if position_sizes is None:
        position_sizes = np.ones(len(trade_signals))
        position_sizes[~trade_signals] = 0  # No position when no signal
    else:
        position_sizes = np.array(position_sizes)
    
    # Portfolio value over time (start with 1.0)
    portfolio_value = np.ones(len(returns))
    
    # Calculate portfolio returns
    # Portfolio return = position_size * actual_return
    portfolio_returns = position_sizes * returns
    
    # Cumulative portfolio value
    for i in range(1, len(portfolio_value)):
        portfolio_value[i] = portfolio_value[i-1] * (1 + portfolio_returns[i-1])
    
    # Total return
    total_return = portfolio_value[-1] - 1.0
    
    # Annualized Sharpe ratio (assuming 252 trading days per year)
    # Sharpe = mean(returns) / std(returns) * sqrt(252)
    if portfolio_returns.std() > 0:
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # Max drawdown
    # Drawdown = (peak - current) / peak
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (running_max - portfolio_value) / running_max
    max_drawdown = drawdown.max()
    
    # Hit rate (percentage of profitable trades)
    trade_returns = portfolio_returns[trade_signals]
    if len(trade_returns) > 0:
        hit_rate = (trade_returns > 0).sum() / len(trade_returns)
    else:
        hit_rate = 0.0
    
    # Turnover (average absolute position size - measures trading activity)
    turnover = np.abs(position_sizes).mean()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'turnover': turnover,
        'portfolio_value': portfolio_value
    }


def optimize_thresholds(y_proba_cal, y_return_pred, y_return_actual, volatility, alpha=0.25, w_max=1.0, transaction_cost=0.001, allow_shorting=True):
    """
    Optimizes bias threshold and EV threshold to maximize Sharpe ratio.
    Supports both long-only and long-short strategies.
    
    **CRITICAL CAVEAT - Overfitting Risk:**
    Threshold optimization is performed on a fixed historical window and may overfit.
    Results are used to study sensitivity and relative performance, not to claim deployable alpha.
    For production use, optimize on one window and evaluate on a later, untouched window.
    
    Args:
        y_proba_cal: Calibrated probabilities (array)
        y_return_pred: Predicted returns (array)
        y_return_actual: Actual returns (array)
        volatility: Volatility values (array)
        alpha: Kelly fraction (default 0.25)
        w_max: Maximum position size (default 1.0)
        transaction_cost: Transaction cost (default 0.001)
        allow_shorting: If True, also test short positions (default True)
    
    Returns:
        dict with best thresholds and performance metrics
    """
    # Convert to numpy arrays
    y_proba_cal = np.array(y_proba_cal)
    y_return_pred = np.array(y_return_pred)
    y_return_actual = np.array(y_return_actual)
    volatility = np.array(volatility)
    
    # Grid search: test different threshold combinations
    long_bias_thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    long_ev_thresholds = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]  # EV threshold (transaction_cost + buffer)
    
    # For shorting: P(up) < threshold means short (e.g., P(up) < 0.45 means 55% chance of down)
    # Mirror the long thresholds: short_bias = 1 - long_bias
    if allow_shorting:
        short_bias_thresholds = [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]  # P(up) < these values
        short_ev_thresholds = [-0.003, -0.0025, -0.002, -0.0015, -0.001, -0.0005]  # Negative returns (stock goes down)
    else:
        short_bias_thresholds = []
        short_ev_thresholds = []
    
    best_sharpe = -np.inf
    best_config = None
    
    results = []
    
    # Test long positions
    for bias_thresh in long_bias_thresholds:
        for ev_thresh in long_ev_thresholds:
            # Long filters: P(up) > threshold AND E[return] > threshold
            long_bias_filter = y_proba_cal > bias_thresh
            long_ev_filter = y_return_pred > ev_thresh
            long_signal = long_bias_filter & long_ev_filter
            
            # Always test long-only strategy first (even if shorting is allowed)
            if long_signal.sum() > 0:
                # Calculate position sizes (long only)
                position_size = np.zeros(len(long_signal))
                position_size[long_signal] = alpha * y_return_pred[long_signal] / (volatility[long_signal] ** 2)
                position_size = np.clip(position_size, 0, w_max)
                
                # Backtest long-only
                results_dict = backtest_strategy(y_return_actual, long_signal, position_size)
                
                # Store results
                config = {
                    'long_bias_threshold': bias_thresh,
                    'long_ev_threshold': ev_thresh,
                    'short_bias_threshold': None,
                    'short_ev_threshold': None,
                    'n_trades': long_signal.sum(),
                    'n_long': long_signal.sum(),
                    'n_short': 0,
                    'sharpe_ratio': results_dict['sharpe_ratio'],
                    'total_return': results_dict['total_return'],
                    'max_drawdown': results_dict['max_drawdown'],
                    'hit_rate': results_dict['hit_rate']
                }
                results.append(config)
                
                # Track best (by Sharpe ratio)
                if results_dict['sharpe_ratio'] > best_sharpe:
                    best_sharpe = results_dict['sharpe_ratio']
                    best_config = config
            
            # If shorting is allowed, also test long-short combinations
            if allow_shorting:
                for short_bias_thresh in short_bias_thresholds:
                    for short_ev_thresh in short_ev_thresholds:
                        # Short filters: P(up) < threshold AND E[return] < threshold (negative)
                        short_bias_filter = y_proba_cal < short_bias_thresh
                        short_ev_filter = y_return_pred < short_ev_thresh
                        short_signal = short_bias_filter & short_ev_filter
                        
                        # Combined: long OR short (but not both on same day)
                        trade_signal = long_signal | short_signal
                        
                        # Skip if no trades at all
                        if trade_signal.sum() == 0:
                            continue
                        
                        # Calculate position sizes
                        position_size = np.zeros(len(trade_signal))
                        
                        # Long positions: positive size
                        position_size[long_signal] = alpha * y_return_pred[long_signal] / (volatility[long_signal] ** 2)
                        # Short positions: negative size (using absolute value of predicted return)
                        position_size[short_signal] = -alpha * np.abs(y_return_pred[short_signal]) / (volatility[short_signal] ** 2)
                        
                        # Clip to [-w_max, w_max]
                        position_size = np.clip(position_size, -w_max, w_max)
                        
                        # Backtest
                        results_dict = backtest_strategy(y_return_actual, trade_signal, position_size)
                        
                        # Count long and short trades
                        n_long = long_signal.sum()
                        n_short = short_signal.sum()
                        
                        # Store results
                        config = {
                            'long_bias_threshold': bias_thresh,
                            'long_ev_threshold': ev_thresh,
                            'short_bias_threshold': short_bias_thresh,
                            'short_ev_threshold': short_ev_thresh,
                            'n_trades': trade_signal.sum(),
                            'n_long': n_long,
                            'n_short': n_short,
                            'sharpe_ratio': results_dict['sharpe_ratio'],
                            'total_return': results_dict['total_return'],
                            'max_drawdown': results_dict['max_drawdown'],
                            'hit_rate': results_dict['hit_rate']
                        }
                        results.append(config)
                        
                        # Track best (by Sharpe ratio)
                        if results_dict['sharpe_ratio'] > best_sharpe:
                            best_sharpe = results_dict['sharpe_ratio']
                            best_config = config
    
    # Convert results to DataFrame for easier viewing
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    else:
        results_df = pd.DataFrame()
        best_config = {
            'long_bias_threshold': 0.55,
            'long_ev_threshold': 0.0015,
            'short_bias_threshold': None,
            'short_ev_threshold': None,
            'n_trades': 0,
            'n_long': 0,
            'n_short': 0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'hit_rate': 0.0
        }
    
    return best_config, results_df


def train_and_evaluate(stock, period):

    df = download_stock_data(stock, period)
    df = create_features(df)
    df = create_labels(df)

    # Data leakage check: All features use only past data (no future information).
    # Features like returns, moving averages, volatility all look backward.
    # Labels use tomorrow's return but are computed correctly (shifted forward).
    # Train/test split preserves time order (shuffle=False) so we test on future data.

    features = ['returns', 'ma5', 'ma20', 'volatility', 'momentum', 'rsi', 'price_to_ma', 'volume_ratio']
    X = df[features]
    y = df['label']  # Binary labels for probability models (Phase 1)
    y_return = df['return_target']  # Return values for return models (Phase 2)

    # 80/20 split, no shuffling (time-series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # Split return target with the same split (same indices since shuffle=False)
    _, _, y_return_train, y_return_test = train_test_split(X, y_return, test_size=0.2, shuffle=False)

    # Scale features (fit on train, transform both)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest (probability model) - main model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_proba_rf = model.predict_proba(X_test_scaled)[:, 1]  # P(up) probabilities
    y_pred = (y_proba_rf > 0.5).astype(int)  # Binary predictions for evaluation

    # Train ensemble of Random Forest models (for uncertainty estimation)
    # Multiple models with different random seeds capture prediction variance
    # Higher variance = more uncertainty (models disagree), Lower variance = less uncertainty (models agree)
    n_models = 5
    ensemble_probas = [y_proba_rf]  # Include main model
    for i in range(1, n_models):  # Start from 1 since we already have main model
        rf_ensemble = RandomForestClassifier(n_estimators=100, random_state=42+i)
        rf_ensemble.fit(X_train_scaled, y_train)
        proba = rf_ensemble.predict_proba(X_test_scaled)[:, 1]
        ensemble_probas.append(proba)

    # Calculate uncertainty: variance and std across ensemble predictions
    # axis=0 means variance/std across models (down columns), not across samples
    ensemble_probas = np.array(ensemble_probas)  # Shape: (n_models, n_samples)
    uncertainty = np.var(ensemble_probas, axis=0)  # Variance across models for each sample
    uncertainty_std = np.std(ensemble_probas, axis=0)  # Std across models for each sample

    # Train Logistic Regression (probability model)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]  # P(up) probabilities
    y_pred_lr = (y_proba_lr > 0.5).astype(int)  # Binary predictions for evaluation

    # Calibrate probabilities using isotonic regression
    # Calibration maps raw probabilities to calibrated probabilities (ensures reliability)
    # Example: If model says P(up)=0.65, after calibration it actually means ~65% historically
    # Uses 5-fold cross-validation internally to fit the calibrator
    calibrated_rf = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_rf.fit(X_train_scaled, y_train)
    y_proba_rf_cal = calibrated_rf.predict_proba(X_test_scaled)[:, 1]

    calibrated_lr = CalibratedClassifierCV(lr_model, method='isotonic', cv=5)
    calibrated_lr.fit(X_train_scaled, y_train)
    y_proba_lr_cal = calibrated_lr.predict_proba(X_test_scaled)[:, 1]

    # Train return models (regression) - Phase 2
    # Linear Regression baseline
    return_model_lr = LinearRegression()
    return_model_lr.fit(X_train_scaled, y_return_train)
    y_return_pred_lr = return_model_lr.predict(X_test_scaled)

    # Gradient Boosting Regressor (non-linear)
    return_model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    return_model_gb.fit(X_train_scaled, y_return_train)
    y_return_pred_gb = return_model_gb.predict(X_test_scaled)

    # Calculate regression metrics (MSE, MAE, R²) for return models
    lr_mse = mean_squared_error(y_return_test, y_return_pred_lr)
    lr_mae = mean_absolute_error(y_return_test, y_return_pred_lr)
    lr_r2 = r2_score(y_return_test, y_return_pred_lr)

    gb_mse = mean_squared_error(y_return_test, y_return_pred_gb)
    gb_mae = mean_absolute_error(y_return_test, y_return_pred_gb)
    gb_r2 = r2_score(y_return_test, y_return_pred_gb)

    # Calculate Brier scores (calibration quality metric)
    # Lower Brier score = better calibration (0 = perfect, 1 = worst)
    # Compares raw vs calibrated to see if calibration improved probabilities
    brier_rf_raw = brier_score_loss(y_test, y_proba_rf)
    brier_rf_cal = brier_score_loss(y_test, y_proba_rf_cal)
    brier_lr_raw = brier_score_loss(y_test, y_proba_lr)
    brier_lr_cal = brier_score_loss(y_test, y_proba_lr_cal)
    
    # Get feature importances for analysis
    rf_feature_importance = model.feature_importances_
    lr_coefficients = lr_model.coef_[0]
    
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'RF_Importance': rf_feature_importance,
        'LR_Coefficient': lr_coefficients
    })
    feature_importance_df = feature_importance_df.sort_values('RF_Importance', ascending=False)

    # Simple baselines for comparison
    y_baseline_always_up = np.ones(len(y_test))  # Always predict up
    yesterday_returns = X_test['returns'].values
    y_baseline_yesterday = (yesterday_returns > 0).astype(int)  # Predict based on yesterday

    baseline_accuracy_1 = accuracy_score(y_test, y_baseline_always_up)
    baseline_accuracy_2 = accuracy_score(y_test, y_baseline_yesterday)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    lr_precision = precision_score(y_test, y_pred_lr)
    lr_recall = recall_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr)
    lr_confusion = confusion_matrix(y_test, y_pred_lr)

    # Basic accuracy metrics (for reference)
    print("\n=== Baseline Models ===")
    print(f"Always predict up: {baseline_accuracy_1:.2%}")
    print(f"Predict based on yesterday's return: {baseline_accuracy_2:.2%}")

    print("\n=== Basic Accuracy (Reference) ===")
    print(f"Random Forest Accuracy: {accuracy:.2%}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")
    
    # Win rate by probability bucket (calibration-focused evaluation)
    # Divides predictions into probability buckets and calculates actual win rate in each bucket
    # Good calibration: predicted probability ≈ actual win rate
    # Monotonic behavior: higher probability buckets should have higher win rates
    print("\n=== Win Rate by Probability Bucket (Random Forest Calibrated) ===")
    buckets = [(0.0, 0.45), (0.45, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.0)]
    bucket_labels = ['Low (<0.45)', 'Neutral (0.45-0.55)', 'Weak (0.55-0.60)', 'Moderate (0.60-0.70)', 'Strong (>0.70)']
    
    for (low, high), label in zip(buckets, bucket_labels):
        # Create mask: True for samples where probability is in this bucket's range
        # Handle edge case for last bucket: include 1.0 (use <= instead of <)
        if high >= 1.0:
            mask = (y_proba_rf_cal >= low) & (y_proba_rf_cal <= high)
        else:
            mask = (y_proba_rf_cal >= low) & (y_proba_rf_cal < high)
        
        # Only calculate statistics if there are samples in this bucket
        if mask.sum() > 0:
            win_rate = y_test[mask].mean()  # Actual positive rate (proportion of up days)
            count = mask.sum()  # Number of samples in this bucket
            avg_prob = y_proba_rf_cal[mask].mean()  # Average predicted probability in this bucket
            print(f"{label:20}: {count:4d} samples, Avg P(up): {avg_prob:.3f}, Win rate: {win_rate:.2%}")
    
    # Coverage rate: percent of days flagged as having edge (outside neutral zone 0.45-0.55)
    # Higher coverage = less selective (more days flagged), Lower coverage = more selective (fewer days flagged)
    neutral_mask = (y_proba_rf_cal >= 0.45) & (y_proba_rf_cal <= 0.55)
    coverage = (1 - neutral_mask.mean()) * 100
    print(f"\nCoverage: {coverage:.1f}% of days flagged (outside neutral zone 0.45-0.55)")

    print("\n=== Feature Importance Analysis ===")
    print(feature_importance_df.to_string(index=False))

    # Error analysis for Random Forest
    analyze_errors(y_test, y_pred, X_test)

    # Print probability distributions and calibration
    print("\n=== Probability Outputs (P(up)) ===")
    print(f"Random Forest - Mean P(up): {y_proba_rf.mean():.3f}")
    print(f"Random Forest - P(up) range: [{y_proba_rf.min():.3f}, {y_proba_rf.max():.3f}]")
    print(f"Logistic Regression - Mean P(up): {y_proba_lr.mean():.3f}")
    print(f"Logistic Regression - P(up) range: [{y_proba_lr.min():.3f}, {y_proba_lr.max():.3f}]")
    
    print("\n=== Calibration Results (Brier Score - lower is better) ===")
    print(f"Random Forest - Raw: {brier_rf_raw:.4f}, Calibrated: {brier_rf_cal:.4f}")
    print(f"Logistic Regression - Raw: {brier_lr_raw:.4f}, Calibrated: {brier_lr_cal:.4f}")
    
    print("\n=== Uncertainty Estimation (Ensemble Variance) ===")
    print(f"Mean uncertainty (variance): {uncertainty.mean():.4f}")
    print(f"Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    print(f"Mean uncertainty (std): {uncertainty_std.mean():.4f}")
    print(f"Uncertainty std range: [{uncertainty_std.min():.4f}, {uncertainty_std.max():.4f}]")
    
    # Print expected return model results (Phase 2)
    print("\n=== Expected Return Model Results (Phase 2) ===")
    print("\nLinear Regression:")
    print(f"  MSE: {lr_mse:.6f}")
    print(f"  MAE: {lr_mae:.6f}")
    print(f"  R²:  {lr_r2:.4f}")
    
    print("\nGradient Boosting:")
    print(f"  MSE: {gb_mse:.6f}")
    print(f"  MAE: {gb_mae:.6f}")
    print(f"  R²:  {gb_r2:.4f}")
    
    # Conditional analysis: Performance on high-confidence days from Phase 1
    # This shows how Phase 2 (return models) performs conditioned on Phase 1 (bias model) confidence
    # High confidence threshold: P(up) > 0.6 (strong bias signal)
    high_conf_mask = y_proba_rf_cal > 0.6
    if high_conf_mask.sum() > 0:
        high_conf_count = high_conf_mask.sum()
        mean_realized_return_high_conf = y_return_test[high_conf_mask].mean()
        # MAE on high-confidence days only
        gb_mae_high_conf = mean_absolute_error(y_return_test[high_conf_mask], y_return_pred_gb[high_conf_mask])
        lr_mae_high_conf = mean_absolute_error(y_return_test[high_conf_mask], y_return_pred_lr[high_conf_mask])
        
        print("\n=== Conditional Analysis (High-Confidence Days: P(up) > 0.6) ===")
        print(f"High-confidence days: {high_conf_count} ({high_conf_count/len(y_test)*100:.1f}% of test set)")
        print(f"Mean realized return on high-confidence days: {mean_realized_return_high_conf:.4f}")
        print(f"Linear Regression MAE (high-confidence only): {lr_mae_high_conf:.6f}")
        print(f"Gradient Boosting MAE (high-confidence only): {gb_mae_high_conf:.6f}")

    # Optimize thresholds (bias threshold and EV threshold)
    # Tests different combinations to find best Sharpe ratio (including shorting)
    # 
    # CAVEAT: Threshold optimization may overfit to the test set. Results are used
    # to study sensitivity and relative performance, not to claim deployable alpha.
    transaction_cost = 0.001  # Fixed transaction cost (0.1%)
    # NOTE: Results shown incorporate transaction costs as a return threshold buffer.
    # A fixed cost can be incorporated as a return threshold buffer (done via EV threshold).
    alpha = 0.25  # Kelly fraction (25% of full Kelly)
    w_max = 1.0
    volatility = X_test['volatility'].values
    allow_shorting = True  # Enable shorting
    
    print("\n=== Optimizing Thresholds (Long-Short Strategy) ===")
    print("NOTE: Threshold optimization may overfit; results used for sensitivity analysis.")
    best_config, results_df = optimize_thresholds(
        y_proba_rf_cal, y_return_pred_gb, y_return_test, volatility,
        alpha=alpha, w_max=w_max, transaction_cost=transaction_cost,
        allow_shorting=allow_shorting
    )
    
    # Use optimized thresholds
    long_bias_threshold = best_config['long_bias_threshold']
    long_ev_threshold = best_config['long_ev_threshold']
    short_bias_threshold = best_config.get('short_bias_threshold', None)
    short_ev_threshold = best_config.get('short_ev_threshold', None)
    
    print(f"Best Configuration (by Sharpe Ratio):")
    print(f"  Long Bias Threshold: {long_bias_threshold:.2f} (P(up) > {long_bias_threshold:.2f})")
    print(f"  Long EV Threshold: {long_ev_threshold:.4f} (E[return] > {long_ev_threshold:.4f})")
    if short_bias_threshold is not None:
        print(f"  Short Bias Threshold: {short_bias_threshold:.2f} (P(up) < {short_bias_threshold:.2f})")
        print(f"  Short EV Threshold: {short_ev_threshold:.4f} (E[return] < {short_ev_threshold:.4f})")
    print(f"  Total Trades: {best_config['n_trades']} (Long: {best_config['n_long']}, Short: {best_config['n_short']})")
    print(f"  Sharpe Ratio: {best_config['sharpe_ratio']:.2f}")
    print(f"  Total Return: {best_config['total_return']:.2%}")
    print(f"  Max Drawdown: {best_config['max_drawdown']:.2%}")
    print(f"  Hit Rate: {best_config['hit_rate']:.2%}")
    
    if len(results_df) > 0:
        print(f"\nTop 5 Configurations:")
        top5 = results_df.head(5)[['long_bias_threshold', 'long_ev_threshold', 'short_bias_threshold', 'short_ev_threshold', 
                                    'n_long', 'n_short', 'sharpe_ratio', 'total_return']]
        print(top5.to_string(index=False))

    # Use optimized thresholds for trading (long-short strategy)
    # Long signals: P(up) > threshold AND E[return] > threshold
    long_bias_filter = y_proba_rf_cal > long_bias_threshold
    long_ev_filter = y_return_pred_gb > long_ev_threshold
    long_signal = long_bias_filter & long_ev_filter
    
    # Short signals: P(up) < threshold AND E[return] < threshold (negative returns)
    if short_bias_threshold is not None and short_ev_threshold is not None:
        short_bias_filter = y_proba_rf_cal < short_bias_threshold
        short_ev_filter = y_return_pred_gb < short_ev_threshold
        short_signal = short_bias_filter & short_ev_filter
    else:
        short_signal = np.zeros(len(y_return_test), dtype=bool)
    
    # Combined trade signal: long OR short (but not both on same day)
    trade_signal = long_signal | short_signal
    
    n_long = long_signal.sum()
    n_short = short_signal.sum()
    n_trades = trade_signal.sum()
    trade_percentage = (n_trades / len(trade_signal)) * 100

    # Calculate position sizes (positive for long, negative for short)
    # CAVEAT: We use a capped, fractional Kelly-style heuristic for risk-aware sizing,
    # not a true Kelly solution. This is a heuristic approximation, not optimal sizing.
    position_size = np.zeros(len(trade_signal))
    
    # Long positions: positive size
    # Kelly-style formula: w = α * E[return] / σ² (fractional Kelly with cap)
    position_size[long_signal] = alpha * y_return_pred_gb[long_signal] / (volatility[long_signal] ** 2)
    
    # Short positions: negative size (using absolute value of predicted return)
    if short_signal.sum() > 0:
        position_size[short_signal] = -alpha * np.abs(y_return_pred_gb[short_signal]) / (volatility[short_signal] ** 2)
    
    # Clip to [-w_max, w_max] to allow short positions
    position_size = np.clip(position_size, -w_max, w_max)
    
    # Backtest strategies (Phase 3, Step 3)
    # Compare 5 strategies:
    # 1. Buy and hold (always long)
    # 2. Always trade long (when P(up) > 0.5)
    # 3. Long filtered (when P(up) > threshold)
    # 4. Long + EV filtered (both conditions with position sizing) - LONG ONLY
    # 5. Long-Short optimized (both long and short with position sizing) - NEW WITH SHORTING
    
    # Strategy 1: Buy and hold (always long)
    buy_hold_signals = np.ones(len(y_return_test), dtype=bool)
    buy_hold_positions = np.ones(len(y_return_test))  # 100% position
    buy_hold_results = backtest_strategy(y_return_test, buy_hold_signals, buy_hold_positions)
    
    # Strategy 2: Always trade long (when P(up) > 0.5)
    always_long_signals = y_proba_rf_cal > 0.5
    always_long_positions = np.ones(len(y_return_test))
    always_long_positions[~always_long_signals] = 0  # No position when P(up) <= 0.5
    always_long_results = backtest_strategy(y_return_test, always_long_signals, always_long_positions)
    
    # Strategy 3: Long filtered (when P(up) > threshold, no EV filter)
    long_only_signals = long_bias_filter
    long_only_positions = np.ones(len(y_return_test))
    long_only_positions[~long_only_signals] = 0  # No position when filtered out
    long_only_results = backtest_strategy(y_return_test, long_only_signals, long_only_positions)
    
    # Strategy 4: Long + EV filtered (both conditions with position sizing) - LONG ONLY
    long_ev_positions = np.zeros(len(trade_signal))
    long_ev_positions[long_signal] = alpha * y_return_pred_gb[long_signal] / (volatility[long_signal] ** 2)
    long_ev_positions = np.clip(long_ev_positions, 0, w_max)  # No shorting
    long_ev_results = backtest_strategy(y_return_test, long_signal, long_ev_positions)
    
    # Strategy 5: Long-Short optimized (both long and short with position sizing)
    long_short_results = backtest_strategy(y_return_test, trade_signal, position_size)
    
    # Print strategy comparison results
    print("\n=== Strategy Comparison (Backtesting Results) ===")
    print("\n1. Buy and Hold (always long):")
    print(f"   Total Return: {buy_hold_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {buy_hold_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {buy_hold_results['max_drawdown']:.2%}")
    print(f"   Hit Rate: {buy_hold_results['hit_rate']:.2%}")
    print(f"   Turnover: {buy_hold_results['turnover']:.2f}")
    
    print("\n2. Always Trade Long (P(up) > 0.5):")
    print(f"   Total Return: {always_long_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {always_long_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {always_long_results['max_drawdown']:.2%}")
    print(f"   Hit Rate: {always_long_results['hit_rate']:.2%}")
    print(f"   Turnover: {always_long_results['turnover']:.2f}")
    
    print("\n3. Long Filtered (P(up) > {:.2f}, no EV filter):".format(long_bias_threshold))
    print(f"   Total Return: {long_only_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {long_only_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {long_only_results['max_drawdown']:.2%}")
    print(f"   Hit Rate: {long_only_results['hit_rate']:.2%}")
    print(f"   Turnover: {long_only_results['turnover']:.2f}")
    
    print("\n4. Long + EV Filtered (long only, with position sizing):")
    print(f"   Total Return: {long_ev_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {long_ev_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {long_ev_results['max_drawdown']:.2%}")
    print(f"   Hit Rate: {long_ev_results['hit_rate']:.2%}")
    print(f"   Turnover: {long_ev_results['turnover']:.2f}")
    print(f"   Number of Long Trades: {n_long} ({n_long/len(trade_signal)*100:.1f}% of days)")
    
    print("\n5. Long-Short Optimized (both long and short with position sizing):")
    print(f"   Total Return: {long_short_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {long_short_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {long_short_results['max_drawdown']:.2%}")
    print(f"   Hit Rate: {long_short_results['hit_rate']:.2%}")
    print(f"   Turnover: {long_short_results['turnover']:.2f}")
    print(f"   Total Trades: {n_trades} ({trade_percentage:.1f}% of days)")
    print(f"   Long Trades: {n_long} ({n_long/len(trade_signal)*100:.1f}% of days)")
    print(f"   Short Trades: {n_short} ({n_short/len(trade_signal)*100:.1f}% of days)")

    return model, lr_model, return_model_lr, return_model_gb        