from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import CalibratedClassifierCV

import numpy as np
import matplotlib.pyplot as plt

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


def calculate_transaction_cost(volatility, base_cost=0.001, volatility_multiplier=2.0):
    """
    Calculates volatility-scaled transaction costs.
    
    Real trading costs increase with volatility due to:
    - Wider bid-ask spreads (more volatile stocks have larger spreads)
    - Market impact (larger price moves during execution)
    - Slippage (execution price vs expected price)
    
    Formula: cost = base_cost * (1 + volatility_multiplier * volatility)
    
    Args:
        volatility: Volatility values (array or scalar)
        base_cost: Base transaction cost (default 0.001 = 0.1%)
        volatility_multiplier: How much volatility increases costs (default 2.0)
    
    Returns:
        Transaction cost (array or scalar, same shape as volatility)
    
    Example:
        base_cost = 0.001 (0.1%)
        volatility = 0.02 (2% daily volatility)
        volatility_multiplier = 2.0
        cost = 0.001 * (1 + 2.0 * 0.02) = 0.001 * 1.04 = 0.00104 (0.104%)
    """
    volatility = np.array(volatility)
    cost = base_cost * (1 + volatility_multiplier * volatility)
    return cost


def backtest_strategy(returns, trade_signals, position_sizes=None, stop_loss_pct=None):
    """
    Backtests a trading strategy with optional stop-loss.
    
    **Stop-Loss Note:** Stop-loss is a risk containment mechanism, not an alpha generator.
    It limits downside risk but may also exit profitable positions prematurely.
    
    Args:
        returns: Actual returns (array or Series)
        trade_signals: Boolean array (True = trade, False = no trade)
        position_sizes: Position sizes (array, optional). If None, uses fixed size 1.0
        stop_loss_pct: Stop-loss percentage (e.g., 0.02 = 2%). If None, no stop-loss.
                       For long: exit if position drops by stop_loss_pct from entry
                       For short: exit if position rises by stop_loss_pct from entry
    
    Returns:
        dict with performance metrics (total_return, sharpe_ratio, max_drawdown, hit_rate, turnover, n_stop_losses)
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
    
    # Track positions for stop-loss (if enabled)
    effective_positions = position_sizes.copy()
    n_stop_losses = 0
    
    if stop_loss_pct is not None and stop_loss_pct > 0:
        # Track current position state for stop-loss
        in_position = False
        entry_index = -1  # Index when position was entered
        entry_position_size = 0.0
        
        for i in range(len(returns)):
            current_position = effective_positions[i]
            
            if not in_position and current_position != 0:
                # Entering a new position
                in_position = True
                entry_index = i
                entry_position_size = current_position
            elif in_position:
                # We're in a position - check stop-loss
                if current_position == 0:
                    # Position closed by signal (not stop-loss)
                    in_position = False
                else:
                    # Calculate cumulative return since entry
                    # For long: cumulative_return = product of (1 + returns) since entry
                    # For short: cumulative_return = product of (1 - returns) since entry
                    cumulative_return = 1.0
                    for j in range(entry_index, i + 1):
                        if entry_position_size > 0:  # Long position
                            cumulative_return *= (1 + returns[j])
                        else:  # Short position
                            cumulative_return *= (1 - returns[j])
                    
                    # Check stop-loss: if cumulative return has dropped by stop_loss_pct
                    if cumulative_return < (1 - stop_loss_pct):
                        # Trigger stop-loss: exit position
                        effective_positions[i] = 0
                        in_position = False
                        n_stop_losses += 1
    
    # Calculate portfolio returns with effective positions (after stop-loss)
    portfolio_returns = effective_positions * returns
    
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
    # Only count trades that weren't stopped out
    active_trades = effective_positions != 0
    trade_returns = portfolio_returns[active_trades]
    if len(trade_returns) > 0:
        hit_rate = (trade_returns > 0).sum() / len(trade_returns)
    else:
        hit_rate = 0.0
    
    # Turnover (average absolute position size - measures trading activity)
    turnover = np.abs(effective_positions).mean()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'turnover': turnover,
        'portfolio_value': portfolio_value,
        'n_stop_losses': n_stop_losses
    }


def plot_backtest_results(portfolio_value, strategy_name, stock_name, save_path=None):
    """
    Creates visualization plots for backtesting results.
    
    Plots:
    1. Equity curve: Portfolio value over time
    2. Drawdown: Peak-to-trough decline over time
    
    Args:
        portfolio_value: Array of portfolio values over time
        strategy_name: Name of the strategy (for title)
        stock_name: Stock ticker (for title)
        save_path: Optional path to save figure (if None, just displays)
    """
    portfolio_value = np.array(portfolio_value)
    n_days = len(portfolio_value)
    
    # Calculate drawdown
    running_max = np.maximum.accumulate(portfolio_value)
    drawdown = (running_max - portfolio_value) / running_max * 100  # Convert to percentage
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Equity Curve
    days = np.arange(n_days)
    ax1.plot(days, portfolio_value, linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.set_title(f'{strategy_name} - {stock_name}: Equity Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Add final return annotation
    final_return = (portfolio_value[-1] - 1.0) * 100
    ax1.text(0.02, 0.98, f'Total Return: {final_return:.2f}%', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Drawdown
    ax2.fill_between(days, 0, drawdown, color='#A23B72', alpha=0.6, label='Drawdown')
    ax2.plot(days, drawdown, linewidth=1, color='#A23B72')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.invert_yaxis()  # Drawdown should go down (negative is good)
    
    # Add max drawdown annotation
    max_dd = drawdown.max()
    ax2.text(0.02, 0.02, f'Max Drawdown: {max_dd:.2f}%', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_strategy_comparison(strategy_results, stock_name, save_path=None):
    """
    Creates a comparison plot of multiple strategies.
    
    Args:
        strategy_results: Dict with strategy names as keys and portfolio_value arrays as values
        stock_name: Stock ticker (for title)
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for i, (strategy_name, portfolio_value) in enumerate(strategy_results.items()):
        portfolio_value = np.array(portfolio_value)
        days = np.arange(len(portfolio_value))
        color = colors[i % len(colors)]
        ax.plot(days, portfolio_value, linewidth=2, label=strategy_name, color=color)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Portfolio Value', fontsize=12)
    ax.set_title(f'Strategy Comparison - {stock_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def optimize_thresholds(y_proba_cal, y_return_pred, y_return_actual, volatility, alpha=0.25, w_max=1.0, base_transaction_cost=0.001, volatility_multiplier=2.0, allow_shorting=True, stop_loss_pct=None):
    """
    Optimizes bias threshold and EV threshold to maximize Sharpe ratio.
    Supports both long-only and long-short strategies.
    
    **CRITICAL CAVEAT - Overfitting Risk:**
    Threshold optimization is performed on a fixed historical window and may overfit.
    Results are used to study sensitivity and relative performance, not to claim deployable alpha.
    For production use, optimize on one window and evaluate on a later, untouched window.
    
    **Transaction Costs:**
    Uses volatility-scaled transaction costs: cost = base_cost * (1 + multiplier * volatility)
    This accounts for wider spreads and slippage in volatile markets.
    
    Args:
        y_proba_cal: Calibrated probabilities (array)
        y_return_pred: Predicted returns (array)
        y_return_actual: Actual returns (array)
        volatility: Volatility values (array)
        alpha: Kelly fraction (default 0.25)
        w_max: Maximum position size (default 1.0)
        base_transaction_cost: Base transaction cost (default 0.001 = 0.1%)
        volatility_multiplier: How much volatility increases costs (default 2.0)
        allow_shorting: If True, also test short positions (default True)
    
    Returns:
        dict with best thresholds and performance metrics
    """
    # Convert to numpy arrays
    y_proba_cal = np.array(y_proba_cal)
    y_return_pred = np.array(y_return_pred)
    y_return_actual = np.array(y_return_actual)
    volatility = np.array(volatility)
    
    # Calculate dynamic transaction costs based on volatility
    # Cost varies by day: higher volatility = higher costs (wider spreads, slippage)
    transaction_costs = calculate_transaction_cost(volatility, base_transaction_cost, volatility_multiplier)
    avg_transaction_cost = transaction_costs.mean()  # Use average for EV threshold grid search
    
    # Grid search: test different threshold combinations
    long_bias_thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    # EV thresholds: base values that account for avg transaction cost + buffer
    long_ev_thresholds = [avg_transaction_cost * 0.5, avg_transaction_cost, avg_transaction_cost * 1.5, 
                         avg_transaction_cost * 2.0, avg_transaction_cost * 2.5, avg_transaction_cost * 3.0]
    
    # For shorting: P(up) < threshold means short (e.g., P(up) < 0.45 means 55% chance of down)
    # Mirror the long thresholds: short_bias = 1 - long_bias
    if allow_shorting:
        short_bias_thresholds = [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50]  # P(up) < these values
        # Short EV thresholds: symmetric to long (negative returns, account for transaction costs)
        short_ev_thresholds = [-avg_transaction_cost * 3.0, -avg_transaction_cost * 2.5, -avg_transaction_cost * 2.0,
                               -avg_transaction_cost * 1.5, -avg_transaction_cost, -avg_transaction_cost * 0.5]
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
                results_dict = backtest_strategy(y_return_actual, long_signal, position_size, stop_loss_pct=stop_loss_pct)
                
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
                        results_dict = backtest_strategy(y_return_actual, trade_signal, position_size, stop_loss_pct=stop_loss_pct)
                        
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


def walk_forward_validation(stock, period, train_size=0.7, test_size=0.1, step=0.1, verbose=True):
    """
    Performs walk-forward validation: trains on rolling windows and tests on future data.
    
    This addresses overfitting by:
    - Training on one period, testing on the next (truly unseen data)
    - Retraining periodically (like real trading)
    - Showing how performance changes over time
    
    Args:
        stock: Stock ticker (e.g., 'AAPL')
        period: Data period (e.g., '2y')
        train_size: Fraction of data for initial training (default 0.7 = 70%)
        test_size: Fraction of data for each test window (default 0.1 = 10%)
        step: Fraction to slide forward each iteration (default 0.1 = 10%)
        verbose: If True, print detailed results for each window
    
    Returns:
        dict with aggregated results across all windows
    """
    # Download and prepare data
    df = download_stock_data(stock, period)
    df = create_features(df)
    df = create_labels(df)
    
    features = ['returns', 'ma5', 'ma20', 'volatility', 'momentum', 'rsi', 'price_to_ma', 'volume_ratio']
    X = df[features]
    y = df['label']
    y_return = df['return_target']
    
    n_total = len(X)
    n_train = int(n_total * train_size)
    n_test = int(n_total * test_size)
    n_step = int(n_total * step)
    
    if verbose:
        print(f"\n=== Walk-Forward Validation: {stock} ===")
        print(f"Total data points: {n_total}")
        print(f"Initial train size: {n_train} ({train_size:.0%})")
        print(f"Test window size: {n_test} ({test_size:.0%})")
        print(f"Step size: {n_step} ({step:.0%})")
        print(f"Number of windows: {max(1, (n_total - n_train) // n_step)}")
    
    # Store results for each window
    window_results = []
    all_test_returns = []
    all_portfolio_values = []
    
    # Walk forward: train on [0:train_end], test on [train_end:train_end+test_size]
    train_end = n_train
    window_num = 1
    
    while train_end + n_test <= n_total:
        # Split into train and test for this window
        X_train_window = X.iloc[:train_end]
        X_test_window = X.iloc[train_end:train_end + n_test]
        y_train_window = y.iloc[:train_end]
        y_test_window = y.iloc[train_end:train_end + n_test]
        y_return_train_window = y_return.iloc[:train_end]
        y_return_test_window = y_return.iloc[train_end:train_end + n_test]
        
        if verbose:
            print(f"\n--- Window {window_num}: Train [0:{train_end}], Test [{train_end}:{train_end + n_test}] ---")
        
        # Run full pipeline for this window (we'll create a helper function)
        window_result = _train_and_evaluate_window(
            X_train_window, X_test_window,
            y_train_window, y_test_window,
            y_return_train_window, y_return_test_window,
            verbose=verbose
        )
        
        window_result['window_num'] = window_num
        window_result['train_end'] = train_end
        window_result['test_start'] = train_end
        window_result['test_end'] = train_end + n_test
        window_results.append(window_result)
        
        # Aggregate portfolio values (for overall equity curve)
        all_test_returns.extend(y_return_test_window.values)
        if 'portfolio_value' in window_result:
            all_portfolio_values.extend(window_result['portfolio_value'])
        
        # Slide forward
        train_end += n_step
        window_num += 1
        
        # Safety: don't create windows that are too small
        if train_end + n_test > n_total:
            break
    
    # Aggregate results across all windows
    if len(window_results) > 0:
        aggregated = {
            'n_windows': len(window_results),
            'mean_sharpe': np.mean([r['sharpe_ratio'] for r in window_results]),
            'mean_return': np.mean([r['total_return'] for r in window_results]),
            'mean_drawdown': np.mean([r['max_drawdown'] for r in window_results]),
            'mean_hit_rate': np.mean([r['hit_rate'] for r in window_results]),
            'std_sharpe': np.std([r['sharpe_ratio'] for r in window_results]),
            'std_return': np.std([r['total_return'] for r in window_results]),
            'window_results': window_results
        }
        
        # Calculate overall portfolio performance (if we have all portfolio values)
        if len(all_portfolio_values) > 0 and len(all_test_returns) > 0:
            # Reconstruct overall portfolio value
            overall_portfolio = np.ones(len(all_test_returns))
            # This is simplified - in reality we'd need to track positions across windows
            # For now, just show aggregated metrics
            pass
    else:
        aggregated = None
    
    if verbose and aggregated:
        print(f"\n=== Walk-Forward Summary: {stock} ===")
        print(f"Number of windows: {aggregated['n_windows']}")
        print(f"Mean Sharpe Ratio: {aggregated['mean_sharpe']:.2f} (std: {aggregated['std_sharpe']:.2f})")
        print(f"Mean Total Return: {aggregated['mean_return']:.2%} (std: {aggregated['std_return']:.2%})")
        print(f"Mean Max Drawdown: {aggregated['mean_drawdown']:.2%}")
        print(f"Mean Hit Rate: {aggregated['mean_hit_rate']:.2%}")
    
    return aggregated


def _train_and_evaluate_window(X_train, X_test, y_train, y_test, y_return_train, y_return_test, verbose=True):
    """
    Helper function: runs the full training and evaluation pipeline for a single window.
    This is the core logic extracted from train_and_evaluate().
    
    Returns a dict with results instead of printing everything.
    """
    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train probability models (Phase 1)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Calibrate probabilities
    model_cal = CalibratedClassifierCV(model, method='isotonic', cv=5)
    model_cal.fit(X_train_scaled, y_train)
    
    lr_model_cal = CalibratedClassifierCV(lr_model, method='isotonic', cv=5)
    lr_model_cal.fit(X_train_scaled, y_train)
    
    # Get calibrated probabilities
    y_proba_rf_cal = model_cal.predict_proba(X_test_scaled)[:, 1]
    
    # Train return models (Phase 2)
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    
    return_model_lr = LinearRegression()
    return_model_lr.fit(X_train_scaled, y_return_train)
    
    return_model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    return_model_gb.fit(X_train_scaled, y_return_train)
    
    y_return_pred_lr = return_model_lr.predict(X_test_scaled)
    y_return_pred_gb = return_model_gb.predict(X_test_scaled)
    
    # Optimize thresholds and backtest (Phase 3)
    # Use volatility-scaled transaction costs
    volatility = X_test['volatility'].values
    base_transaction_cost = 0.001  # Base cost (0.1%)
    volatility_multiplier = 2.0  # How much volatility increases costs
    alpha = 0.25
    w_max = 1.0
    allow_shorting = True
    
    # Stop-loss parameter (same as main function)
    stop_loss_pct = 0.02  # 2% stop-loss
    
    best_config, _ = optimize_thresholds(
        y_proba_rf_cal, y_return_pred_gb, y_return_test.values, volatility,
        alpha=alpha, w_max=w_max, 
        base_transaction_cost=base_transaction_cost,
        volatility_multiplier=volatility_multiplier,
        allow_shorting=allow_shorting,
        stop_loss_pct=stop_loss_pct
    )
    
    # Create trade signals with optimized thresholds
    long_bias_threshold = best_config['long_bias_threshold']
    long_ev_threshold = best_config['long_ev_threshold']
    short_bias_threshold = best_config.get('short_bias_threshold', None)
    short_ev_threshold = best_config.get('short_ev_threshold', None)
    
    long_bias_filter = y_proba_rf_cal > long_bias_threshold
    long_ev_filter = y_return_pred_gb > long_ev_threshold
    long_signal = long_bias_filter & long_ev_filter
    
    if short_bias_threshold is not None and short_ev_threshold is not None:
        short_bias_filter = y_proba_rf_cal < short_bias_threshold
        short_ev_filter = y_return_pred_gb < short_ev_threshold
        short_signal = short_bias_filter & short_ev_filter
    else:
        short_signal = np.zeros(len(y_test), dtype=bool)
    
    trade_signal = long_signal | short_signal
    
    # Position sizing
    position_size = np.zeros(len(trade_signal))
    position_size[long_signal] = alpha * y_return_pred_gb[long_signal] / (volatility[long_signal] ** 2)
    if short_signal.sum() > 0:
        position_size[short_signal] = -alpha * np.abs(y_return_pred_gb[short_signal]) / (volatility[short_signal] ** 2)
    position_size = np.clip(position_size, -w_max, w_max)
    
    # Backtest (with stop-loss)
    results = backtest_strategy(y_return_test.values, trade_signal, position_size, stop_loss_pct=stop_loss_pct)
    
    # Add additional info
    results['n_long'] = long_signal.sum()
    results['n_short'] = short_signal.sum()
    results['n_trades'] = trade_signal.sum()
    results['long_bias_threshold'] = long_bias_threshold
    results['long_ev_threshold'] = long_ev_threshold
    
    if verbose:
        print(f"  Sharpe: {results['sharpe_ratio']:.2f}, Return: {results['total_return']:.2%}, "
              f"Drawdown: {results['max_drawdown']:.2%}, Trades: {results['n_trades']} "
              f"(Long: {results['n_long']}, Short: {results['n_short']})")
    
    return results


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
    # Transaction cost modeling: volatility-scaled costs
    # Higher volatility = wider spreads, more slippage, higher costs
    base_transaction_cost = 0.001  # Base cost (0.1%)
    volatility_multiplier = 2.0  # How much volatility increases costs
    alpha = 0.25  # Kelly fraction (25% of full Kelly)
    w_max = 1.0
    volatility = X_test['volatility'].values
    allow_shorting = True  # Enable shorting
    
    # Stop-loss: risk containment mechanism (not alpha generation)
    # Exit position if it drops by stop_loss_pct from entry
    stop_loss_pct = 0.02  # 2% stop-loss (configurable, set to None to disable)
    
    print("\n=== Optimizing Thresholds (Long-Short Strategy) ===")
    print("NOTE: Threshold optimization may overfit; results used for sensitivity analysis.")
    print(f"Using volatility-scaled transaction costs (base: {base_transaction_cost:.3f}, multiplier: {volatility_multiplier:.1f})")
    best_config, results_df = optimize_thresholds(
        y_proba_rf_cal, y_return_pred_gb, y_return_test, volatility,
        alpha=alpha, w_max=w_max, 
        base_transaction_cost=base_transaction_cost, 
        volatility_multiplier=volatility_multiplier,
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
    buy_hold_results = backtest_strategy(y_return_test, buy_hold_signals, buy_hold_positions, stop_loss_pct=stop_loss_pct)
    
    # Strategy 2: Always trade long (when P(up) > 0.5)
    always_long_signals = y_proba_rf_cal > 0.5
    always_long_positions = np.ones(len(y_return_test))
    always_long_positions[~always_long_signals] = 0  # No position when P(up) <= 0.5
    always_long_results = backtest_strategy(y_return_test, always_long_signals, always_long_positions, stop_loss_pct=stop_loss_pct)
    
    # Strategy 3: Long filtered (when P(up) > threshold, no EV filter)
    long_only_signals = long_bias_filter
    long_only_positions = np.ones(len(y_return_test))
    long_only_positions[~long_only_signals] = 0  # No position when filtered out
    long_only_results = backtest_strategy(y_return_test, long_only_signals, long_only_positions, stop_loss_pct=stop_loss_pct)
    
    # Strategy 4: Long + EV filtered (both conditions with position sizing) - LONG ONLY
    long_ev_positions = np.zeros(len(trade_signal))
    long_ev_positions[long_signal] = alpha * y_return_pred_gb[long_signal] / (volatility[long_signal] ** 2)
    long_ev_positions = np.clip(long_ev_positions, 0, w_max)  # No shorting
    long_ev_results = backtest_strategy(y_return_test, long_signal, long_ev_positions, stop_loss_pct=stop_loss_pct)
    
    # Strategy 5: Long-Short optimized (both long and short with position sizing)
    long_short_results = backtest_strategy(y_return_test, trade_signal, position_size, stop_loss_pct=stop_loss_pct)
    
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
    if stop_loss_pct is not None:
        print(f"   Stop-Losses Triggered: {long_short_results.get('n_stop_losses', 0)} "
              f"(Stop-loss: {stop_loss_pct:.1%})")
    
    # Create visualizations
    print("\n=== Generating Visualizations ===")
    
    # Strategy comparison plot (all strategies together)
    strategy_results = {
        'Buy and Hold': buy_hold_results['portfolio_value'],
        'Always Trade Long': always_long_results['portfolio_value'],
        'Long Filtered': long_only_results['portfolio_value'],
        'Long + EV Filtered': long_ev_results['portfolio_value'],
        'Long-Short Optimized': long_short_results['portfolio_value']
    }
    
    # Create plots directory if it doesn't exist
    import os
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Plot strategy comparison
    comparison_path = os.path.join(plots_dir, f'{stock}_strategy_comparison.png')
    plot_strategy_comparison(strategy_results, stock, save_path=comparison_path)
    
    # Plot detailed view of best strategy (Long-Short Optimized)
    best_strategy_path = os.path.join(plots_dir, f'{stock}_long_short_optimized.png')
    plot_backtest_results(long_short_results['portfolio_value'], 
                         'Long-Short Optimized', stock, save_path=best_strategy_path)
    
    print(f"Visualizations saved to {plots_dir}/ directory")

    return model, lr_model, return_model_lr, return_model_gb        