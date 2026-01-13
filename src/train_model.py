from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
    y = df['label']

    # 80/20 split, no shuffling (time-series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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

    return model, lr_model        