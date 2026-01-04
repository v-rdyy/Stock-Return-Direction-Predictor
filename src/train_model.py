from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Train Logistic Regression for comparison
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

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

    print("\n=== Baseline Models ===")
    print(f"Always predict up: {baseline_accuracy_1:.2%}")
    print(f"Predict based on yesterday's return: {baseline_accuracy_2:.2%}")

    print("\n=== Random Forest Results ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1: {f1:.2%}")
    print(f"Confusion Matrix:\n{confusion}")

    print("\n=== Logistic Regression Results ===")
    print(f"Accuracy: {lr_accuracy:.2%}")
    print(f"Precision: {lr_precision:.2%}")
    print(f"Recall: {lr_recall:.2%}")
    print(f"F1: {lr_f1:.2%}")
    print(f"Confusion Matrix:\n{lr_confusion}")

    print("\n=== Model Comparison ===")
    print(f"Random Forest Accuracy: {accuracy:.2%}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2%}")

    print("\n=== Feature Importance Analysis ===")
    print(feature_importance_df.to_string(index=False))

    # Error analysis for Random Forest
    analyze_errors(y_test, y_pred, X_test)

    return model, lr_model
        