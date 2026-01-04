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

def train_and_evaluate(stock, period):

    df = download_stock_data(stock, period)
    df = create_features(df)
    df = create_labels(df)

    # Note: All features at time t use only data from time <= t (no data leakage).
    # - Returns, moving averages, volatility, momentum all use historical data only
    # - Labels use next_day_return (shifted forward) but are computed correctly
    # - Train/test split preserves temporal order (shuffle=False)

    features = ['returns', 'ma5', 'ma20', 'volatility', 'momentum', 'rsi', 'price_to_ma', 'volume_ratio']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    rf_feature_importance = model.feature_importances_

    lr_coefficients = lr_model.coef_[0]

    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'RF_Importance': rf_feature_importance,
        'LR_Coefficient': lr_coefficients
    })

    feature_importance_df = feature_importance_df.sort_values('RF_Importance', ascending=False)

    y_baseline_always_up = np.ones(len(y_test))

    yesterday_returns = X_test['returns'].values
    y_baseline_yesterday = (yesterday_returns > 0).astype(int)

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

    print(f"\nBaseline (always up): {baseline_accuracy_1:.2%}")
    print(f"Baseline (yesterday's return): {baseline_accuracy_2:.2%}")

    print(f"\nAccuracy: {accuracy:.2%}")
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

    return model