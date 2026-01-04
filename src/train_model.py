from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import numpy as np

from src.data_loader import download_stock_data
from src.feature_engineering import create_features, create_labels

def train_and_evaluate(stock, period):

    df = download_stock_data(stock, period)
    df = create_features(df)
    df = create_labels(df)

    features = ['returns', 'ma5', 'ma20', 'volatility', 'momentum']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_baseline_always_up = np.ones(len(y_test))
    
    yesterday_returns = X_test['returns'].values
    y_baseline_yesterday = (yesterday_returns > 0).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    
    baseline_accuracy_1 = accuracy_score(y_test, y_baseline_always_up)
    baseline_accuracy_2 = accuracy_score(y_test, y_baseline_yesterday)

    precision = precision_score(y_test, y_pred)
    
    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    confusion = confusion_matrix(y_test, y_pred)

    print(f"Baseline (always up): {baseline_accuracy_1:.2%}")
    print(f"Baseline (yesterday's return): {baseline_accuracy_2:.2%}")

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1: {f1:.2%}")
    print(f"Confusion Matrix:\n{confusion}")

    return model