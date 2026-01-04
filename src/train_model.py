from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2%}")

    return model