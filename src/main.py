import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_model import train_and_evaluate


if __name__ == "__main__":
    print("Starting...")

    stock = str(input("Enter Stock: (E.g: AAPL) "))
    period = str(input("Enter Timeframe: (E.g: 2y) "))
    result = train_and_evaluate(stock, period)

    print("Done!")