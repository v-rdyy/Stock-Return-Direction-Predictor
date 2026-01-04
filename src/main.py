import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_model import train_and_evaluate


if __name__ == "__main__":

    stocks = ['AAPL', 'MSFT', 'AMZN']

    for stock in stocks:
        print(f"\n{'='*50}")
        print(f"Testing {stock}")
        print(f"{'='*50}")
        
        result = train_and_evaluate(stock, "2y")
        
        print(f"\nCompleted {stock}\n")