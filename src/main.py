import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_model import train_and_evaluate, walk_forward_validation


if __name__ == "__main__":
    # Choose validation method:
    # - "single": Single train/test split (faster, for development)
    # - "walk_forward": Walk-forward validation (more realistic, addresses overfitting)
    validation_method = "walk_forward"  # Change to "single" for original behavior
    
    stocks = ['AAPL', 'MSFT', 'AMZN']

    for stock in stocks:
        print(f"\n{'='*50}")
        print(f"Testing {stock}")
        print(f"{'='*50}")
        
        if validation_method == "walk_forward":
            # Walk-forward validation: more realistic, addresses overfitting
            # Train on 70% initially, test on 10%, slide forward 10% each time
            results = walk_forward_validation(stock, "2y", train_size=0.7, test_size=0.1, step=0.1, verbose=True)
        else:
            # Original single split (80/20)
            prob_model_rf, prob_model_lr, return_model_lr, return_model_gb = train_and_evaluate(stock, "2y")
                
        print(f"\nCompleted {stock}\n")