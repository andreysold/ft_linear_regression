import argparse
from typing import List, Tuple

from linear_regression import LinearRegressionProtocol, LinearRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a linear regression model to predict car prices.')
    parser.add_argument('-d', '--dataset', type=str, default='data.csv',
                        help='Path to CSV file containing mileage and price data. Default=data.csv')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='Learning rate for gradient descent. Default=0.01)')
    parser.add_argument('-i', '--iterations', type=int, default=100000,
                        help='Number of training iterations. Default=100000')
    args = parser.parse_args()
    lr: LinearRegressionProtocol = LinearRegression(learning_rate=args.learning_rate, iterations=args.iterations)
    km_with_price: List[Tuple[float, float]] = lr.load_data(args.dataset)
    lr.train(km_price_list=km_with_price)
    lr.save_thetas("thetas.txt")
