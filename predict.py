from typing import Tuple

from train import LinearRegression as Lr


def predict(theta0: float, theta1: float, m: float) -> float:
    return theta0 + theta1 * m


if __name__ == "__main__":
    """
    First program. Predict script.
    """

    try:
        mileage: float = float(input("Enter mileage in km:"))
    except ValueError:
        raise ValueError("You should send float type!")
    if mileage < 0:
        raise ValueError("Mileage can't be negative!")
    thetas: Tuple[float, float] = Lr.read_thetas()
    predicted_price: float = predict(theta0=thetas[0], theta1=thetas[1], m=mileage)
    print(f"Predicted price for {mileage} km: ${predicted_price:.2f}")
