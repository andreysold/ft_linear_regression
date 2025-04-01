from typing import Protocol, Self, List


class PredictServiceProtocol(Protocol):
    """
    Interface for predict result by theta0 and theta1
    """

    def train(self: Self):
        ...

    def load_thetas(self: Self) -> None:
        ...

    # def estimate_price(self: Self, mileage: float) -> float:
    #     ...

    def load_dataset(self: Self) -> None:
        ...


class PredictService(PredictServiceProtocol):
    """
    Implementation for interface PredictProtocol.
    """

    def __init__(
        self: Self,
        theta0: float = 0.0,
        theta1: float = 0.0,
        max_iteration: int = 1000,
        learning_rate: float = 0.1
    ):
        self.theta0 = theta0
        self.theta1 = theta1
        self.data = []
        self.normalized_km = []
        self.normalized_price = []
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate
        self.max_km = 0
        self.max_price = 0
        self.min_km = 0
        self.min_price = 0

    def estimate_price2(self: Self, mileage: float) -> float:
        return self.theta0 + (self.theta1 * mileage)

    def estimate_price(self, theta0, theta1, x):
        return theta0 + theta1 * x

    def load_thetas(self: Self) -> None:
        """
        Set theta0/1 from thetas_data.txt or set zero to theta0/1
        :return: None
        """

        theta0: float = 0.0
        theta1: float = 0.0
        try:
            with open("thetas.txt", "r+") as f:
                thetas: list[str] = f.readline().split(",")
                theta0: float = float(thetas[0])
                theta1: float = float(thetas[1])
        except FileNotFoundError:
            ...
        finally:
            self.theta0 = theta0
            self.theta1 = theta1

    def load_dataset(self: Self) -> None:
        """
        Load mileage and cost from dataset.csv
        :return: None
        """

        with open("data.csv", "r+") as f:
            lines = f.readlines()[1:]
        data: List[tuple] = []
        for line in lines:
            km, price = map(float, line.strip().split(","))
            data.append((km, price))
        self.data = data

    def normalize(self: Self) -> None:
        """
        Normalize features from data.csv
        :return: None
        """

        mileages = [x[0] for x in self.data]
        prices = [x[1] for x in self.data]
        min_km, max_km = min(mileages), max(mileages)
        min_price, max_price = min(prices), max(prices)

        self.min_km = min_km
        self.max_km = max_km
        self.min_price = min_price
        self.max_price = max_price

        for km, price in self.data:
            norm_km = (km - min_km) / (max_km - min_km)
            norm_price = (price - min_price) / (max_price - min_price)
            self.normalized_km.append(norm_km)
            self.normalized_price.append(norm_price)

    def fit(self: Self) -> None:
        """
        Fit
        :return: None
        """

        m = len(self.normalized_km)

        for _ in range(self.max_iteration):
            tmp_theta0 = 0
            tmp_theta1 = 0
            for km, price in zip(self.normalized_km, self.normalized_price):
                tmp_theta0 += self.estimate_price(self.theta0, self.theta1, km) - price
                tmp_theta1 += (self.estimate_price(self.theta0, self.theta1, km) - price) * price
            self.theta0 -= (self.learning_rate / m) * tmp_theta0
            self.theta1 -= (self.learning_rate / m) * tmp_theta1
        self.theta0 = self.theta0 * (self.max_price - self.min_price) + self.min_price - self.theta1 * self.min_km * (self.max_price - self.min_price) / (
                    self.max_km - self.min_km)
        self.theta1 = self.theta1 * (self.max_price - self.min_price) / (self.max_km - self.min_km)
