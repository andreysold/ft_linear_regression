from typing import List, Tuple, Self, Protocol


class LinearRegressionProtocol(Protocol):
    """
    Interface linear regression.
    """

    def load_data(self: Self, file_name: str = "data.csv") -> List[Tuple[float, float]]: ...

    def normalize(self: Self, value: float, min_val: float, max_val: float) -> float: ...

    def denormalize(self: Self, value: float, min_val: float, max_val: float) -> float: ...

    def train(self: Self, km_price_list: List[Tuple[float, float]]) -> None: ...

    def save_thetas(self: Self, filename: str = "thetas.txt"): ...

    @staticmethod
    def read_thetas(file_name: str = "thetas.txt") -> Tuple[float, float]: ...


class LinearRegression(LinearRegressionProtocol):
    """
    Implementation interface of linear regression.
    """

    def __init__(
        self: Self,
        theta0: float = 0.0,
        theta1: float = 0.0,
        min_km: int = 0,
        max_km: int = 0,
        min_price: int = 0,
        max_price: int = 0,
        learning_rate: float = 0.01,
        iterations: int = 100000,
    ):
        self.theta0: float = theta0
        self.theta1: float = theta1
        self.min_km: int = min_km
        self.max_km: int = max_km
        self.min_price: int = min_price
        self.max_price: int = max_price
        self.learning_rate = learning_rate
        self.iterations = iterations

    def load_data(self: Self, file_name: str = "data.csv") -> List[Tuple[float, float]]:
        """
        Load data from file_name.
        :param file_name: file name
        :return: List[Tuple[float, float]]
        """

        km_price_list: List[Tuple[float, float]] = []
        try:
            with open(file_name, "r+") as f:
                f.readline()
                for line in f:
                    km, price = map(float, line.strip().split(','))
                    km_price_list.append((km, price))
        except FileNotFoundError:
            raise ValueError(f"File = {file_name} doesn't exist.")
        return km_price_list

    def normalize(self: Self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize value.
        :param value: float
        :param min_val: float
        :param max_val: float
        :return: float
        """

        return (value - min_val) / (max_val - min_val)

    def denormalize(self: Self, value: float, min_val: float, max_val: float) -> float:
        """
        Denormalize value.
        :param value: float
        :param min_val: float
        :param max_val: float
        :return: float
        """

        return value * (max_val - min_val) + min_val

    def train(self: Self, km_price_list: List[Tuple[float, float]]) -> None:
        """
        Train model.
        :param km_price_list: List[Tuple[float, float]]
        :param learning_rate: float. default = 0.01
        :param iterations: int. default = 10000
        :return: None
        """

        mileages: List[float] = [x[0] for x in km_price_list]
        prices: List[float] = [x[1] for x in km_price_list]
        self.min_km, self.max_km = min(mileages), max(mileages)
        self.min_price, self.max_price = min(prices), max(prices)
        norm_km_price_list: List[Tuple[float, float]] = [
            (self.normalize(km, self.min_km, self.max_km),
             self.normalize(price, self.min_price, self.max_price))
            for km, price in km_price_list
        ]

        m: int = len(norm_km_price_list)
        for _ in range(self.iterations):
            temp0: float = 0.0
            temp1: float = 0.0

            for km, price in norm_km_price_list:
                prediction: float = self.theta0 + self.theta1 * km
                error: float = prediction - price
                temp0 += error
                temp1 += error * km

            self.theta0 -= (self.learning_rate / m) * temp0
            self.theta1 -= (self.learning_rate / m) * temp1

        self.theta0: float = self.denormalize(
            self.theta0,
            self.min_price,
            self.max_price
        ) - self.theta1 * self.min_km * (self.max_price - self.min_price) / (self.max_km - self.min_km)
        self.theta1 = self.theta1 * (self.max_price - self.min_price) / (self.max_km - self.min_km)

    def save_thetas(self: Self, filename: str = "thetas.txt"):
        """
        Save thetas to file after train model.
        :param filename: str
        :return: None
        """

        with open(file=filename, mode='w') as f:
            f.write(f"{self.theta0},{self.theta1}")

    @staticmethod
    def read_thetas(file_name: str = "thetas.txt") -> Tuple[float, float]:
        """
        Read thetas from file after train model.
        :param file_name: str. default = thetas.txt
        :return: Tuple[float, float]
        """

        try:
            with open(file=file_name, mode="r+") as f:
                thetas = f.readline().split(",")
        except FileNotFoundError:
            return 0.0, 0.0
        return float(thetas[0]), float(thetas[1])
