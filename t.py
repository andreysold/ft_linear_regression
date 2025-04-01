def train_model(data_path, learning_rate=0.0001, epochs=10000):
    # Read and parse data
    with open(data_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    data = []
    for line in lines:
        km, price = map(float, line.strip().split(','))
        data.append((km, price))

    # Normalize mileage and price (min-max scaling)
    mileages = [x[0] for x in data]
    prices = [x[1] for x in data]
    min_km, max_km = min(mileages), max(mileages)
    min_price, max_price = min(prices), max(prices)

    normalized_data = []
    for km, price in data:
        norm_km = (km - min_km) / (max_km - min_km)
        norm_price = (price - min_price) / (max_price - min_price)
        normalized_data.append((norm_km, norm_price))

    # Initialize thetas (small random values)
    theta0, theta1 = 1.0, 0.1
    m = len(normalized_data)

    # Gradient Descent
    for _ in range(epochs):
        sum_error0 = 0.0
        sum_error1 = 0.0
        for km, price in normalized_data:
            prediction = theta0 + (theta1 * km)
            error = prediction - price
            sum_error0 += error
            sum_error1 += error * km

        theta0 -= (learning_rate * (1 / m) * sum_error0)
        theta1 -= (learning_rate * (1 / m) * sum_error1)

    # Rescale thetas back to original scale
    theta0 = theta0 * (max_price - min_price) + min_price - theta1 * min_km * (max_price - min_price) / (
                max_km - min_km)
    theta1 = theta1 * (max_price - min_price) / (max_km - min_km)

    with open('thetas.txt', 'w') as f:
        f.write(f"{theta0},{theta1}")
    print(f"Training complete. Thetas: theta0={theta0:.2f}, theta1={theta1:.6f}")


if __name__ == "__main__":
    train_model("data.csv")