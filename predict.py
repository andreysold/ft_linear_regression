from predict_service import PredictService

if __name__ == "__main__":
    predict_instance: PredictService = PredictService()
    # predict_instance.load_thetas()
    # mileage: float = float(input("mileage="))
    # result: float = predict_instance.estimate_price(mileage=mileage)
    # print(result)
    # predict_instance.load_dataset()
    predict_instance.load_dataset()
    predict_instance.normalize()
    predict_instance.fit()
    mileage=float(input())
    res = predict_instance.estimate_price2(mileage=mileage)
    print(res)
