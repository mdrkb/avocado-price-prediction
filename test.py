from data_loading import get_evaluation_metric


def test_AR(model_fitted, start, end, test_data):
    pred = model_fitted.predict(start=start, end=end)
    print('The lag value is: %s' % model_fitted.k_ar)
    print('The coefficients of the model are:\n %s\n' % model_fitted.params)
    get_evaluation_metric(test_data, pred)

def test_ARIMA(model_fitted, test_length, test_data):
    pred = model_fitted.forecast(steps=test_length)[0]
    print('The lag value is: %s' % model_fitted.k_ar)
    print('The coefficients of the model are:\n %s\n' % model_fitted.params)
    get_evaluation_metric(test_data, pred)