import warnings
warnings.filterwarnings('ignore')
from models.AR import autoregressive
from models.ARIMA import arima


def train_AR(train_data):
    model = autoregressive(train_data)
    model_fitted = model.fit()
    return model_fitted

def train_ARIMA(train_data, *param):
    model = arima(train_data, *param)
    model_fitted = model.fit()
    return model_fitted