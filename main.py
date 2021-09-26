from data_loading import *
from train import *
from test import *

def driver_AR(data="data/avocado.csv"):
    df = read_data(data) 
    df = preprocess_data(df)
    df_US, df_regions, df_subregions = split_df_by_regions(df)

    print("-"*15, "TotalUS", "-"*15)
    # Get time series for conventional avocado for totalUS
    df_US_conventional = df_US[df_US['type_organic']==0]["AveragePrice"]
    df_US_conventional = make_stationary(df_US_conventional)
    train, test, split_range = train_test_split(df_US_conventional)
    model_fitted = train_AR(train)
    test_AR(model_fitted, split_range, len(df_US_conventional)-1, test)

    # Get time series for organic avocado for totalUS 
    df_US_organic = df_US[df_US['type_organic']==1]["AveragePrice"]
    df_US_organic = make_stationary(df_US_organic)
    train, test, split_range = train_test_split(df_US_organic)
    model_fitted = train_AR(train)
    test_AR(model_fitted, split_range, len(df_US_organic)-1, test)
    
    regions = ['West', 'Midsouth', 'Northeast', 'SouthCentral', 'Southeast']
    for region in regions:
        print("-"*15, region, "-"*15)
         # Get time series for conventional avocado for a region
        df_region_conventional = df_regions[(df_regions['type_organic']==0) & (df_regions['region']==region)]["AveragePrice"]
        df_region_conventional = make_stationary(df_region_conventional)
        train, test, split_range = train_test_split(df_region_conventional)
        model_fitted = train_AR(train)
        test_AR(model_fitted, split_range, len(df_region_conventional)-1, test)
        
        # Get time series for organic avocado for a region
        df_region_organic = df_regions[(df_regions['type_organic']==1) & (df_regions['region']==region)]["AveragePrice"]
        df_region_organic = make_stationary(df_region_organic)
        train, test, split_range = train_test_split(df_region_organic)
        model_fitted = train_AR(train)
        test_AR(model_fitted, split_range, len(df_region_organic)-1, test)

def driver_ARIMA(data="data/avocado.csv"):
    df = read_data(data) 
    df = preprocess_data(df)
    df_US, df_regions, df_subregions = split_df_by_regions(df)

    print("-"*15, "TotalUS", "-"*15)
    # Get time series for conventional avocado for totalUS
    df_US_conventional = df_US[df_US['type_organic']==0]["AveragePrice"]
    train, test, split_range = train_test_split(df_US_conventional)
    # grid_serach_arima(train) # Grid search for parameters
    model_fitted = train_ARIMA(train, *(3,0,4))
    test_ARIMA(model_fitted, len(test), test)

    # Get time series for organic avocado for totalUS 
    df_US_organic = df_US[df_US['type_organic']==1]["AveragePrice"]
    train, test, split_range = train_test_split(df_US_organic)
    # grid_serach_arima(train) # Grid search for parameters
    model_fitted = train_ARIMA(train, *(4,0,3))
    test_ARIMA(model_fitted, len(test), test)


if __name__ == "__main__":
    print("\n", "#"*15, "AR", "#"*15, "\n")
    driver_AR()
    print("\n", "#"*15, "ARIMA", "#"*15, "\n")
    driver_ARIMA()