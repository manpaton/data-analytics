import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import fetch_california_housing


def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    return df


def compute_vif(X: pd.DataFrame):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif


def feature_engineering(df: pd.DataFrame):
    df = df.copy()
    df['RoomsPerBedroom'] = df['AveRooms'] / df['AveBedrms']
    df['GeoMean'] = (df['Latitude'] + df['Longitude']) / 2

    df = df.drop(['AveRooms', 'AveBedrms', 'Latitude', 'Longitude'], axis=1)
    return df