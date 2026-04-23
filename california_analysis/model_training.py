import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_data, feature_engineering, compute_vif


def run_model(X, y, name="model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(f"\n{name}")
    print("R2:", r2_score(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

    return model, scaler, pred, y_test


def main():
    df = load_data()
    df = feature_engineering(df)

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # VIF
    vif = compute_vif(X)
    print("\nVIF:")
    print(vif)

    # basic model
    model, scaler, pred, y_test = run_model(X, y, "Linear Model")

    # log model
    y_log = np.log1p(y)

    model_log, scaler_log, pred_log, y_test_log = run_model(
        X, y_log, "Log Model"
    )


if __name__ == "__main__":
    main()