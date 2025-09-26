import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_csv
from data_processing import convert_social_time, time_spend, sleep_problem, check_missing, fill_missing

def prepare_data(file_path):
    df = load_csv(file_path)
    df = convert_social_time(df, time_spend)
    check_missing(df, ['social_time_num', sleep_problem])
    df = fill_missing(df, 'social_time_num', method='median')
    df = fill_missing(df, sleep_problem, method='median')
    return df

def train_and_plot(df, feature_col='social_time_num', target_col=None, model_type="linear"):
    if target_col is None:
        target_col = sleep_problem

    X = df[[feature_col]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "forest":
        model = RandomForestRegressor(random_state=42, n_estimators=100)
    else:
        raise ValueError("model_type должен быть 'linear', 'tree' или 'forest'")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nМодель: {model_type}")
    print(f"MSE: {mse:.2f}, R²: {r2:.2f}")

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, color="blue", alpha=0.7, edgecolor="black")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Идеал")
    plt.xlabel("Реальные значения")
    plt.ylabel("Предсказанные значения")
    plt.title(f"Реальные vs предсказанные ({model_type})")
    plt.legend()
    plt.show()

    return model

def predict(model, value):
    # ИСПРАВЛЕННАЯ ВЕРСИЯ - вариант 3
    return model.predict(np.array([[value]]))[0]

if __name__ == "__main__":
    file_path = 'smmhdataset.csv'
    df = prepare_data(file_path)

    model_linear = train_and_plot(df, model_type="linear")
    model_tree = train_and_plot(df, model_type="tree")
    model_forest = train_and_plot(df, model_type="forest")

    print("Прогноз при 4 часах в соцсетях (линейная модель):", predict(model_linear, 4))