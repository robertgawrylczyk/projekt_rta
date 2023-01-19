import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import flask
import time
import requests
from flask import Flask, request
from yahoo_fin import stock_info as si
import datetime

start_date1 = (datetime.datetime.now() - datetime.timedelta(days=3)).strftime(
    "%Y-%m-%d"
)
interval1 = "1m"


def prepare_data(stocks_ticker, start_date, interval):
    df = si.get_data(ticker=stocks_ticker, start_date=start_date, interval=interval)
    df = df.drop(["ticker"], axis=1)
    for i in df.columns[0:5]:
        for j in range(4):
            df[f"{i}_lag{j + 1}"] = df[i].shift(j + 1)
    last_row = df.iloc[
        -1,
    ].values
    pred_time = df.iloc[
        -1,
    ].name + datetime.timedelta(minutes=1)
    df["close_next_minute"] = df["close"].shift(-1)
    df = df.dropna()
    return df, last_row, pred_time


# podział zbioru
def split_data(df):
    random.seed(42)

    train_fraction = 0.6
    validation_fraction = 0.9 - train_fraction
    test_fraction = 0.1

    random_list = random.sample(range(1, len(df) + 1), len(df))
    sample_list = np.divide(random_list, len(df))

    train_data = df[(sample_list <= 1) & (sample_list > 1 - train_fraction)]
    validation_data = df[
        (sample_list <= 1 - train_fraction) & (sample_list > test_fraction)
    ]
    test_data = df[(sample_list <= test_fraction) & (sample_list >= 0)]

    X_train = np.array(train_data.drop(["close_next_minute"], axis=1))
    y_train = np.array(train_data["close_next_minute"])
    X_validation = np.array(validation_data.drop(["close_next_minute"], axis=1))
    y_validation = np.array(validation_data["close_next_minute"])
    X_test = np.array(test_data.drop(["close_next_minute"], axis=1))
    y_test = np.array(test_data["close_next_minute"])

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def train_model(stocks_ticker, start_date, intereval):
    df, last_row, pred_time = prepare_data(stocks_ticker, start_date1, interval1)
    X_train, y_train, X_validation, y_validation, X_test, y_test = split_data(df)

    param_grid = {
        "bootstrap": [True],
        "max_depth": [7, 14, 28, None],
        "max_features": ["log2", None, 1, 2],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [3, 6, 12, 24],
        "n_estimators": [50],
    }

    model_rf = RandomForestRegressor()
    random_search = RandomizedSearchCV(
        estimator=model_rf,
        n_iter=10,
        param_distributions=param_grid,
        n_jobs=-1,
        verbose=2,
    )
    random_search.fit(X_train, y_train)

    model_rf = RandomForestRegressor(**random_search.best_params_)
    model_rf.fit(X_train, y_train)

    pred_train = model_rf.predict(X_train)
    pred_valid = model_rf.predict(X_validation)
    pred_test = model_rf.predict(X_test)

    MAPE = round(mean_absolute_percentage_error(y_validation, pred_valid) * 100, 4)

    pred = model_rf.predict(last_row.reshape(1, -1))

    diff = pred[0] - last_row[3]

    current_price = round(last_row[3], 4)

    if diff > 0:
        recommendation = "BUY"
    else:
        recommendation = "SELL"
    return round(pred[0], 4), MAPE, recommendation, pred_time, current_price


app = flask.Flask(__name__)


@app.route("/learn")
def learn():
    def update():
        if si.get_quote_data(api_stock_ticker)["marketState"] == "CLOSED":
            yield "data: Market is closed. Try again later. \n\n"
        else:
            yield "data: Training model... \n\n"
            # Preapre model
            time.sleep(3)
            pred = None
            for i in range(1, 101):

                if pred != None:
                    diff = round(pred - current_price, 2)
                else:
                    diff = None

                market_state = si.get_quote_data(api_stock_ticker)["marketState"]
                pred, MAPE, recommendation, pred_time, current_price = train_model(
                    api_stock_ticker, start_date1, interval1
                )

                yield f"data: Prediction no. {i} for {pred_time}: {pred} USD, curent price: {current_price}, MAPE: {MAPE}% market state: {market_state}, recommendation: {recommendation}, last pred diff: {diff} \n\n"

                time.sleep(60)

        yield "data: close\n\n"

    return flask.Response(update(), mimetype="text/event-stream")


@app.route("/", methods=["GET", "POST"])
def index():
    train_model = False
    global api_stock_ticker
    if flask.request.method == "POST":
        if "train_model" in list(flask.request.form):
            train_model = True
        api_stock_ticker = request.form.get("api_stock_ticker")

    return flask.render_template("index.html", train_model=train_model)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
