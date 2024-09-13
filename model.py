import json
import os
import numpy as np
import pandas as pd
import pickle
from zipfile import ZipFile
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")


def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return

    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)

            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close"
                ]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)

def load_frame(frame, timeframe):
    print(f"Loading data...")
    df = frame.loc[:, ['open', 'high', 'low', 'close']].dropna()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)
    df['date'] = frame['date'].apply(pd.to_datetime)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    return df.resample(f'{timeframe}', label='right', closed='right', origin='end').mean()

def prepare_lstm_data(df, look_back):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df.iloc[i:i + look_back].values)
        y.append(df['close'].iloc[i + look_back])
    return np.array(X), np.array(y)

def train_model(timeframe):
    # Load the price data
    price_data = pd.read_csv(training_price_data_path)
    df = load_frame(price_data, timeframe)

    print(df.tail())

    look_back = 60  # Define the number of time steps for LSTM
    X_train, y_train = prepare_lstm_data(df, look_back)
    
    # Define the model
    if MODEL == "LinearRegression":
        model = LinearRegression()
    elif MODEL == "SVR":
        model = SVR()
    elif MODEL == "KernelRidge":
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
    elif MODEL == "BilSTM":
        model = Sequential()
        model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(look_back, 4)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32)
        
        # Save the model
        model.save(model_file_path.replace('.pkl', '.h5'))
    else:
        raise ValueError("Unsupported model")
    
    if MODEL != "BilSTM":
        # Train the traditional models
        model.fit(X_train, y_train)
        # Save the trained model to a file
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    """Load model and predict current price."""
    if MODEL == "BilSTM":
        from keras.models import load_model
        model = load_model(model_file_path.replace('.pkl', '.h5'))
    else:
        with open(model_file_path, "rb") as f:
            model = pickle.load(f)

    # Get current price
    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{TOKEN}USDT", region), timeframe)
    
    print(X_new.tail())
    print(X_new.shape)

    if MODEL == "BilSTM":
        look_back = 60  # Define the number of time steps for LSTM
        X_new, _ = prepare_lstm_data(X_new, look_back)
        current_price_pred = model.predict(X_new)
    else:
        current_price_pred = model.predict(X_new)

    return current_price_pred[0]
