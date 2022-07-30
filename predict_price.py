import yfinance as yf
from datetime import datetime
import joblib
from keras.models import load_model
import numpy as pd
import pandas as pd
import pandas_ta as ta
import numpy as np


df = yf.download(
    "^GSPC", start=datetime(2022, 7, 1),
    interval='30m'
)
df.reset_index(inplace=True)
df.ta.sma(length=14, append=True)
df.ta.rsi(close="Close", append=True)
df.ta.atr(close="Close", high="High", low="Low", append=True)
df.ta.stdev(close="Close", append=True)

df = df[["Datetime", "Close", "SMA_14",	"RSI_14", "ATRr_14", "STDEV_30"]]
df = df.loc[:]
df["extra"] = 0
df_copy = df.iloc[(df.shape[0] - 21):-1, :]

scaler = joblib.load("resources/parameters/scaler.save")
scaled_data = scaler.transform(df_copy.iloc[:, 1:].values)
sample = scaled_data[:, :-1]
sample = [sample]
sample = np.array(sample)
print(sample.shape)

model = load_model('resources/model/my_model_sequential.h5')
prediction = model.predict(sample)[0][0]
prediction = scaler.inverse_transform([[prediction] * 5 + [prediction]])

comparisson = df[["Datetime", "Close"]]
result = comparisson.iloc[[-1]]
result["prediction"] = prediction[0][-1]
print(result)
