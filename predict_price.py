import yfinance as yf
from datetime import datetime
from termcolor import colored
import joblib
from keras.models import load_model
import numpy as pd
import pandas as pd
import pandas_ta as ta
import numpy as np
from add_indicators import add_indicators, get_column_indicators


APPROACH = "second"
SHIFT = 25

df = yf.download(
    "^NDX", start=datetime(2022, 6, 1),
    interval='30m'
)
df.reset_index(inplace=True)
add_indicators(df)
df.dropna(inplace=True)

df = df[["Datetime", "Close"] + get_column_indicators(df)]
df = df.loc[:]
df["extra"] = 0
df_copy = df.iloc[
    (df.shape[0] - (20 + SHIFT)):
    (df.shape[0] if SHIFT == 0 else -1 * SHIFT), :]

print(df_copy)
scaler = joblib.load(f"resources/{APPROACH}_approach/parameters/scaler.save")
scaled_data = scaler.transform(df_copy.iloc[:, 1:].values)
sample = scaled_data[:, :-1]
sample = [sample]
sample = np.array(sample)
print(sample.shape)


model = load_model(
    f'resources/{APPROACH}_approach/model/my_model_sequential.h5')
prediction = model.predict(sample)[0][0]
prediction = scaler.inverse_transform([[prediction] * 5 + [prediction]])

if SHIFT != 0:
    comparisson = df[["Datetime", "Close"]]
    result = comparisson.iloc[[(df.shape[0] if SHIFT == 0 else -1 * SHIFT)]]
    result["prediction"] = prediction[0][-1]
    print(df_copy)
    print(colored('\n *** Predicted value *** \n', 'yellow'))
    print(result)
else:
    print(colored('\n *** Predicted value *** \n', 'cyan'))
    print(prediction[0][-1])
