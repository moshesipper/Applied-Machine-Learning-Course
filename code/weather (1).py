# copyright 2022 moshe sipper
# www.moshesipper.com

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(r'G:\My Drive\prof\progs\python\aml\weather.csv', header=0)
print(df.columns.values)  # ['Humidity' 'Wind' 'Temperature' 'Clouds' 'Raining?']
X, y = df.to_numpy()[:, 0:4], df.to_numpy()[:, 4]

model = LogisticRegression()
model.fit(X, y)

print(model.predict(np.array([60, 20, 15, 90]).reshape(1, -1)))
