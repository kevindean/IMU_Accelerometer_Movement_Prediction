import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

X = np.genfromtxt("/home/kdean/imu_data.csv")
X = X.reshape(-1, 1, 3)

y = np.genfromtxt("/home/kdean/imu_truth.csv")
y = y.reshape(-1, 1)

inputs = Input(shape=(None, 3))

lstm1 = LSTM(32, return_sequences=True)(inputs)
dr1 = Dropout(0.2)(lstm1)
lstm2 = LSTM(16, return_sequences=True)(dr1)
dr2 = Dropout(0.3)(lstm2)
lstm3 = LSTM(8, return_sequences=True)(dr2)
dr3 = Dropout(0.4)(lstm3)
lstm4 = LSTM(4, return_sequences=True)(dr3)
dr4 = Dropout(0.5)(lstm4)
lstm5 = LSTM(2, return_sequences=True)(dr4)
d1 = Dense(1)(lstm5)

model = Model(inputs=inputs, outputs=d1)
model.compile(loss=Huber(), optimizer=Adam(lr=1e-4), metrics=["accuracy"])
model.summary()
model.fit(X, y, epochs=1000, batch_size=10, shuffle=True, validation_split=0.3)

preds = []
count = 0
for i in X:
    if count % 100 == 0.0:
        print(count)
    preds.append(model.predict(i.reshape(-1, 1, 3)).flatten()[0])
    count += 1


plt.title("Accelermoter Movement Prediction (1 - Moving, 0 - Not moving)")
plt.plot(y, label="Truth")
plt.plot(preds, alpha=0.5, label="Prediction")
plt.legend()
plt.show()
