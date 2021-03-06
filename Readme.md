When calculting position based off the values of an accelerometer, there is drift involved
due to the constant tilt of the orientation. Using a recurrent neural network like a time
series prediction, one can predict whether or not the accelerometer is moving (based off
the x, y, z axis accelerometer values).

# INPUT DATA
![accelerometer data](https://github.com/kevindean/IMU_Accelerometer_Movement_Prediction/blob/master/AccelNNPredictionData.png)

# OUTPUT DATA PREDICTION - preliminary
![prediction and truth](https://github.com/kevindean/IMU_Accelerometer_Movement_Prediction/blob/master/AccelermoterMovementPredictionPreliminary.png)

# RNN Accuracy Results - preliminary
![Accuracy](https://github.com/kevindean/IMU_Accelerometer_Movement_Prediction/blob/master/RNNResults.png)

# RNN Val Loss - Preliminary (ha, definitely suggests that I need more data; in order to prevent overfitting)
![Validation Loss](https://github.com/kevindean/IMU_Accelerometer_Movement_Prediction/blob/master/ValidationLoss.png)

# will be collecting more data, proceed with training, and update further results.
