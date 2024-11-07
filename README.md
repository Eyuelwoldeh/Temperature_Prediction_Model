# Temperature Prediction Using LSTM

This project demonstrates how to use machine learning to predict temperature values based on historical data using a Long Short-Term Memory (LSTM) model. LSTM is a type of recurrent neural network (RNN) well-suited for time-series forecasting tasks due to its ability to learn from sequences of data over time.

In this project, we utilize temperature data from a publicly available climate dataset, which includes temperature readings from 2009 to 2016. The goal is to predict future temperature values based on past observations.

## Dataset Overview

The dataset used for this project is from the **Jena Climate Dataset**, which contains hourly temperature data for multiple years (2009-2016). This dataset includes various weather-related measurements, such as temperature, pressure, and humidity, taken at hourly intervals.

We are specifically using the temperature (`T (degC)`) feature for our predictions, and we perform a downsampling process by selecting every 6th data point to reduce the size of the dataset and make it manageable for model training.

### Dataset Link

You can access the dataset from TensorFlow's built-in dataset storage:
- [Jena Climate Dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)

## Problem Statement

The task is to predict the temperature at the next time step, given the temperature values of the previous 5 time steps. This is a classic time-series forecasting problem where the objective is to capture the underlying temporal dependencies between observations.

## Steps

1. **Data Preprocessing**:
   - Load and clean the data.
   - Select every 6th data point to reduce the volume.
   - Convert the date and time column to a pandas datetime format.
   - Extract the temperature values for model training.
   
2. **Data Transformation**:
   - Create sliding windows of past temperature data (window size = 5) as input features.
   - The corresponding target (output) is the temperature value at the 6th time step.

3. **Model Development**:
   - Build an LSTM model using TensorFlow/Keras for predicting the next temperature value based on the past 5 values.
   - The model uses a single LSTM layer followed by a fully connected Dense layer to predict the next value.

4. **Model Training**:
   - Train the model on a training set, validating it on a separate validation set.
   - Use callbacks to save the best model during training based on performance.

5. **Model Evaluation**:
   - Evaluate the model's performance using root mean squared error (RMSE) and visualize the predictions against the actual values.

6. **Testing**:
   - Predict temperatures using the trained model on the test set and visualize the results.

## Requirements

Before running the code, you'll need to install the following Python libraries:

- **TensorFlow**: A deep learning framework that we use to build and train the LSTM model.
- **Pandas**: A data analysis library to manipulate the dataset.
- **Numpy**: A library for numerical computing, especially for creating and handling arrays.
- **Matplotlib**: A library for data visualization, used here to plot predictions against actual values.

You can install these libraries by running:

```bash
pip install tensorflow pandas numpy matplotlib
```
## Data Preprocessing

The Jena Climate Dataset is loaded into a pandas DataFrame. Initially, the data contains multiple features, including temperature, pressure, and humidity. However, for the purpose of this model, we only need the `T (degC)` column, which represents the temperature in degrees Celsius.

We reduce the dataset by selecting every 6th data point using `df[5::6]`. This step helps in downsampling the data to a more manageable size, making it easier to train the model. The `Date Time` column is converted into pandas' `datetime` format, and we use this column as the index of the DataFrame.

```bash
import tensorflow as tf
import os
import pandas as pd
import numpy as np

# Download and unzip the dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

# Load the dataset
df = pd.read_csv(csv_path)

# Downsample by taking every 6th data point
df = df[5::6]

# Convert the 'Date Time' column to datetime
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
```
### Data Transformation

The model is trained using a sliding window of temperature data. We define a function `df_to_x_y` that takes a DataFrame and a `window_size` as inputs, and returns the input-output pairs for the model. The input features are the previous 5 temperature readings (i.e., a window of 5), and the target output is the temperature value at the 6th time step.

```bash
def df_to_x_y(df, window_size=5):
  df_as_np = df.to_numpy()  # Convert to NumPy array
  x = []  # List to store input features
  y = []  # List to store target labels
  for i in range(len(df_as_np) - window_size):
    row = [[a] for a in df_as_np[i:i+5]]  # Window of 5 past temperature values
    x.append(row)
    label = df_as_np[i+5]  # The temperature at the 6th time step is the target
    y.append(label)
  return np.array(x), np.array(y)
```
### Model Architecture

The model uses an LSTM layer to capture temporal dependencies in the data, followed by a Dense layer for predicting the next temperature value. The architecture is as follows:

1. **Input Layer**: Accepts a 5x1 input (5 past temperature readings).
2. **LSTM Layer**: A single LSTM layer with 64 units to capture sequential patterns.
3. **Dense Layer**: A fully connected layer with ReLU activation.
4. **Output Layer**: A Dense layer with a linear activation to output the predicted temperature.

```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential()
model.add(InputLayer((5,1)))  # Input layer
model.add(LSTM(64))  # LSTM layer with 64 units
model.add(Dense(8, 'relu'))  # Dense layer with ReLU activation
model.add(Dense(1, 'linear'))  # Output layer with linear activation

# Model summary
model.summary()
```
### Model Training

The model is compiled using **Mean Squared Error** as the loss function and **Root Mean Squared Error** as a metric. The Adam optimizer with a learning rate of 0.0001 is used to train the model. We train the model for 10 epochs, using `ModelCheckpoint` to save the best version of the model during training.

```bash
cp = ModelCheckpoint('model/best_model.keras', save_best_only=True)

# Compile the model
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# Train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=[cp])
```
### Model Evaluation

After training, we load the best model and evaluate its performance on the training, validation, and test datasets. We use **Root Mean Squared Error (RMSE)** as the primary evaluation metric to assess the model's accuracy.

```bash
from tensorflow.keras.models import load_model

# Load the best saved model
model = load_model('model/best_model.keras')

# Predict on the training, validation, and test sets
train_predictions = model.predict(x_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions': train_predictions, 'Actuals': y_train})

# Plot train predictions vs actuals
import matplotlib.pyplot as plt
plt.plot(train_results['Train Predictions'][:100])
plt.plot(train_results['Actuals'][:100])

# Repeat for validation and test predictions
val_predictions = model.predict(x_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions': val_predictions, 'Actuals': y_val})

plt.plot(val_results['Val Predictions'][:100])
plt.plot(val_results['Actuals'][:100])

test_predictions = model.predict(x_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions': test_predictions, 'Actuals': y_test})

plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])
```
### Accuracy and Results

There is a csv file within this repsository which contains all the prediction values and the absolute differences between the Actual and the Predicted values. These were then used to calculate the predictive accuracy of the model.

The model's predictive accuracy is calculated as the percentage difference between the predicted and actual temperatures. The final accuracy of the model is 95.83%, demonstrating that the model is quite effective at predicting the temperature based on the historical data. 

### Formula for Prediction Accuracy

To evaluate the accuracy of the model, we calculate the **percentage difference** between the predicted and actual temperature values. The formula used is:

$$\[
\text{Prediction Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \left( 1 - \left| \frac{y_{\text{pred}}^i - y_{\text{actual}}^i}{y_{\text{actual}}^i} \right| \right) \times 100
\]$$

Where:
- $$\( y_{\text{pred}}^i \)$$ is the predicted temperature for the \(i^{\text{th}}\) sample,
- $$\( y_{\text{actual}}^i \)$$ is the actual temperature for the \(i^{\text{th}}\) sample,
- $$\( n \)$$ is the total number of samples.

The closer the result is to 100%, the better the model's predictions are. This formula provides a percentage that reflects how closely the predicted temperatures match the actual values.


## Conclusion

This project demonstrates the power of LSTM models for time-series forecasting tasks, specifically temperature prediction. The trained model successfully predicts future temperature values based on past observations, and the results show that LSTM is a suitable approach for handling sequential data.

Future improvements could include experimenting with different time steps, hyperparameters, or using additional features from the dataset (e.g., humidity, pressure) to improve the model's performance.
