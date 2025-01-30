import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
file_path = r"C:\Users\pavan\Downloads\Eartquakes-1990-2023.csv\Eartquakes-1990-2023.csv"
df = pd.read_csv(file_path)

# Step 2: Check the column names to identify the correct column names
print("Columns in the dataset:", df.columns)

# Step 3: Update dataset selection based on correct column names
# Use 'magnitudo' instead of 'magnitude'
df['time'] = pd.to_datetime(df['time'])
df['time_num'] = (df['time'] - df['time'].min()).dt.days  # Convert time to numeric (days since the first event)

data = df[['time_num', 'latitude', 'longitude', 'magnitudo']]  # Use 'magnitudo' for magnitude column

# Step 4: Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 5: Prepare the data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, :-1])  # Features (time, lat, lon)
        y.append(data[i + time_step, -1])  # Target (magnitude)
    return np.array(X), np.array(y)

time_step = 30  # Use 30 previous events to predict the next
X, y = create_dataset(scaled_data, time_step)

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 7: Define and compile the LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output a single value (magnitude)

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 8: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 11: Optionally, inverse transform the predicted results to the original scale
y_pred_inverse = scaler.inverse_transform(np.concatenate([np.zeros((y_pred.shape[0], 3)), y_pred], axis=1))[:, 3]
print("Predictions:", y_pred_inverse)
