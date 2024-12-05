import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from tkinter import Tk, Label, Button, OptionMenu, StringVar, Toplevel
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM
from keras.models import Sequential

# Function to process data and plot results
def process_data(company):
    file_name = f"{company}.csv"
    df = pd.read_csv(file_name)
    df = df.reset_index()
    columns_to_drop = ['Date']
    if 'Adj Close' in df.columns:
        columns_to_drop.append('Adj Close')

    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Plot initial dataset
    plt.plot(df.Close)
    plt.title(f"{company} Stock Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.show()

    # Splitting data
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    # Preparing training data
    x_train = []
    y_train = []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Building LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=20, verbose=0)  # Train without printing epochs

    # Preparing testing data
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Making predictions
    y_predicted = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Plotting results
    def plot_results():
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label="Original Price")
        plt.plot(y_predicted, 'r', label="Predicted Price")
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f"{company} Stock Price Prediction")
        plt.legend()
        plt.show()

    # Show plot in a new window
    new_window = Toplevel(root)
    new_window.title(f"{company} Stock Price Prediction")
    Label(new_window, text=f"Prediction for {company}").pack(pady=10)
    Button(new_window, text="Show Plot", command=plot_results).pack()

# GUI setup
root = Tk()
root.title("Stock Price Prediction")
root.geometry("300x200")

# Dropdown menu for selecting company
Label(root, text="Select a Company", font=("Arial", 14)).pack(pady=20)
selected_company = StringVar(root)
selected_company.set("Tesla")  # Default value
companies = ["Tesla", "Microsoft", "Apple", "Google"]
OptionMenu(root, selected_company, *companies).pack(pady=10)

# Process button
Button(root, text="Process Dataset", font=("Arial", 12), command=lambda: process_data(selected_company.get())).pack(pady=20)

root.mainloop()
