# Stock Price Prediction Web App

This repository contains the code for predicting stock prices using machine learning techniques. The code is implemented in Python using various libraries such as Pandas, Scikit-learn, TensorFlow, etc. 

## Dataset

The dataset used for this project is obtained from Yahoo Finance API using the `yfinance` library. It includes daily stock prices of various companies, such as Apple, Amazon, Google, etc. The dataset is preprocessed and cleaned to remove any missing values or outliers. 

## Models

Several machine learning models are implemented in this project to predict stock prices. These include:

- Linear Regression
- Long Short-Term Memory (LSTM) Neural Network

The models are trained on a subset of the data and tested on the remaining data to evaluate their performance. The best performing model is used to make future stock price predictions.

## Competition Feature

In addition to predicting stock prices, this project includes a competition feature. Users can compete against each other to see who can make the most accurate stock price predictions. The competition feature includes a leaderboard to track users' performance.

## Dashboard

This project includes a dashboard that displays stock prices and predictions in real-time. The dashboard is implemented using Plotly and Dash libraries in Python. Users can interact with the dashboard to view different stocks and time periods.

## Accuracy

The accuracy of the models is evaluated using various metrics such as mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE). The accuracy of each model is compared using these metrics to determine the best performing model.

## Usage

To use this code, follow these steps:

1. Clone this repository to your local machine.
2. Install the required libraries by running `pip install -r requirements.txt` in the terminal.
3. Open the `main.ipynb` file in a Jupyter Notebook or any Python IDE of your choice.
4. Run the code cells in order to preprocess the data, train the models, and make predictions.
5. You can modify the code to use your own dataset or tweak the model parameters to improve performance.
.

![Stock Price Prediction Demo](https://media.giphy.com/media/3oxHQqivSAZbKjxKnC/giphy.gif)
