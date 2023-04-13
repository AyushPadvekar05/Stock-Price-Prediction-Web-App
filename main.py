import pandas_datareader as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import yfinance as yf
import plotly.graph_objs as go
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Stock Safari", page_icon=":chart_with_upwards_trend:")

EXAMPLE_NO = 1 

def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Prediction", "Data Accuracy","Comparison","Dashboard"],  # required
                icons=["house", "file-ppt","graph-up-arrow","columns-gap", "clipboard-data"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

# HOME MENU
if selected == "Home":
    col1, mid, col2 = st.columns([1,1,20])
    with col1:
        image = Image.open("Images/logo-removebg-preview.png")
        st.image(image,width=60)
    with col2:
        st.subheader('Stock Safari')
    image = Image.open("Images/stock-market-concept-design_1017-13713.jpg")
    st.image(image,width=700)




# Prediction
if selected == "Prediction":
    st.title('Stock Price Prediction')

    ticker = st.text_input('Enter ticker symbol')

    start_date = (datetime.today() - timedelta(days=365*20)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    stock_info = {}
    if ticker != '':
        try:
            stock = yf.Ticker(ticker)
            stock_info = {
                'Name': stock.info['longName'],
                'Symbol': stock.info['symbol'],
                'Exchange': stock.info['exchange'],
                'Market Cap': '₹{:,.2f}B'.format(stock.info['marketCap']/1e9) if 'marketCap' in stock.info else 'N/A',
                'PE Ratio': '{:.2f}'.format(stock.info['trailingPE']) if 'trailingPE' in stock.info else 'N/A',
                'Forward PE Ratio': '{:.2f}'.format(stock.info['forwardPE']) if 'forwardPE' in stock.info else 'N/A',
                'Open Price': '₹{:.2f}'.format(stock.info['regularMarketOpen']) if 'regularMarketOpen' in stock.info else 'N/A',
                'Close Price': '₹{:.2f}'.format(stock.info['regularMarketPrice']) if 'regularMarketPrice' in stock.info else 'N/A',
                'High Price': '₹{:.2f}'.format(stock.info['regularMarketDayHigh']) if 'regularMarketDayHigh' in stock.info else 'N/A',
                'Low Price': '₹{:.2f}'.format(stock.info['regularMarketDayLow']) if 'regularMarketDayLow' in stock.info else 'N/A'
            }
        except:
            st.write('Invalid symbol. Please enter a valid stock symbol.')

    # Display the stock summary information in a table format
    if stock_info:
        st.write('<p style="font-size:26px; color:green;"><b>Info<b></p>', unsafe_allow_html=True)
        hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
        info_data = pd.DataFrame([[key, stock_info[key]] for key in ['Name', 'Symbol', 'Exchange', 'Market Cap', 'PE Ratio', 'Forward PE Ratio']])
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(info_data.style.hide_index().set_properties(**{'text-align': 'left', 'padding-left': '10px'}))

        st.write('<p style="font-size:26px; color:red;"><b>Price<b></p>', unsafe_allow_html=True)
        price_data = pd.DataFrame([[key, stock_info[key]] for key in ['Open Price', 'Close Price', 'High Price', 'Low Price']])
        st.table(price_data.style.hide_index().set_properties(**{'text-align': 'left', 'padding-left': '10px'}))

    def load_data(ticker):
            data = yf.download(ticker, start_date, end_date)
            return data

    def train_model(data):
            X = data.drop(['Close'], axis=1)
            y = data['Close']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model, X_test, y_test

    if ticker:
            data = load_data(ticker)
            st.write('Historical Data')
            st.line_chart(data['Close'])
            model, X_test, y_test = train_model(data)
            tomorrow = data.iloc[-1].drop(['Close']).values.reshape(1, -1)
            prediction = model.predict(tomorrow)
            st.write('Predicted Close Price for Next Day:', prediction[0])




# Data Accuracy
if selected == "Data Accuracy":

    st.title('Data Accuracy ')

    user_input = st.text_input('Enter Stock Ticker')

    if (user_input==""):
        st.write("")
    else:
        start = (datetime.today() - timedelta(days=365*20)).strftime('%Y-%m-%d')
        end = datetime.today().strftime('%Y-%m-%d')

        yf.pdr_override()
        df = yf.download(user_input, start, end)

        st.write('Closing Price History')
        st.line_chart(df['Close'])
        data = df.filter(['Close'])

        # Converting the dataframe to a numpy array
        dataset = data.values

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # Creating the training dataset
        training_data_len = math.ceil(len(dataset)*0.8)
        training_data = scaled_data[0:training_data_len, :]

        # Splitting the data into x_train and y_train datasets
        x_train = []
        y_train = []
        for i in range(60, len(training_data)):
            x_train.append(training_data[i-60:i, 0])
            y_train.append(training_data[i, 0])

        # Converting the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshaping the data to be 3-dimensional
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Creating the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # Compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # Creating the testing dataset
        test_data = scaled_data[training_data_len - 60:, :]

        # Splitting the data into x_test and y_test datasets
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        # Converting the x_test to a numpy array
        x_test = np.array(x_test)

        # Reshaping the data to be 3-dimensional
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Getting the model's predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculating the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        st.write('RMSE:', rmse)

        # Plotting the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        st.write('Model')
        st.line_chart(train['Close'])
        st.line_chart(valid[['Close', 'Predictions']])



# Dashboard
if selected == "Dashboard":
     st.title("Stock Data Analysis Dashboard")

     stocks = st.text_input('Enter ticker symbol')
     if (stocks == ""):
        st.write("")
     else:
        start_date = (datetime.today() - timedelta(days=365*20)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')

        def load_data(ticker):
            data = yf.download(ticker, start=start_date, end=end_date)
            data.reset_index(inplace=True)
            return data

        df = load_data(stocks)

        # Define the charts
        def stock_price_chart():
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Closing Price'))
            chart.layout.update(title_text='Stock Prices', xaxis_rangeslider_visible=True)
            return chart

        def daily_return_chart():
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=df['Date'], y=df['Close'].pct_change(), name='Daily Returns'))
            chart.layout.update(title_text='Daily Returns', xaxis_rangeslider_visible=True)
            return chart

        def moving_average_chart():
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=df['Date'], y=df['Close'].rolling(window=50).mean(), name='50-Day Moving Average'))
            chart.layout.update(title_text='Moving Average', xaxis_rangeslider_visible=True)
            return chart

        def volume_chart():
            chart = go.Figure()
            chart.add_trace(go.Scatter(x=df['Date'], y=df['Volume'], name='Volume'))
            chart.layout.update(title_text='Volume', xaxis_rangeslider_visible=True)
            return chart

        def candlestick_chart():
            chart = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Candlestick')])
            chart.layout.update(title_text='Candlestick Chart', xaxis_rangeslider_visible=True)
            return chart

        # Define the layout of the dashboard
        def app_layout():
            st.subheader('Stock Prices')
            st.plotly_chart(stock_price_chart(), use_container_width=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader('Daily Returns')
                daily_return_fig = daily_return_chart()
                if daily_return_fig:
                    st.plotly_chart(daily_return_fig, use_container_width=True)

                st.subheader('Moving Average')
                moving_avg_fig = moving_average_chart()
                if moving_avg_fig:
                    st.plotly_chart(moving_avg_fig, use_container_width=True)

            with col2:
                st.subheader('Volume')
                volume_fig = volume_chart()
                if volume_fig:
                    st.plotly_chart(volume_fig, use_container_width=True)

                st.subheader('Candlestick Chart')
                candlestick_fig = candlestick_chart()
                if candlestick_fig:
                    st.plotly_chart(candlestick_fig, use_container_width=True)

        app_layout()


# Comparison
if selected == "Comparison":

    st.title("Stock Comparison")

    colu1, colu2 = st.columns(2)
    with colu1:
        stock1 = st.text_input("Enter the ticker symbol for the first stock:")
    with colu2:
        stock2 = st.text_input("Enter the ticker symbol for the second stock:")

    start_date = (datetime.today() - timedelta(days=365*20)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    if stock1 == "":
        st.write("")
    elif stock2 =="":
        st.write("")
    else:
        #  Define function to plot stock prices
        def plot_stock1(symbol):
            data = yf.download(stock1, start_date, end_date)
            fig1, ax = plt.subplots()
            ax.plot(data.index, data['Close'], label=symbol)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            return fig1

        def plot_stock2(symbol):
            data = yf.download(stock2, start_date, end_date)
            fig2, ax = plt.subplots()
            ax.plot(data.index, data['Close'], label=symbol)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            return fig2

        st.write(f"Comparing {stock1.upper()} and {stock2.upper()}:")

        graph1, graph2 = st.columns(2)
        with graph1:
            fig1 = plot_stock1(stock1)
            st.pyplot(fig1)

            stock1 = yf.Ticker(stock1)

            if stock1 != '':
                try:
                    
                    info = stock1.info
                    st.write('<p style="font-size:26px; color:green; text-align: center;"><b>Info<b></p>',unsafe_allow_html=True)
                    st.write('**Name:**', info['longName'])
                    st.write('**Symbol:**', info['symbol'])
                    st.write('**Exchange:**', info['exchange'])
                    st.write('**Market Cap:**', '₹{:,.2f}B'.format(info['marketCap']/1e9) if 'marketCap' in info else 'N/A')
                    st.write('**PE Ratio:**', '{:.2f}'.format(info['trailingPE']) if 'trailingPE' in info else 'N/A')
                    st.write('**Forward PE Ratio:**', '{:.2f}'.format(info['forwardPE']) if 'forwardPE' in info else 'N/A')
                    st.write('<p style="font-size:26px; color:red;text-align: center;"><b>Price<b></p>',unsafe_allow_html=True)
                    st.write('**Open Price:**', '₹{:.2f}'.format(info['regularMarketOpen']) if 'regularMarketOpen' in info else 'N/A')
                    st.write('**Close Price:**', '₹{:.2f}'.format(info['regularMarketPrice']) if 'regularMarketPrice' in info else 'N/A')
                    st.write('**High Price:**', '₹{:.2f}'.format(info['regularMarketDayHigh']) if 'regularMarketDayHigh' in info else 'N/A')
                    st.write('**Low Price:**', '₹{:.2f}'.format(info['regularMarketDayLow']) if 'regularMarketDayLow' in info else 'N/A')
                except:
                    st.write('Invalid symbol. Please enter a valid stock symbol.')

        with graph2:
            fig2 = plot_stock2(stock2)
            st.pyplot(fig2)

            stock2 = yf.Ticker(stock2)

            if stock2 != '':
                try:
                    info = stock2.info
                    st.write('<p style="font-size:26px; color:green;text-align: center;"><b>Info<b></p>',unsafe_allow_html=True)
                    st.write('**Name:**', info['longName'])
                    st.write('**Symbol:**', info['symbol'])
                    st.write('**Exchange:**', info['exchange'])
                    st.write('**Market Cap:**', '₹{:,.2f}B'.format(info['marketCap']/1e9) if 'marketCap' in info else 'N/A')
                    st.write('**PE Ratio:**', '{:.2f}'.format(info['trailingPE']) if 'trailingPE' in info else 'N/A')
                    st.write('**Forward PE Ratio:**', '{:.2f}'.format(info['forwardPE']) if 'forwardPE' in info else 'N/A')
                    st.write('<p style="font-size:26px; color:red;text-align: center;"><b>Price<b></p>',unsafe_allow_html=True)
                    st.write('**Open Price:**', '₹{:.2f}'.format(info['regularMarketOpen']) if 'regularMarketOpen' in info else 'N/A')
                    st.write('**Close Price:**', '₹{:.2f}'.format(info['regularMarketPrice']) if 'regularMarketPrice' in info else 'N/A')
                    st.write('**High Price:**', '₹{:.2f}'.format(info['regularMarketDayHigh']) if 'regularMarketDayHigh' in info else 'N/A')
                    st.write('**Low Price:**', '₹{:.2f}'.format(info['regularMarketDayLow']) if 'regularMarketDayLow' in info else 'N/A')
                except:
                    st.write('Invalid symbol. Please enter a valid stock symbol.')

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Add footer with attribution
st.markdown(
    """
    ---
    Created with ❤️ by [Ayush Padvekar](https://github.com/Ayush05-pixel).
    """
)