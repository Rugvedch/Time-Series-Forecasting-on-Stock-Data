import numpy as np
import pandas as pd
import pandas_datareader as data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
plt.style.use('bmh')
import streamlit as st
df=pd.read_csv("merged_file.csv")
column_to_drop=0
df.drop(df.columns[column_to_drop], axis=1, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.legend(['Original Close price'])
st.pyplot(fig)

#describing data
if st.checkbox("Show raw data", False):
    st.subheader('Raw Data From 2010-2021')
    st.write(df)
    st.subheader('Described Raw data')
    st.write(df.describe())


#100ma vs closeprice
st.subheader('Closing Price vs Time Chart With 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df.Close)
plt.plot(ma100)
plt.legend(['Original Close Price','100Days Moving Average'])
st.pyplot(fig)




#100ma+200ma+closeprice
st.subheader('Closing Price vs Time Chart With 100MA & 200mMA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.xlabel('Year-Month')
plt.ylabel('Close Price USD ($)')
plt.plot(df.Close, 'b')
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.legend(['Original Close Price','100Days Moving Average','200Days Moving Average'])
st.pyplot(fig)

#

def main():
    st.title('Predicted Close')

    # Create a slider for date selection
    start_date = st.date_input('Select Start Date', min_value=df['Date'].min().date(), max_value=df['Date'].max().date())
    end_date = st.date_input('Select End Date', min_value=df['Date'].min().date(), max_value=df['Date'].max().date())

    # Convert date input values to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the dataset based on the selected date range
    filtered_data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    st.subheader('Selected Data')
    st.write(filtered_data)

    # Create a line chart to display the selected data
    st.subheader('Price Over Time')
    #import plotly.express as px
    #px.lineplot(filtered_data.set_index('Date')['Close'])
    st.line_chart(filtered_data.set_index('Date')['Close'])

if __name__ == '__main__':
    main()





