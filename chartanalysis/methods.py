import pandas_datareader as web
import numpy as np
import datetime
import os 
from chartanalysis import stock
import pandas as pd
end = datetime.date.today()
start = datetime.datetime(2016,1,1).date()
dirname = os.path.dirname(__file__)
filepathNasdaq = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static/chartanalysis/stocks/stocks_nasdaq.csv'))
fileNasdaq = pd.read_csv(filepathNasdaq)
"""
def getHeaderStocks():
    stock_1 = "Alibaba: " + str(int(web.DataReader("BABA", "yahoo", start, end)['High'][0])) + "$"
    stock_2 = "SAP: " + str(int(web.DataReader("SAP", "yahoo", start, end)['High'][0])) + "$"
    stock_3 = "Apple: " + str(int(web.DataReader("AAPL", "yahoo", start, end)['High'][0])) + "$"
    stock_4 = "Alphabet Inc: " + str(int(web.DataReader("GOOG", "yahoo", start, end)['High'][0])) + "$"
    stock_5 = "Volkswagen: " + str(int(web.DataReader("VWAGY", "yahoo", start, end)['High'][0])) + "$"
    stock_6 = "JP Morgan: " + str(int(web.DataReader("JPM", "yahoo", start, end)['High'][0])) + "$"
    stock_7 = "Palantir: " + str(int(web.DataReader("PLTR", "yahoo", start, end)['High'][0])) + "$"
    stock_8 = "Microsoft: " + str(int(web.DataReader("MSFT", "yahoo", start, end)['High'][0])) + "$"
    stock_9 = "Facebook: " + str(int(web.DataReader("FB", "yahoo", start, end)['High'][0])) + "$"
    stock_10 = "Commerzbank: " + str(int(web.DataReader("CBK.DE", "yahoo", start, end)['High'][0])) + "$"
    stock_11 = "Lufthansa: " + str(int(web.DataReader("LHA.DE", "yahoo", start, end)['High'][0])) + "$"
    stock_12 = "Goldman Sachs: " + str(int(web.DataReader("GS", "yahoo", start, end)['High'][0])) + "$"
    stock_13 = "Accenture: " + str(int(web.DataReader("ACN", "yahoo", start, end)['High'][0])) + "$"
    stock_14 = "Tesla: " + str(int(web.DataReader("TSLA", "yahoo", start, end)['High'][0])) + "$"        
    stock_15 = "BMW: " + str(int(web.DataReader("BMW.DE", "yahoo", start, end)['High'][0])) + "$" 
    headerStocks = [stock_1, stock_2, stock_3, stock_4, stock_5, stock_6, stock_7, stock_8, stock_9, stock_10, stock_11, stock_12, stock_13, stock_14, stock_15] 
    return headerStocks

"""

def getProposals():
    stockList = []
    for index, row in fileNasdaq.iterrows():
        stockList.append(row['Name'] + " (" + row['Symbol'] + ")")
    return(stockList)

def getHeaderStocks():
    stock_1 = "Alibaba: "
    stock_2 = "SAP: "
    stock_3 = "Apple: "
    stock_4 = "Alphabet Inc: "
    stock_5 = "Volkswagen: " 
    stock_6 = "JP Morgan: " 
    stock_7 = "Palantir: " 
    stock_8 = "Microsoft: " 
    stock_9 = "Facebook: "
    stock_10 = "Commerzbank: "
    stock_11 = "Lufthansa: "
    stock_12 = "Goldman Sachs: " 
    stock_13 = "Accenture: " 
    stock_14 = "Tesla: "      
    stock_15 = "BMW: "
    headerStocks = [stock_1, stock_2, stock_3, stock_4, stock_5, stock_6, stock_7, stock_8, stock_9, stock_10, stock_11, stock_12, stock_13, stock_14, stock_15] 
    return headerStocks


def movingaverage(interval, window_size=10):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std  = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)
    return rolling_mean, upper_band, lower_band