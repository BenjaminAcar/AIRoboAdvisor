import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
import datetime
import pylab
import plotly.express as px
import plotly
from sys import exit
import time
import easygui
import os
import json
from flask import Flask, request, render_template
from django.http import HttpResponseRedirect
from django.http import JsonResponse
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import norm
import plotly.graph_objects as go
import chart_studio.plotly as py
from sklearn.preprocessing import PolynomialFeatures
from chartanalysis import methods
import ta
import pyautogui
"""
That is the main stock class. Every user creates one instance of the stock class.
The object includes all kinds of information, like the stock data (price + date), the current displayed plot etc.
For changing the methods respectively the plot modes, different methods are provided.
"""

#.....................................................static variables...........................................................
dirname = os.path.dirname(__file__)
filepathNasdaq = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static/chartanalysis/stocks/stocks_nasdaq.csv'))
fileNasdaq = pd.read_csv(filepathNasdaq)
formatDate = '%Y-%m-%d'
#Fibonacci golden ratios
line_1 = 1
line_2 = 0.618
line_3 = 0.382
line_4 = 0.236
line_5 = 0

yf.pdr_override()
#the screensize is used to resize the plot after starting the full screen mode
#user32 = ctypes.windll.user32

screensize = pyautogui.size()[0], pyautogui.size()[1]
#................................................................................................................................

class Stock:
    maxBoundary = 0
    minBoundary = 0
    graphStamp = ""
    timespanStamp ="6month"
    start = datetime.datetime(2016,1,1).date()
    end = datetime.date.today()
    stock = ""
    maxBoundary = ""
    minBoundary = ""
    fig = ""   
    currentGraph ="" 
    symbolOfStock = ""

    def __init__(self, symbolOfStock):
        self.symbolOfStock = symbolOfStock
        self.getStock()
        self.stock.head()
        self.getBoundaries()
        self.createBasicChart()
        rangeBoundaries = self.maxBoundary - self.minBoundary
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    
    def getStock(self):
        """
            Function to call the Yahoo API, to get a new data for one specific stock. 
            The Yahoo API uses Ticker symbols instead of company names
        """
        self.stock = pdr.get_data_yahoo(self.symbolOfStock, self.start, self.end)

    def daysBetween(self):
        """
            For not every stock the current chosen timespan is provided. For example: If a company 
            is new on the exchange market, only a couple of days are provided. 
            This function checks the latest date of the current data to update the GUI.
        """
        d1 = self.start
        d2 = self.end
        days = abs((d2 - d1).days)
        if days <= 7:
            self.timespanStamp = "1week"
            return self.timespanStamp
        if days <= 31:
            self.timespanStamp = "1month"
            return self.timespanStamp
        if days <= 93:
            self.timespanStamp = "3month"
            return self.timespanStamp
        if days <= 183:
            self.timespanStamp = "6month"
            return self.timespanStamp
        if days <= 365:
            self.timespanStamp = "1year"
            return self.timespanStamp
        if days <= 1095:
            self.timespanStamp = "3year"
            return self.timespanStamp
        if days <= 1825:
            self.timespanStamp = "5year"
            return self.timespanStamp
        if days <= 3650:
            self.timespanStamp = "10year"
            return self.timespanStamp

    def getBoundaries(self):
        """
            Function to update the boundary information. 
            The information is used to create, for example, the Fibonacci Lines.
        """
        self.maxBoundary = max(self.stock['High'])
        self.minBoundary = min(self.stock['High'])

    def getStartDate(self):
        self.start = self.stock.index.min()
        self.end = self.stock.index.max()

    def addLinesToGraph(self, index_1, index_2, index_3, index_4, index_5):
        """
            Function that adds the Fibonacci Lines to the Basic Chart.
            The Fibonacci Lines are based on the so-called golden Ratios.
        """
        self.fig.update_layout(shapes=[
            dict(
            type= 'line',
            yref= 'y', y0= index_1, y1= index_1,
            xref=  "x", x0= self.start, x1= self.end,
            line_color = "white"
            ),
            dict(
            type= 'line',
            yref= 'y', y0= index_2, y1= index_2,
            xref=  "x", x0= self.start, x1= self.end,
            line_color = "white"
            ),
            dict(
            type= 'line',
            yref= 'y', y0= index_3, y1= index_3,
            xref=  "x", x0= self.start, x1= self.end,
            line_color = "white"
            ),
            dict(
            type= 'line',
            yref= 'y', y0= index_4, y1= index_4,
            xref=  "x", x0= self.start, x1= self.end,
            line_color = "white"
            ),
            dict(
            type= 'line',
            yref= 'y', y0= index_5, y1= index_5,
            xref=  "x", x0= self.start, x1= self.end,
            line_color = "white"
            )
        ])

    def calcLines(self, indicatorCase):
        """
            Function that calculates the location of the Fibonacci Lines.
        """
        #bottom line
        idx_5r = self.maxBoundary
        idx_5s = self.maxBoundary
        #head line
        idx_1r = self.minBoundary
        idx_1s = self.minBoundary
        idx_2r = idx_5r-(idx_5r-idx_1r)*(line_2)
        idx_3r = idx_5r-(idx_5r-idx_1r)*(line_3)
        idx_4r = idx_5r-(idx_5r-idx_1r)*(line_4)
        idx_2s = idx_1s+(idx_5s-idx_1s)*(line_2)
        idx_3s = idx_1s+(idx_5s-idx_1s)*(line_3)
        idx_4s = idx_1s+(idx_5s-idx_1s)*(line_4)

        if indicatorCase == "Resistance":
            self.addLinesToGraph(idx_1r, idx_2r, idx_3r, idx_4r, idx_5r)
        if indicatorCase == "Support":
            self.addLinesToGraph(idx_1s, idx_2s, idx_3s, idx_4s, idx_5s)
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def getStartByInput(self, input):
        """
            Function that allows the user to change the displayed timespan of one stock.
        """
        if input == "1week":
            self.timespanStamp = "1week"
            self.start = self.end - datetime.timedelta(days=7)
        if input == "1month":
            self.timespanStamp = "1month"
            self.start = self.end - datetime.timedelta(days=31)
        if input == "3month":
            self.timespanStamp = "3month"
            self.start = self.end - datetime.timedelta(days=93)
        if input == "6month":
            self.timespanStamp = "6month"
            self.start = self.end - datetime.timedelta(days=183)
        if input == "1year":
            self.timespanStamp = "1year"
            self.start = self.end - datetime.timedelta(days=365)
        if input == "3year":
            self.timespanStamp = "3year"
            self.start = self.end - datetime.timedelta(days=1095)
        if input == "5year":
            self.timespanStamp = "5year"
            self.start = self.end - datetime.timedelta(days=1825)
        if input == "10year":
            self.timespanStamp = "10year"
            self.start = self.end - datetime.timedelta(days=3650)

    def updateParam(self):
        self.getBoundaries()
        self.getStartDate()

    def getAggregation(self):
        """
            Function that determines the aggreagtion of the candle charts. 
        """
        if self.timespanStamp == "1week":
            return "1"
        if self.timespanStamp == "1month":
            return "1"
        if self.timespanStamp == "3month":
            return "1"
        if self.timespanStamp == "6month":
            return "1"
        if self.timespanStamp == "1year":
            return "1"
        if self.timespanStamp == "3year":
            return "1"
            #return "M2"
        if self.timespanStamp == "5year":
            return "1"
            #return "M3"
        if self.timespanStamp == "10year":
            return "1"
            #return "M4"
    
    def createCandleChart(self):
        """
            Function that creates the candle chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        self.graphStamp = "Candle"
        self.fig = go.Figure(data=[go.Candlestick(x=self.stock.index,
                    open=self.stock['Open'],
                    high=self.stock['High'],
                    low=self.stock['Low'],
                    close=self.stock['Close'], 
                    xperiod = self.getAggregation())])
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Accumulation/Distribution Index
    def createADIChart(self):
        """
            Function that creates the ADI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.acc_dist_index(self.stock['High'], self.stock['Low'], self.stock['Close'], self.stock['Volume'], fillna=True)

        self.graphStamp = "ADI"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Volume Weighted Average Price
    def createVWAPChart(self):
        """
            Function that creates the VWAP chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.VolumeWeightedAveragePrice(self.stock['High'], self.stock['Low'], self.stock['Close'], self.stock['Volume'], 14, fillna=True)
        self.graphStamp = "VWAP"
        values = vars(values)
        values = values["vwap"]
        self.fig.update_traces(line_color='#00f700')
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Chaikin Money Flow 
    def createCMFChart(self):
        """
            Function that creates the CMF chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.chaikin_money_flow(self.stock['High'], self.stock['Low'], self.stock['Close'], self.stock['Volume'], 20, fillna=True)
        self.graphStamp = "CMF"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Ease of movement
    def createEMVChart(self):
        """
            Function that creates the EMV chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.ease_of_movement(self.stock['High'], self.stock['Low'], self.stock['Volume'], 14, fillna=True)
        self.graphStamp = "EMV"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Force Index
    def createFIChart(self):
        """
            Function that creates the FI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.force_index(self.stock['Close'], self.stock['Volume'], 13, fillna=True)
        self.graphStamp = "FI"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Money Flow Index 
    def createMFIChart(self):
        """
            Function that creates the MFI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.money_flow_index(self.stock['High'], self.stock['Low'],self.stock['Close'], self.stock['Volume'], 14, fillna=True)
        self.graphStamp = "MFI"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Negative Volume Index
    def createNVIChart(self):
        """
            Function that creates the NVI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.negative_volume_index(self.stock['Close'], self.stock['Volume'], fillna=True)
        self.graphStamp = "NVI"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #On-balance volume
    def createOBVChart(self):
        """
            Function that creates the OBV chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.on_balance_volume(self.stock['Close'], self.stock['Volume'], fillna=True)
        self.graphStamp = "OBV"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')
    
    #Volume-price trend
    def createVPTChart(self):
        """
            Function that creates the VPT chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volume.volume_price_trend(self.stock['Close'], self.stock['Volume'], fillna=True)
        self.graphStamp = "VPT"
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Awesome Oscillator
    def createAOChart(self):
        """
            Function that creates the AO chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.AwesomeOscillatorIndicator(self.stock['High'], self.stock['Low'], 5, 34, fillna=True)
        self.graphStamp = "AO"
        values = vars(values)
        values = values["_ao"]
        self.fig.update_traces(line_color='#00f700')
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createKAMAChart(self):
        """
            Function that creates the KAMA chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.KAMAIndicator(self.stock['Close'], 10, 2, 30, fillna=True)
        self.graphStamp = "KAMA"
        values_all = vars(values)
        values = values_all["_kama"]
        values_close = values_all["_close"]
        self.fig = px.line(values_close, x=values_close.index, y=values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createPPOChart(self):
        """
            Function that creates the PPO chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.PercentagePriceOscillator(self.stock['Close'], 26, 12, 19, fillna=True)
        self.graphStamp = "PPO"
        values = vars(values)
        values = values["_ppo"]
        self.fig.update_traces(line_color='#00f700')
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createPVOChart(self):
        """
            Function that creates the PVO chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.PercentageVolumeOscillator(self.stock['Volume'], 26, 12, 19, fillna=True)
        self.graphStamp = "PVO"
        values = vars(values)
        values = values["_pvo"]
        self.fig.update_traces(line_color='#00f700')
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createROCChart(self):
        """
            Function that creates the ROC chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.ROCIndicator(self.stock['Close'], 12, fillna=True)
        self.graphStamp = "ROC"
        values = vars(values)
        values = values["_roc"]
        self.fig.update_traces(line_color='#00f700')
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Stochastic RSI
    def createSRSIChart(self):
        """
            Function that creates the SRSI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.StochRSIIndicator(self.stock['Close'], 14, 3, 4, fillna=True)
        self.graphStamp = "SRSI"
        values = vars(values)
        values = values["_stochrsi_k"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Stochastic Oscillator
    def createSOChart(self):
        """
            Function that creates the SO chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.StochasticOscillator(self.stock['High'], self.stock['Low'], self.stock['Close'], fillna=True)
        self.graphStamp = "SO"
        values = vars(values)
        values = values["_stoch_k"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Ultimate Oscillator
    def createUOChart(self):
        """
            Function that creates the UO chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.UltimateOscillator(self.stock['High'], self.stock['Low'], self.stock['Close'], fillna=True)
        self.graphStamp = "UO"
        values = vars(values)
        values = values["_uo"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Williams R indicator
    def createWRIChart(self):
        """
            Function that creates the WRI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.momentum.WilliamsRIndicator(self.stock['High'], self.stock['Low'], self.stock['Close'], 14, fillna=True)
        self.graphStamp = "WRI"
        values = vars(values)
        values = values["_wr"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')


    #Average True Range 
    def createATRChart(self):
        """
            Function that creates the WRI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volatility.AverageTrueRange(self.stock['High'], self.stock['Low'], self.stock['Close'], 14, fillna=True)
        self.graphStamp = "ATR"
        values = vars(values)
        print(values)
        values = values["_atr"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    # Donchian Channel
    def createDCChart(self):
        """
            Function that creates the WRI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volatility.DonchianChannel(self.stock['High'], self.stock['Low'], self.stock['Close'], 20, 2, fillna=True)
        self.graphStamp = "DC"
        values_all = vars(values)
        
        values_h = values_all["_hband"]
        values_l = values_all["_lband"]
        self.fig = px.line(values_h, x=values_h.index, y=values_h.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.add_scatter(x=values_l.index, y=values_l.values, mode='lines')
        self.fig.update_layout(showlegend=False)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    # Keltner Channel 
    def createKCChart(self):
        """
            Function that creates the WRI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volatility.KeltnerChannel(self.stock['High'], self.stock['Low'], self.stock['Close'], 20, 10, fillna=True, original_version=True)
        self.graphStamp = "KC"
        values_all = vars(values)
        values_h = values_all["_tp_high"]
        values_l = values_all["_tp_low"]
        self.fig = px.line(values_h, x=values_h.index, y=values_h.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.add_scatter(x=values_l.index, y=values_l.values, mode='lines')
        self.fig.update_layout(showlegend=False)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')


    # Ulcer Index 
    def createUIChart(self):
        """
            Function that creates the UI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.volatility.UlcerIndex(self.stock['Close'], 14, fillna=True)
        self.graphStamp = "UI"
        values = vars(values)
        values = values["_ulcer_idx"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Aroon Indicator
    def createAIChart(self):
        """
            Function that creates the AI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.AroonIndicator(self.stock['Close'], 25, fillna=True)
        self.graphStamp = "AI"
        values_all = vars(values)
        
        values_h = values_all["_aroon_up"]
        values_l = values_all["_aroon_down"]
        self.fig = px.line(values_h, x=values_h.index, y=values_h.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.add_scatter(x=values_l.index, y=values_l.values, mode='lines')
        self.fig.update_layout(showlegend=False)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Commodity Channel Index
    def createCCIChart(self):
        """
            Function that creates the CCI chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.CCIIndicator(self.stock['High'], self.stock['Low'], self.stock['Close'], 20, 0.015, fillna=True)
        self.graphStamp = "CCI"
        values = vars(values)
        values = values["_cci"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Detrended Price Oscillator 
    def createDPOChart(self):
        """
            Function that creates the DPO chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.DPOIndicator(self.stock['Close'], 20, fillna=True)
        self.graphStamp = "DPO"
        values = vars(values)
        values = values["_dpo"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Ichimoku Indicator
    def createIIChart(self):
        """
            Function that creates the EMA chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.IchimokuIndicator(self.stock['High'], self.stock['Low'], 9, 26, 52, visual=True, fillna=True)
        self.graphStamp = "II"
        values_all = vars(values)
        
        values_h = values_all["_conv"]
        values_l = values_all["_base"]
        self.fig = px.line(values_h, x=values_h.index, y=values_h.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.add_scatter(x=values_l.index, y=values_l.values, mode='lines')
        self.fig.update_layout(showlegend=False)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #KST Signal
    def createKSTChart(self):
        """
            Function that creates the KST chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.KSTIndicator(self.stock['Close'], 10, 15, 220, 30, 10, 10, 10, 15, 9, fillna=True)
        self.graphStamp = "KST"
        values = vars(values)
        values = values["_kst"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Moving Average Convergence Divergence
    def createMACDChart(self):
        """
            Function that creates the MACD chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.MACD(self.stock['Close'], 26, 12, 9, fillna=True)
        self.graphStamp = "MACD"
        values = vars(values)
        values = values["_macd"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Mass Index
    def createMIChart(self):
        """
            Function that creates the MassIndex chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.MassIndex(self.stock['High'], self.stock['Low'], 9, 25, fillna=True)
        self.graphStamp = "MI"
        values = vars(values)
        values = values["_mass"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Parabolic Stop and Reverse 
    def createPSARChart(self):
        """
            Function that creates the MassIndex chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values_all = ta.trend.PSARIndicator(self.stock['High'], self.stock['Low'], self.stock['Close'], 0.02, 0.2, fillna=True)
        values_all = vars(values_all)
        self.graphStamp = "PSAR"
        values_h = values_all["_psar_up"]
        values_l = values_all["_psar_down"]
        self.fig = px.line(values_h, x=values_h.index, y=values_h.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.add_scatter(x=values_l.index, y=values_l.values, mode='lines')
        self.fig.update_layout(showlegend=False)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')


    #Schaff Trend Cycle
    def createSTCChart(self):
        """
            Function that creates the MassIndex chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.STCIndicator(self.stock['Close'], 50, 23, 10, 3, 3, fillna=True)
        self.graphStamp = "STC"
        values = vars(values)
        values = values["_stc"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Trix 
    def createTRIXChart(self):
        """
            Function that creates the MassIndex chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.TRIXIndicator(self.stock['Close'], 15, fillna=True)
        self.graphStamp = "TRIX"
        values = vars(values)
        values = values["_trix"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Vortex Indicator 
    def createVIChart(self):
        """
            Function that creates the MassIndex chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values_all = ta.trend.VortexIndicator(self.stock['High'], self.stock['Low'], self.stock['Close'], 14, fillna=True)
        values_all = vars(values_all)
        self.graphStamp = "VI"
        values_h = values_all["_vip"]
        values_l = values_all["_vin"]
        self.fig = px.line(values_h, x=values_h.index, y=values_h.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.add_scatter(x=values_l.index, y=values_l.values, mode='lines')
        self.fig.update_layout(showlegend=False)
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    #Weighted Moving Average 
    def createWMAChart(self):
        """
            Function that creates the MassIndex chart. It uses aggregations, if the timespan is too big to display well the candles.
        """
        values = ta.trend.WMAIndicator(self.stock['Close'], 9, fillna=True)
        self.graphStamp = "WMA"
        values = vars(values)
        values = values["_wma"]
        self.fig = px.line(values, x=values.index, y=values.values)
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(xaxis_rangeslider_visible=False) 
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createBollingerBands(self):
        """
            Function that creates the Bolling Bands
        """
        self.graphStamp = "BollingerBands"
        INCREASING_COLOR = '#17BECF'
        DECREASING_COLOR = '#7F7F7F'
        data = [ dict(
            type = 'candlestick',
            open = self.stock.Open,
            high = self.stock.High,
            low = self.stock.Low,
            close = self.stock.Close,
            x = self.stock.index,
            yaxis = 'y2',
            name = 'GS',
            increasing = dict( line = dict( color = INCREASING_COLOR ) ),
            decreasing = dict( line = dict( color = DECREASING_COLOR ) ),
        ) ]
        layout = dict()
        self.fig = dict( data=data, layout=layout )
        self.fig['layout'] = dict()
        self.fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ), showgrid=False, showspikes = True )
        self.fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False, showgrid=False, showspikes = True )
        self.fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8], showgrid=False, showspikes = True )
        self.fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
        self.fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )
        rangeselector=dict(
            visible = True,
            x = 0, y = 0.9,
            bgcolor = '#00f700',
            font = dict( size = 13 ),
            buttons=list([
                dict(count=1,
                    label='reset',
                    step='all'),
                dict(count=1,
                    label='1yr',
                    step='year',
                    stepmode='backward'),
                dict(count=3,
                    label='3 mo',
                    step='month',
                    stepmode='backward'),
                dict(count=1,
                    label='1 mo',
                    step='month',
                    stepmode='backward'),
                dict(step='all')
            ]))
            
        self.fig['layout']['xaxis']['rangeselector'] = rangeselector
        mv_y = methods.movingaverage(self.stock.Close)
        mv_x = list(self.stock.index)

        # Clip the ends
        mv_x = mv_x[5:-5]
        mv_y = mv_y[5:-5]

        self.fig['data'].append( dict( x=mv_x, y=mv_y, type='scatter', mode='lines', 
                                line = dict( width = 1 ),
                                marker = dict( color = '#00f700' ),
                                yaxis = 'y2', name='Moving Average' ) )
        colors = []

        for i in range(len(self.stock.Close)):
            if i != 0:
                if self.stock.Close[i] > self.stock.Close[i-1]:
                    colors.append(INCREASING_COLOR)
                else:
                    colors.append(DECREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
        self.fig['data'].append( dict( x=self.stock.index, y=self.stock.Volume,                         
                                marker=dict( color=colors ),
                                type='bar', yaxis='y', name='Volume') )

        bb_avg, bb_upper, bb_lower = methods.bbands(self.stock.Close)

        self.fig['data'].append( dict( x=self.stock.index, y=bb_upper, type='scatter', yaxis='y2', 
                                line = dict( width = 1 ),
                                marker=dict(color='#f5ff00'), hoverinfo='none', 
                                legendgroup='Bollinger Bands', name='Bollinger Bands') )

        self.fig['data'].append( dict( x=self.stock.index, y=bb_lower, type='scatter', yaxis='y2',
                                line = dict( width = 1 ),
                                marker=dict(color='#fe019a'), hoverinfo='none',
                                legendgroup='Bollinger Bands', showlegend=False ) )
        self.fig['layout']['plot_bgcolor'] = '#000000'
        self.fig['layout']['paper_bgcolor'] = '#000000'
        self.fig['layout']['font_color'] = '#FFFFFF'
        self.fig['layout']['width'] = int(screensize[0]/2)
        self.fig['layout']['height'] = int(screensize[1]/2)
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createMonteChart(self):
        """
            Function that creates the Monte Carlo Simulation btw the plot of the simulation.
            The volatility is used as the stochastic parameter (gets calculated before plotting)
        """
        prices = self.stock['Close']
        returns = prices.pct_change()
        last_price = prices[-1]
        num_simulations = 100
        num_days = 250
        simulation_df = pd.DataFrame()

        for x in range(num_simulations):
            count = 0
            daily_vol = returns.std()     
            price_series = []
            price = last_price*(1 + np.random.normal(0, daily_vol))
            price_series.append(price)
            for y in range(num_days):
                if count == 251:
                    break
                price = price_series[count] * (1 + np.random.normal(0, daily_vol))
                price_series.append(price)
                count += 1
            simulation_df[x] = price_series

        self.fig = go.Figure()
        self.fig.update_layout(xaxis_title = 'Simulated Days', yaxis_title = 'Simulated Stock Price')

        for i in range(1,num_simulations):
            self.fig.add_trace(go.Scatter(
                x=np.arange(0,num_days), y=simulation_df.iloc[:,i],
            ))
        
        
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white")
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createBasicChart(self):
        """
            Function that creates the Basic Chart.
        """
        self.graphStamp = "Basic"
        self.fig = px.line(self.stock, x=self.stock.index, y="High")
        self.fig.update_traces(line_color='#00f700')
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createLinearRegression(self):
        """
            Function that creates the linear Regression
        """
        self.graphStamp = "LinearRegression"
        Y=self.stock['High']
        X=self.stock.index
        X_numbers = list(range(self.stock.shape[0]))
        # linear regression
        reg = LinearRegression().fit(np.vstack(X_numbers), Y)
        linearCurve = reg.predict(np.vstack(X_numbers))

        # plotly figure setup
        self.fig = px.line(self.stock, x=self.stock.index, y="High")
        self.fig.update_traces(line_color='#00f700')
        self.fig.add_trace(go.Scatter(name='Linear Regression', x=X, y=linearCurve, mode='lines', line=dict(color="#ff073a")))
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def createRSI(self):
        """
            Function that creates the RSI plot
        """
        self.graphStamp = "RSI"
        window_length = 14
        close = self.stock['Adj Close']
        delta = close.diff()
        delta = delta[1:] 

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the EWMA
        roll_up1 = up.ewm(span=window_length).mean()
        roll_down1 = down.abs().ewm(span=window_length).mean()

        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))

        # Calculate the SMA
        roll_up2 = up.rolling(window_length).mean()
        roll_down2 = down.abs().rolling(window_length).mean()

        # Calculate the RSI based on SMA
        RS2 = roll_up2 / roll_down2
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))
        
        self.fig = go.Figure()
        self.fig.add_trace(go.Scatter(
            x=RS2.index, y=RS2, name = "RSI via SMA"
        ))
        self.fig.add_trace(go.Scatter(
            x=RSI2.index, y=RSI2, name = "RSI via SMA"
        ))  
        self.fig.add_trace(go.Scatter(
            x=RS1.index, y=RS1, name = "RSI via EWMA", visible = "legendonly"
        ))
        self.fig.add_trace(go.Scatter(
            x=RSI1.index, y=RSI1, name = "RSI via EWMA", visible = "legendonly"
        ))  
                      
        self.fig.update_layout(paper_bgcolor ='#000000', plot_bgcolor = '#000000', font_color="white", width=1000)
        self.fig.update_xaxes(showgrid=False, showspikes = True)
        self.fig.update_yaxes(showgrid=False, showspikes = True)
        self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
        self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')

    def stockSearch(self, request):
        """
            Function that searchs for the input-ticket-symbol. If the symbol
            is not included in the "fileNasdaq" file, the symbol will not be valid.
        """
        stockName = (''.join(list(request.GET.keys()))).replace(" ", "").upper()
        foundRow = fileNasdaq.loc[fileNasdaq['Symbol'] == stockName]
        if len(foundRow) == 1:
            self.symbolOfStock = stockName
            self.getStock()
            self.createBasicChart()
            self.updateParam()
            return [True, foundRow['Name'].values[0]]
        else:
            return[False]

    def fullScreen(self):
        """
            Function that starts the full screen (resizing the plot)
        """
        if self.graphStamp == "BollingerBands":
            self.fig['layout']['width'] = screensize[0]
            self.fig['layout']['height'] = screensize[1]
            self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')
        else:
            self.fig.update_layout(width=screensize[0], height=screensize[1])
            self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')


    def closeFullScreen(self):
        """
            Function that closes the full screen (resizing the plot)
        """
        if self.graphStamp == "BollingerBands":
            self.fig['layout']['width'] = int(screensize[0]/2)
            self.fig['layout']['height'] = int(screensize[1]/2)
            self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')
        else:
            self.fig.update_layout(width=int(screensize[0]/2), height=int(screensize[1]/2))
            self.currentGraph = plotly.offline.plot(self.fig, include_plotlyjs=False, output_type='div')