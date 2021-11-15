import numpy as np
import pandas as pd
from datetime import timedelta
import datetime
from sklearn.preprocessing import RobustScaler
import ta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
import keras
import yfinance as yf
import pandas_datareader.data as pdr
import sys
import matplotlib.pyplot as plt
import os
import seaborn as sns
#Numpy 1.19.4 dependency!
#pip install --upgrade pandas
#pip install --upgrade pandas-datareader
# Importing Libraries
plt.style.use("dark_background")

# Technical Analysis library

class AI():

    start = datetime.datetime(2016,1,1).date()
    end = datetime.date.today()
    
    def __init__(self, symbolOfStock):
        self.symbolOfStock = symbolOfStock
        self.getStock()
        self.pre_processing()
        self.set_scaler()
        self.scale_data()
        self.run_analysis()
        self.validate_regression()
        self.predict_prices()
    
    def getStock(self):
        """
            Function to call the Yahoo API, to get a new data for one specific stock. 
            The Yahoo API uses Ticker symbols instead of company names
        """
        print("We are looking for this ticketsymbol:")
        print(self.symbolOfStock)
        self.stock = pdr.get_data_yahoo(self.symbolOfStock, self.start, self.end)

    def pre_processing(self):
        # Dropping any NaNs
        self.stock.dropna(inplace=True)



        ## Technical Indicators

        # Adding all the indicators
        self.stock = ta.add_all_ta_features(self.stock, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        # Dropping everything else besides 'Close' and the Indicators
        self.stock.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

        # Only using the last 1000 days of data to get a more accurate representation of the current market climate
        self.stock = self.stock.tail(1000)


    def set_scaler(self):
        ## Scaling

        # Scale fitting the close prices separately for inverse_transformations purposes later
        self.close_scaler = RobustScaler()

        self.close_scaler.fit(self.stock[['Close']])

        # Normalizing/Scaling the self.stock
        self.scaler = RobustScaler()

    def scale_data(self):
        self.stock = pd.DataFrame(self.scaler.fit_transform(self.stock), columns=self.stock.columns, index=self.stock.index)

    def split_sequence(self, seq, n_steps_in, n_steps_out):
        """
        Splits the multivariate time sequence
        """
        
        # Creating a list for both variables
        X, y = [], []
        
        for i in range(len(seq)):
            
            # Finding the end of the current sequence
            end = i + n_steps_in
            out_end = end + n_steps_out
            
            # Breaking out of the loop if we have exceeded the dataset's length
            if out_end > len(seq):
                break
            
            # Splitting the sequences into: x = past prices and indicators, y = prices ahead
            seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
            
            X.append(seq_x)
            y.append(seq_y)
        
        return np.array(X), np.array(y)
  
  
    def visualize_training_results(self, results):
        """
        Plots the loss and accuracy for the training and testing data
        """
        history = results.history
        plt.figure(figsize=(16,5))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        
        plt.figure(figsize=(16,5))
        plt.plot(history['val_accuracy'])
        plt.plot(history['accuracy'])
        plt.legend(['val_accuracy', 'accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
    
    
    def layer_maker(self, n_layers, n_nodes, activation, drop=None, d_rate=.5):
        """
        Creates a specified number of hidden layers for an RNN
        Optional: Adds regularization option - the dropout layer to prevent potential overfitting (if necessary)
        """
        
        # Creating the specified number of hidden layers with the specified number of nodes
        for x in range(1,n_layers+1):
            self.model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

            # Adds a Dropout layer after every Nth hidden layer (the 'drop' variable)
            try:
                if x % drop == 0:
                    self.model.add(Dropout(d_rate))
            except:
                pass
          
          
    def validater(self):
        """
        Runs a 'For' loop to iterate through the length of the self.stock and create predicted values for every stated interval
        Returns a self.stock containing the predicted values for the self.model with the corresponding index values based on a business day frequency
        """
        
        # Creating an empty self.stock to store the predictions
        predictions = pd.DataFrame(index=self.stock.index, columns=[self.stock.columns[0]])

        for i in range(self.n_per_in, len(self.stock)-self.n_per_in, self.n_per_out):
            # Creating rolling intervals to predict off of
            x = self.stock[-i - self.n_per_in:-i]

            # Predicting using rolling intervals
            yhat = self.model.predict(np.array(x).reshape(1, self.n_per_in, self.n_features))

            # Transforming values back to their normal prices
            yhat = self.close_scaler.inverse_transform(yhat)[0]

            # self.stock to store the values and append later, frequency uses business days
            pred_df = pd.DataFrame(yhat, 
                                index=pd.date_range(start=x.index[-1], 
                                                    periods=len(yhat), 
                                                    freq="B"),
                                columns=[x.columns[0]])

            # Updating the predictions self.stock
            predictions.update(pred_df)
            
        return predictions


    def val_rmse(self, df1, df2):
        """
        Calculates the root mean square error between the two Dataframes
        """
        df = df1.copy()
        
        # Adding a new column with the closing prices from the second self.stock
        df['close2'] = df2.Close
        
        # Dropping the NaN values
        df.dropna(inplace=True)
        
        # Adding another column containing the difference between the two DFs' closing prices
        df['diff'] = df.Close - df.close2
        
        # Squaring the difference and getting the mean
        rms = (df[['diff']]**2).mean()
        
        # Returning the sqaure root of the root mean square
        return float(np.sqrt(rms))

    def run_analysis(self):

        # How many periods looking back to learn
        self.n_per_in  = 90# How many periods to predict
        self.n_per_out = 30# Features 
        self.n_features = self.stock.shape[1]# Splitting the data into appropriate sequences
        X, y = self.split_sequence(self.stock.to_numpy(), self.n_per_in, self.n_per_out)


        ## Creating the NN

        # Instatiating the self.model
        self.model = Sequential()

        # Activation
        activ = "tanh"

        # Input layer
        self.model.add(LSTM(90, 
                    activation=activ, 
                    return_sequences=True, 
                    input_shape=(self.n_per_in, self.n_features)))

        #self.model.add(Conv1D(90,
        #               (10),
        #               activation=activ, 
        #               input_shape=(self.n_per_in, self.n_features)))
        # Hidden layers
        self.layer_maker(n_layers=1, 
                    n_nodes=30, 
                    activation=activ)

        # Final Hidden layer
        self.model.add(LSTM(60, activation=activ))

        # Output layer
        self.model.add(Dense(self.n_per_out))

        # self.model summary
        self.model.summary()

        # Compiling the data with selected specifications
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        ## Fitting and Training
        self.res = self.model.fit(X, y, epochs=1, batch_size=128, validation_split=0.1)


    def validate_regression(self):

        # Transforming the actual values to their original price
        actual = pd.DataFrame(self.close_scaler.inverse_transform(self.stock[["Close"]]), 
                            index=self.stock.index, 
                            columns=[self.stock.columns[0]])

        # Getting a self.stock of the predicted values to validate against
        predictions = self.validater()

        # Printing the RMSE
        print("RMSE:", self.val_rmse(actual, predictions))
            
        # Plotting
        plt.figure(figsize=(9.24,4.84))

        # Plotting those predictions
        plt.plot(predictions, label='Predicted')

        # Plotting the actual values
        plt.plot(actual, label='Actual')

        plt.title(f"Predicted vs Actual Closing Prices")
        plt.ylabel("Price")
        plt.xlabel("Dates")
        plt.legend()
        #plt.show()
        dir_img = os.path.join("static", "chartanalysis", "img")
        plt.savefig(os.path.join(dir_img, "Figure_1.png"))

    def predict_prices(self):
        # Predicting off of the most recent days from the original self.stock
        yhat = self.model.predict(np.array(self.stock.tail(self.n_per_in)).reshape(1, self.n_per_in, self.n_features))

        # Transforming the predicted values back to their original format
        yhat = self.close_scaler.inverse_transform(yhat)[0]

        # Creating a self.stock of the predicted prices
        preds = pd.DataFrame(yhat, 
                            index=pd.date_range(start=self.stock.index[-1]+timedelta(days=1), 
                                                periods=len(yhat), 
                                                freq="B"), 
                            columns=[self.stock.columns[0]])

        # Number of periods back to plot the actual values
        pers = self.n_per_in

        # Transforming the actual values to their original price
        actual = pd.DataFrame(self.close_scaler.inverse_transform(self.stock[["Close"]].tail(pers)), 
                            index=self.stock.Close.tail(pers).index, 
                            columns=[self.stock.columns[0]]).append(preds.head(1))

        # Printing the predicted prices
        
        # Plotting
        plt.figure(figsize=(9.24,4.84))
        plt.plot(actual, label="Actual Prices")
        plt.plot(preds, label="Predicted Prices")
        plt.ylabel("Price")
        plt.xlabel("Dates")
        plt.title(f"Forecasting the next {len(yhat)} days")
        plt.legend()
        #plt.show()
        dir_img = os.path.join("static", "chartanalysis", "img")
        plt.savefig(os.path.join(dir_img, "Figure_2.png"))


