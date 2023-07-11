# disable dependency warnings
import warnings
warnings.filterwarnings("ignore")

# import necessary libraries
import pickle

import sklearn
import numpy as np
import pandas as pd


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Load required pretrained models

# Load feature pruning model
with open('rfe_model.pkl', 'rb') as f:
    rfe = pickle.load(f)

# Load fitted linear regression model
with open('lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)



def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    


# Feature extraction:

def features(df):
    #get Boolinger Bands
    df['MA_20'] = df.Close.rolling(window=20).mean()
    df['SD20'] = df.Close.rolling(window=20).std()
    df['Upper_Band'] = df.Close.rolling(window=20).mean() + (df['SD20']*2)
    df['Lower_Band'] = df.Close.rolling(window=20).mean() - (df['SD20']*2)

    #shifting for lagged data 
    df['S_Close(t-1)'] = df.Close.shift(periods=1)
    df['S_Close(t-2)'] = df.Close.shift(periods=2)
    df['S_Close(t-3)'] = df.Close.shift(periods=3)
    df['S_Close(t-5)'] = df.Close.shift(periods=5)
    df['S_Open(t-1)'] = df.Open.shift(periods=1)

    #simple moving average
    df['MA5'] = df.Close.rolling(window=5).mean()
    df['MA10'] = df.Close.rolling(window=10).mean()
    df['MA20'] = df.Close.rolling(window=20).mean()
    df['MA50'] = df.Close.rolling(window=50).mean()

    #Exponential Moving Averages
    df['EMA10'] = df.Close.ewm(span=5, adjust=False).mean().fillna(0)
    df['EMA20'] = df.Close.ewm(span=5, adjust=False).mean().fillna(0)
    df['EMA50'] = df.Close.ewm(span=5, adjust=False).mean().fillna(0)

    #Moving Average Convergance Divergances
    df['EMA_12'] = df.Close.ewm(span=12, adjust = False).mean()
    df['EMA_26'] = df.Close.ewm(span=26, adjust = False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_EMA'] = df.MACD.ewm(span=9, adjust=False).mean()

    #Commodity Channel index
    tp = (df['High'] + df['Low'] + df['Close']) /3
    ma = tp/20 
    md = (tp-ma)/20
    df['CCI'] = (tp-ma)/(0.015 * md)

    #Rate of Change 
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10)))*100

    #Stochastic K
    df['SO%K'] = ((df.Close - df.Low.rolling(window=14).min()) / (df.High.rolling(window=14).max() - df.Low.rolling(window=14).min())) * 100

    #Standard Deviation of last 5 days returns
    df['per_change'] = df.Close.pct_change()
    df['STD5'] = df.per_change.rolling(window=5).std()

    #Force Index
    df['ForceIndex1'] = df.Close.diff(1) * df.Volume
    df['ForceIndex20'] = df.Close.diff(20) * df.Volume

    df = df.drop(columns=['MA_20', 'per_change', 'EMA_12', 'EMA_26'])

    return df



def preprocess(df):

    # na values

    (df := df.drop(["Date"], axis = 1)) if "Date" in df.columns else print("No date column")
    (df := df.drop(["Adj Close"], axis = 1)) if "Adj Close" in df.columns else print("No adj close column")
    
    df = df.fillna(df.mean())

    df = features(df)

    c = df.to_numpy()
    X = c[np.logical_not(np.isnan(c))].reshape(1, -1)

    X_final = rfe.transform(X)

    return X_final



def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    X = preprocess(data)
    preds = lr.predict(X)
    pred1 = preds[0,0]
    pred2 = preds[0,1]
    return [pred1, pred2]


if __name__== "__main__":
    evaluate()