import yfinance as yf
import math
import numpy as np
import pandas as pd
import talib as ta
from sklearn import svm


#Historical price data
data = yf.download("AAPL", "2015-7-2", "2019-12-31")

#Google trends and wikipedia pageviews data 
traindata = pd.ExcelFile("path/dataset.xlsx")
traindata = traindata.parse("dataset")
traindata.index = traindata["Date"]

#Calculating features
traindata = pd.DataFrame(traindata, index = data.index)
traindata = traindata[["Wiki", "Google"]]
traindata = pd.concat([traindata, data], axis=1)

wikima = ta.MA(traindata["Wiki"], timeperiod = 5)
traindata["wikidisparity"] = (traindata["Wiki"] / wikima) * 100
traindata["googleema"] = ta.EMA(traindata["Google"], timeperiod = 5)
traindata["slowk"], traindata["slowd"] = ta.STOCH(traindata["High"], traindata["Low"], 
                                        traindata["Close"], fastk_period = 5, slowk_period = 3, slowk_matype = 0, 
                                        slowd_period =3 , slowd_matype = 0)
traindata["pricema"] = ta.MA(traindata["Adj Close"], timeperiod = 5)
traindata["pricemom"] = ta.MOM(traindata["Adj Close"], timeperiod = 5)
traindata["priceroc"] = ta.ROC(traindata["Adj Close"], timeperiod = 5)
traindata["pricersi"] = ta.RSI(traindata["Adj Close"], timeperiod = 5)

#Determining how many days ahead we will predict
days_ahead = 4

#Determining y values
traindata["classes"] = np.where(traindata["Adj Close"] - traindata["Adj Close"].shift(days_ahead) > 0, 1, 0)
traindata["classes"] = np.where(traindata["Adj Close"] - traindata["Adj Close"].shift(days_ahead) < 0, -1, traindata["classes"])
traindata = traindata.dropna()

#Check if dataset is imbalanced
print(traindata["classes"].value_counts())

#Determining X values
traindata.fillna(value=-99999, inplace=True)
traindata['label'] = traindata["classes"].shift(-days_ahead)
del(traindata["classes"])
X = np.array(traindata.drop(['label'], 1))
X = X[:-days_ahead]
traindata.dropna(inplace = True)

#Determining y values
y = np.array(traindata['label'])

#Rescaling the data
from sklearn.preprocessing import MinMaxScaler
X = MinMaxScaler().fit_transform(X)

#Data splitting
length = int(len(X) * 0.7)
X_train = X[:length]
X_test = X[length:]

y_train = y[:length]
y_test = y[length:]

#Model
clf = svm.SVC(class_weight = "balanced")
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

#Calcuating returns
finaldata = data[-len(y_test):]
final = pd.DataFrame(index = finaldata.index)
final["price"] = finaldata["Adj Close"]
final["prediction"] = clf.predict(X_test)
final["market_return"] = np.log(final["price"] / final["price"].shift(days_ahead))
final["strategy_return"] = final["market_return"] * final["prediction"].shift(days_ahead)
final["cumulative_return"] = final["strategy_return"].cumsum()

#Printing results
print ("\nDirectional Accuracy:", str(round(confidence, 4) * 100) + "%")
print ("Period-end cumulative return:", str(round(final["cumulative_return"][-1], 4) * 100) + "%")
print ("Average return:", str(round(final["strategy_return"].mean(), 4) * 100) + "%")

#Ploting cumulative returns of both market and strategy
returns = final[["strategy_return", "market_return"]].cumsum().plot(title="Cumulative Returns")
