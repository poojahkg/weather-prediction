import numpy as np
import pandas as pd  
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('WeatherDATA1.csv')
X = dataset.iloc[:, 4:7].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

a = mean_squared_error(y_test, y_pred)

print(model.predict([[27, 11, 2019]]))