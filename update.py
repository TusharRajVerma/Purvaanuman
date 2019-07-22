# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing msme dataset
dataset = pd.read_csv('msme_update.csv')

dataset.dtypes
dataset.head()

x_train = dataset.iloc[0:19,0:4].values
x_test = dataset.iloc[19:38,0:4].values
y_train = dataset.iloc[0:19,4].values
y_test = dataset.iloc[19:38,4].values

# fitting dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x_train[:, 0] = le.fit_transform(x_train[:, 0])
x_test[:, 0] = le.fit_transform(x_test[:, 0])

x_train
y_train
x_test
y_test

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)
y_pred

# Visualising the Random Forest Regression results (higher resolution)
plt.scatter(x_train[:,0], y_train, color = 'red')
plt.scatter(x_test[:,0], y_pred, color = 'blue')
plt.xlabel('Product')
plt.ylabel('Pct')
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
plt.scatter(x_train[:,1], y_train, color = 'red')
plt.scatter(x_test[:,1], y_pred, color = 'blue')
plt.xlabel('Quality')
plt.ylabel('Pct')
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
plt.scatter(x_train[:,2], y_train, color = 'red')
plt.scatter(x_test[:,2], y_pred, color = 'blue')
plt.xlabel('Quantity')
plt.ylabel('Pct')
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
plt.scatter(x_train[:,3], y_train, color = 'red')
plt.scatter(x_test[:,3], y_pred, color = 'blue')
plt.xlabel('Price')
plt.ylabel('Pct')
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
plt.bar(x_train[:,0], x_train[:,2], color = 'red')
plt.bar(x_test[:,0], x_test[:,2], color = 'blue')
plt.xlabel('Product')
plt.ylabel('quantity')
plt.show()