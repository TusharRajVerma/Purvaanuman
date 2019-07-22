# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing msme dataset
dataset = pd.read_csv('msme_2.csv')

dataset.dtypes
dataset.head()

x = dataset.iloc[:, [0,2,3]].values
y = dataset.iloc[:, 4].values

# fitting dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])

x
y

x_train = dataset.iloc[0:716,[0,2,3]].values
x_test = dataset.iloc[716:958,[0,2,3]].values
y_train = dataset.iloc[0:716,4].values
y_test = dataset.iloc[716:958,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
x_train[:, 1] = le.fit_transform(x_train[:, 1])
x_test[:, 1] = le.fit_transform(x_test[:, 1])

x_train
x_test
y_train
y_test

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)

# Predicting a new result
y_pred = regressor.predict(x_test)
y_pred

# Visualising the Random Forest Regression results (higher resolution)
plt.bar(x_train[:,0], y_train, color = 'red')
plt.bar(x_test[:,0], y_pred, color = 'blue')
plt.xlabel('Year')
plt.ylabel('Pct')
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
plt.bar(x_train[:,1], y_train, color = 'red')
plt.bar(x_test[:,1], y_pred, color = 'blue')
plt.xlabel('Product')
plt.ylabel('Pct')
plt.show()

# Visualising the Random Forest Regression results (higher resolution)
plt.bar(x_train[:,2], y_train, color = 'red')
plt.bar(x_test[:,2], y_pred, color = 'blue')
plt.xlabel('Quantity')
plt.ylabel('Pct')
plt.show()



