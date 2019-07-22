# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('msme_products.csv')

dataset.dtypes
dataset.head()

x=dataset.iloc[:, 0:5].values
y=dataset.iloc[: ,[5,6]].values

x
y

# fitting dataset and encoding categorical variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])
x[:, 1] = le.fit_transform(x[:, 1])

x

x=pd.DataFrame(x)
y=pd.DataFrame(y)

x_train = x.iloc[0:20, 0:5].values
y_train = y.iloc[0:20, [0,1]].values
x_test = x.iloc[20:30,0:5].values
y_test1 = y.iloc[20:30, 0].values
y_test2 = y.iloc[20:30, 1].values

x_train
y_train
x_test
y_test1

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train
x_test

# Importing the Keras libraries and packages (to initialize & create layers)
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
def baseline_model():
    model = Sequential()
    model.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))
    model.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
    model.add(Dense(output_dim = 1, init = 'uniform'))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error'])
    return model

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold,cross_val_score
estimator = KerasRegressor(build_fn = baseline_model, epochs = 1000, batch_size = 5, verbose=1)
kfold = KFold(n_splits = 10, random_state = 1)
results = cross_val_score(estimator, x_train, y_train[:,0], cv = kfold ,n_jobs=1)
estimator.fit(x_train, y_train[:,0])

y_pred = estimator.predict(x_test)
y_pred

plt.scatter(x_train[:,0], y_train[:,0], color = 'red')
plt.plot(x_test[:,0], y_pred, color = 'blue')
plt.xlabel('Product')
plt.ylabel('pct')
plt.show()

plt.scatter(x_train[:,2], y_train[:,0], color = 'red')
plt.plot(x_test[:,2], y_pred, color = 'blue')
plt.xlabel('Product')
plt.ylabel('pct')
plt.show()

plt.scatter(x_train[:,3], y_train[:,0], color = 'red')
plt.plot(x_test[:,3], y_pred, color = 'blue')
plt.xlabel('Product')
plt.ylabel('pct')
plt.show()

plt.scatter(x_train[:,4], y_train[:,0], color = 'red')
plt.plot(x_test[:,4], y_pred, color = 'blue')
plt.xlabel('Product')
plt.ylabel('pct')
plt.show()


