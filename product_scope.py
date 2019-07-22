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
#y_test1 = y.iloc[20:30, 0].values
y_test2 = y.iloc[20:30, 1].values

x_train
y_train
x_test
y_test2

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
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 5))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile( loss = "binary_crossentropy", 
               optimizer = sgd, 
               metrics=['accuracy']
             )

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train[:,1], batch_size =5, nb_epoch = 80)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.75)
y_pred

# Visualising the ANN results (higher resolution)
plt.scatter(x_train[:,0], y_train[:,1], color = 'red')
plt.plot(x_test[:,0], y_pred, color = 'blue')
plt.xlabel('Product')
plt.ylabel('Scope')
plt.show()

# Visualising the ANN results (higher resolution)
plt.scatter(x_train[:,2], y_train[:,1], color = 'red')
plt.plot(x_test[:,2], y_pred, color = 'blue')
plt.xlabel('Price')
plt.ylabel('Scope')
plt.show()

# Visualising the ANN results (higher resolution)
plt.scatter(x_train[:,3], y_train[:,1], color = 'red')
plt.plot(x_test[:,3], y_pred, color = 'blue')
plt.xlabel('Quality')
plt.ylabel('Scope')
plt.show()

# Visualising the ANN results (higher resolution)
plt.scatter(x_train[:,4], y_train[:,1], color = 'red')
plt.plot(x_test[:,4], y_pred, color = 'blue')
plt.xlabel('Quantity')
plt.ylabel('Scope')
plt.show()