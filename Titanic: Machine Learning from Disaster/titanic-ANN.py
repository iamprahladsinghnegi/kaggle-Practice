# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:39:33 2019

@author: Pankaj Negi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset1 = pd.read_csv('train.csv')
dataset2 = pd.read_csv('test.csv')

dataset1.head()

X_train = dataset1.iloc[:,[2,4,5,6,7,9]].values
y_train = dataset1.iloc[:,1].values
X_test = dataset2.iloc[:,[1,3,4,5,6,8]].values


pd.isnull(X_train).sum() > 0
pd.isnull(X_test).sum() > 0


dataset1['Age'].isnull().sum()
dataset1['Pclass'].isnull().sum()
dataset1['Sex'].isnull().sum()
dataset1['SibSp'].isnull().sum()
dataset1['Parch'].isnull().sum()
dataset1['Fare'].isnull().sum()
#dataset1['Embarked'].isnull().sum()


dataset2['Age'].isnull().sum()
dataset2['Pclass'].isnull().sum()
dataset2['Sex'].isnull().sum()
dataset2['SibSp'].isnull().sum()
dataset2['Parch'].isnull().sum()
dataset2['Fare'].isnull().sum()
#dataset2['Embarked'].isnull().sum()

from sklearn.preprocessing import Imputer
# Removing NaN values from X_train by taking mean 
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer= imputer.fit(X_train[:,2:3])
X_train[:,2:3]=imputer.transform(X_train[:,2:3])

# Dealing with missing categorial values
'''from sklearn.impute import SimpleImputer
sm = SimpleImputer(strategy='most_frequent')
sm= sm.fit(X_train[:,6:7])
X_train[:,6:7] = sm.transform(X_train[:,6:7])'''


#checking the X_train for NaN values
pd.isnull(X_train).sum() > 0

# Removing NaN values from X_test 
imputer2= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer2= imputer2.fit(X_train[:,2:3])
X_test[:,2:3]=imputer2.transform(X_test[:,2:3])
imputer3= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer3= imputer3.fit(X_train[:,5:6])
X_test[:,5:6]=imputer2.transform(X_test[:,5:6])

#checking the X_test for NaN values
pd.isnull(X_test).sum() > 0



from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# Dealng with categorial values in X_tarin
labelencoder_X=LabelEncoder()
X_train[:,1]=labelencoder_X.fit_transform(X_train[:,1])

'''
labelencoder_X2=LabelEncoder()
X_train[:,-1]=labelencoder_X2.fit_transform(X_train[:,-1])


onehotencoder=OneHotEncoder(categorical_features=[-1])
X_train=onehotencoder.fit_transform(X_train).toarray() 
X_train=X_train[:,1:]
'''


# Dealng with categorial values in X_test
X_test[:,1]=labelencoder_X.fit_transform(X_test[:,1])

'''
X_test[:,-1]=labelencoder_X2.fit_transform(X_test[:,-1])

onehotencoder2=OneHotEncoder(categorical_features=[-1])
X_test=onehotencoder2.fit_transform(X_test).toarray() 
X_test=X_test[:,1:]
'''


# featute scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)



# Creating an ANN

# Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding input layer and First hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))

# Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 1000)




# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

'''
ls=[]
for i in range(418):
  if y_pred[i]==False:
    y_pred[i]=0
    ls.append([0])
  else:
    ls.append([1])
    
pred=np.asarray(ls)
'''

y_pred=y_pred.astype(int)

pred=y_pred.reshape((418,)) 



#Creating a CSV file
submission = pd.DataFrame({'PassengerId':dataset2['PassengerId'],'Survived':pred})
submission.to_csv('ANN_prediction',index=False)
