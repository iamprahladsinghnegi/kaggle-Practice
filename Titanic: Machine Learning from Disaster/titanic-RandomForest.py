# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 22:55:38 2019

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

dataset2['Age'].isnull().sum()
dataset2['Pclass'].isnull().sum()
dataset2['Sex'].isnull().sum()
dataset2['SibSp'].isnull().sum()
dataset2['Parch'].isnull().sum()
dataset2['Fare'].isnull().sum()

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer= imputer.fit(X_train[:,2:3])
X_train[:,2:3]=imputer.transform(X_train[:,2:3])

pd.isnull(X_train).sum() > 0


imputer2= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer2= imputer2.fit(X_train[:,2:3])
X_test[:,2:3]=imputer2.transform(X_test[:,2:3])
imputer3= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer3= imputer3.fit(X_train[:,5:6])
X_test[:,5:6]=imputer2.transform(X_test[:,5:6])

#checking the X_test for NaN values
pd.isnull(X_test).sum() > 0

'''from sklearn.impute import SimpleImputer
sm = SimpleImputer(strategy='most_frequent')
sm= sm.fit(X[:,5:6])
X[:,5:6] = sm.transform(X[:,5:6])
y[:,5:6] = sm.transform(y[:,5:6])'''


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X_train[:,1]=labelencoder_X.fit_transform(X_train[:,1])
'''onehotencoder=OneHotEncoder(categorical_features=[1])
X_train=onehotencoder.fit_transform(X_train).toarray() '''


X_test[:,1]=labelencoder_X.fit_transform(X_test[:,1])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)


# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(X_train,y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)



#Creating a CSV file
submission = pd.DataFrame({'PassengerId':dataset2['PassengerId'],'Survived':y_pred})
submission.to_csv('First_prediction4',index=False)


