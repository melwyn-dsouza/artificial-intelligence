# -*- coding: utf-8 -*-
"""
Name: Melwyn D Souza
Student Number: R00209495
Email: melwyn.dsouza@mycit.ie
Course: MSc Artificial Intelligence
Module: Practical Machine learning
Date: 14/11/2021

"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def main():
    
    train = np.genfromtxt("trainingData.csv", delimiter = ",")
    test = np.genfromtxt("testData.csv", delimiter = ",")
    train_data, test_data = train[:,0:20], test[:,0:20]
    train_class, test_class  = train[:,-1], test[:,-1]
    
    normalKNN(train_data,train_class,test_data,test_class,3)
    normalizedKNN(train_data,train_class,test_data,test_class,3)
    minmaxscale(train_data,train_class,test_data,test_class,3)
    scaleKNN(train_data,train_class,test_data,test_class,3)

def normalKNN(X_train,y_train,X_test,y_test,k):
    
    model = KNeighborsClassifier(n_neighbors=k, p=1, weights='distance')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    print("\nNormal data score = ", model.score(X_test, y_test))

    
def normalizedKNN(X_train,y_train,X_test,y_test,k):
    
    normalizer = preprocessing.Normalizer().fit(X_train)     
    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    
    nknn = KNeighborsClassifier(n_neighbors=k, p=1, weights='distance')
    nknn.fit(X_train_norm,y_train)
    y_pred1 = nknn.predict(X_test)   
    
    print("\nNormalized data score = ", nknn.score(X_test_norm, y_test))

def minmaxscale(X_train,y_train,X_test,y_test,k):
    
    mms = preprocessing.MinMaxScaler().fit(X_train)
    X_train_mms = mms.transform(X_train)
    X_test_mms = mms.transform(X_test)     
    
    mms_model = KNeighborsClassifier(n_neighbors=k, p=1, weights='distance')
    mms_model.fit(X_train_mms,y_train)
    y_pred1 = mms_model.predict(X_test)   
    
    print("\nMin-Max scaled data score = ", mms_model.score(X_test_mms, y_test))

def scaleKNN(X_train,y_train,X_test,y_test,k):
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scl = scaler.transform(X_train)
    X_test_scl = scaler.transform(X_test)
    
    sknn = KNeighborsClassifier(n_neighbors=k, p=1, weights='distance')
    sknn.fit(X_train_scl, y_train)
    ypred2 = sknn.predict(X_test_scl)
    
    print("\nStandardized data score = ", sknn.score(X_test_scl, y_test))
    
    
    
if __name__=='__main__':
    main()

