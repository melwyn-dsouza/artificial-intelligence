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

def main():
    """Uncomment one of the "p="
    the first p = predict_class does not use inverse distance weights for predictions
    p = ineverse_d uses inverse distance weights for predictions"""
    
    train = np.genfromtxt("trainingData.csv", delimiter = ",")
    test = np.genfromtxt("testData.csv", delimiter = ",")
    
    train_data, test_data =  normalize(train[:,0:20], test[:,0:20]) #normalising the data
    train_class = train[:,20]
    test_class = test[:,20]
    prediction = []
    
    for query_instance in test_data:
        # p = predict_class(train_data, train_class, query_instance, 1, 3)    #normal KNN
        p = inverse_d(train_data, train_class, query_instance, 1, 3 )     #inverse distance weighted KNN
        prediction.append(p)
    prediction = np.array(prediction)

    calculate_accuracy(test_class, prediction)

def normalize(train_x,test_x):
    """ Normalization is a part of data preparation
    The data set could have features few of which are very large and few very small
    The data is scaled between 0-1
    
    Parameters
    train_x - 2D array training data
    test_x : 2D array test data
    Returns
    train_norm, test_norm : 2D arrays, normalised (Train and Test)"""
    
    minimum = np.min(train_x, axis=0)
    maximum = np.max(train_x, axis = 0)
    train_norm = (train_x - minimum) / (maximum - minimum)
    test_norm = (test_x - minimum) / (maximum - minimum)        #use the same min and max value from the train data on test data
    return train_norm, test_norm

def minkowski_distance(ti,qp,a):
    """find the distance between two points, 
    a=1 for Manhattan distance, a=2 for Euclidean distance
    
    Parameters
    ti - 2D Numpy array with instances
    qp - Query Point (distnace is meausred from this point all instances in ti
    a - hyperparameter for minkowski distance
    
    Returns
    d - 1D array with all the distances between query point and the intances in ti"""
    
    d = (np.sum(abs(ti - qp)**a, axis = 1))**(1/a)
    return d
   
def predict_class(td, cl, qi, a, k):
    """Predict the class of the query instance
    Parameters 
    td,cl,a,k = training data, train classes, minkowski hyperparameter, number of neighbors
    Return
    pred - prediction depeneding on the K neigboring classes 
    """
    
    d = minkowski_distance(td, qi, a)
    sorted_index = np.argsort(d)    #getting distance array sorted, argsort lists indexes only
    unq_cl = dict.fromkeys(np.unique(cl),0) #find all classes in the data to classify
    for i in sorted_index[0:k]:
        unq_cl[cl[i]] += 1
    pred = max(unq_cl, key=unq_cl.get) #find the classes of K neighbors
    return pred

def inverse_d(td, cl, qi, a, k):
    """Inverse distance weighted KNN
    Parameters 
    td,cl,a,k = training data, train classes, minkowski hyperparameter, number of neighbors
    Return
    pred - prediction depeneding on the K neigboring classes """
    
    d = minkowski_distance(td, qi, a)
    sorted_index = np.argsort(d)
    d_sort_inv = 1/np.sort(d)[0:k] 
    d_sum = np.sum(d_sort_inv[0:k])
    vote = d_sort_inv/d_sum
    unq_cl = dict.fromkeys(np.unique(cl),0)
    
    for i in range(0,k):
        unq_cl[cl[sorted_index[i]]] += vote[i]  
    pred = max(unq_cl, key=unq_cl.get)
    return pred
    
def calculate_accuracy(tc,pc):
    """Accuracy of the model, adn all classes
    Parameters 
    tc,pc - 1D arrays with true classes and predicted classes
    """
    values = tc == pc
    values = np.array(values)
    c, counts = np.unique(values, return_counts=True)
    a = (counts[1]/len(tc))*100
    print("Overall accuracy of my model is",a)

    count_0,count_1,count_2,count_3 = 0,0,0,0
    true_0,true_1,true_2,true_3 = 0,0,0,0
    for i in range(0,len(tc)):
        if tc[i] == 0.0:
            count_0 += 1
            if pc[i] == 0.0:
                true_0 += 1
        if tc[i] == 1.0:
            count_1 += 1
            if pc[i] == 1.0:
                true_1 += 1
        if tc[i] == 2.0:
            count_2 += 1
            if pc[i] == 2.0:
                true_2 += 1
        if tc[i] == 3.0:
            count_3 += 1
            if pc[i] == 3.0:
                true_3 += 1        
    a = (true_0/count_0)*100
    print("Accuracy of class 0 is", a)
    a = (true_1/count_1)*100
    print("Accuracy of class 1 is", a)        
    a = (true_2/count_2)*100
    print("Accuracy of class 2 is", a)        
    a = (true_3/count_3)*100
    print("Accuracy of class 3 is", a)
    
if __name__=='__main__':
    main()
