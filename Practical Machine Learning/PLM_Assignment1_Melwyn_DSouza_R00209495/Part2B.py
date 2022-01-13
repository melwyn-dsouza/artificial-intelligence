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
import matplotlib.pyplot as plt
import random, sys, time

start = time.time()

random.seed(1)

def main():
    feature_data = np.genfromtxt("clusteringData.csv", delimiter=",")
    feature_data = normalize(feature_data)

    x,y = list(), list()   
    for k in range(2,10):
        x.append(k)
        bestAssCent, bestCost = restart_miniBatchKMeans(feature_data, k, 10, 10)
        print("Best assigned centroids and cost at {} is: \nAssigned Centroids: {} \nCost: {}".format(k,bestAssCent, bestCost))
        y.append(bestCost)
    plt.figure(dpi=500)
    plt.plot(x,y)
    plt.ylabel("Distortion Cost")
    plt.xlabel("Centroids")
    plt.title("KMeans - Elbow Plot")

def normalize(array):
    """ Normalization is a part of data preparation
    The data set could have features few of which are very large and few very small
    The data is scaled between 0-1"""
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis = 0)
    norm = (array - minimum) / (maximum - minimum)
    return norm

def mini_batch(feature_data,b):
    """mini batch selects 'b' random instances from the original feature instance
    i/p
    feature_data
    b
    o/p
    2D array containing random feature instances"""
    noRows = feature_data.shape[0]
    randomInd = np.random.choice(noRows, size=b, replace=False)
    mini_feature = feature_data[randomInd, :]
    return mini_feature

def initalize_centroids(feature_data, k): 
    """Random selection of k instances from data as centroids
    i/p 
    feature_data - The training data instances, numpy 2D array
    k - number of centroids
    o/p
    1D array of random selected of k centroids"""
    current_centroids = [] 
    for i in range(k):
        current_centroids.append(random.choice(feature_data))
    current_centroids = np.array(current_centroids)
    return current_centroids

def minkowski_distance(ti,qp,a):
    """fucntion to find distance between two points
    i/p
    ti - The training data instances, numpy 2D array
    qp - query point (test data)
    a - hyperparameter
    o/p
    1D array containing distance from qp to ti"""
    d = (np.sum(abs(ti - qp)**a, axis = 1))**(1/a)
    d = np.array(d)
    return d

def assign_centroids(feature_data,centroids):
    """assign the centroids to the data set, the nearest centroid will be assigned to the insatnce 
    i/p
    feature_data - 2D array of training data instances
    centroids - 1D array current centroids, initially will be randomly selected centriods
    o/p
    centroid_indices - 1D array with every element representing the centroid assiged to data instance
    """
    distances = []
    for centroid in centroids:
        distances.append(minkowski_distance(feature_data, centroid, 2))
    distances = np.array(distances)
    centroids_indices = np.argmin(distances, axis=0)
    return centroids_indices

def move_centroids(feature_data, centroids_indices, current_centroids):
    """get the mean of all data instances assigned to each class, 
    hence the centroid will move towards the center
    o/p
    current_centroids - returns 1D array with new centroids"""
    for i in range(len(current_centroids)):
        feat_sub = feature_data[centroids_indices==i]
        current_centroids[i] = np.mean(feat_sub, axis=0)
    return current_centroids

def calculate_cost(feature_data, centroids_indices, current_centroids):
    """Calculate the distortion cost
    Uncomment the end of line below to enable mean distortion cost
    returns cost (float value)"""
    cost = (((feature_data - current_centroids[centroids_indices])**2).sum())#/(feature_data.shape[0]) 
    return cost

def mini_batch_move_centroids(feature_data, centroid_indices, current_centroids,v):
    for i in range(len(current_centroids)):
        for j in range(len(centroid_indices)):
            if centroid_indices[j] == i:
                v[i] += 1
                learning_rate = 1/v[i]
                current_centroids[i] = (1-learning_rate)*current_centroids[i] + learning_rate*feature_data[j]
    return current_centroids
    
def restart_miniBatchKMeans(feature_data, noCentroids, iterations, noRestarts):
    """Run KMeans with different configs
    i/p  
    feature_data - training data instances
    iterations - number of inner loops,nested for loop, assign centroids and move centroids 
    noCentroids - number of centroids 
    noRestarts - number outer for loops, here the centroids will be initialized randomly
                then the nested for loop runs for "iterations" numnber of times
    o/p
    returns best cost and best assigned centroids after the restart config runs
    """
    k = noCentroids
    cost = sys.maxsize
    b = int(len(feature_data)/10) #10
    for i in range(noRestarts):
        centroids = initalize_centroids(feature_data, k)
        v = dict.fromkeys( list(range(len(centroids))), 0)
        for j in range(iterations):
            feature_data = mini_batch(feature_data, b)
            assigned_centroids = assign_centroids(feature_data, centroids)
            centroids  = mini_batch_move_centroids(feature_data, assigned_centroids, centroids, v)
            # centroids = move_centroids(feature_data, assigned_centroids, centroids)
            current_cost = calculate_cost(feature_data, assigned_centroids, centroids)
            if current_cost < cost:
                    best_indices = assigned_centroids
                    cost  = current_cost
    return best_indices, cost

if __name__=='__main__':
    main()

total_time = time.time() - start

print("Total time taken by the mini batch model is",total_time)