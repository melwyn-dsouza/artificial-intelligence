"""
Author: Melwyn D Souza
Student Number: R00209495
Email: melwyn.dsouza@mycit.ie
Course: MSc Artificial Intelligence
Module: Metaheuristic Optimization 
Date: 07/11/2021
file: Individual.py
"""

import random
import math
import numpy as np

class Individual:
    def __init__(self, _size, _data, cgenes, hType):
        """
        Parameters and general variables
        """
        self.fitness    = 0
        self.genes      = []
        self.genSize    = _size
        self.data       = _data

        """
        The Initialization is selcted betweenn random/ Nearest neighbour heuristic 
        depends on user input (from dsouza_melwyn_r00209495.py)
        """
        if cgenes: # Child genes from crossover
            self.genes = cgenes
        elif (hType == "random"):   # Random initialisation of genes
            self.genes = list(self.data.keys())
            random.shuffle(self.genes)
            # print(self.genes)
        elif (hType == "NN"): #NN initialisation of genes
            self.genes = self.insertion_heuristic1_precomp()
            # print(self.genes)
    
    def copy(self):
        """
        Creating a copy of an individual
        """
        ind = Individual(self.genSize, self.data,self.genes[0:self.genSize], "")
        ind.fitness = self.getFitness()
        return ind

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt( (d1[0]-d2[0])**2 + (d1[1]-d2[1])**2 )
    
    
    """
    The functionns below are copieed over from the earlier lab solutions,
    few lines are added and altered to suit my script
    The nearest neighbour selects a random entry from the file provided
    It refers the dictionary created in the function genDists to find the smallest distance to next city
    """
    def genDists(self):
        instance = self.data
        cities=list(instance.keys())
        nCities=len(cities)
        dists= np.zeros((nCities+1,nCities+1),dtype=int)
        for i in range(1,nCities+1):
            for j in range(i,nCities+1):
                # dists[i][j]=dists[j][i]=self.euclideanDistance(instance[cities[i-1]], instance[cities[j-1]])
                dists[i][j]=dists[j][i]=self.euclideanDistance(cities[i-1], cities[j-1])
        return dists
    
    #code copied from lab1 solutions and altered according to the requirements
    def insertion_heuristic1_precomp(self):
        instance = self.data
        distances = self.genDists()
        # print(distances)
        cities = list(instance.keys())
        cIndex = random.randint(0, len(instance)-1)
        tCost = 0
        solution = [cities[cIndex]]   
        del cities[cIndex]
        current_city = solution[0]
        while len(cities) > 0:
            bCity = cities[0]
            bCost = distances[current_city][bCity]
            bIndex = 0
            for city_index in range(1, len(cities)):
                city = cities[city_index]
                cost = distances[current_city][city] 
    
                if bCost > cost:
                    bCost = cost
                    bCity = city
                    bIndex = city_index
            tCost += bCost
            current_city = bCity
            solution.append(current_city)
            del cities[bIndex]
        tCost += distances[current_city][solution[0]] 
        # print(solution, tCost)
        return solution # tcost
        
    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness    = self.euclideanDistance(self.genes[0], self.genes[len(self.genes)-1])
        for i in range(0, self.genSize-1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i+1])

        # print(self.fitness)