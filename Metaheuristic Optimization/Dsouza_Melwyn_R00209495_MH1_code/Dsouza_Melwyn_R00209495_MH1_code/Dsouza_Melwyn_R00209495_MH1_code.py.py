"""
Author: Melwyn D Souza
Student Number: R00209495
Email: melwyn.dsouza@mycit.ie
Course: MSc Artificial Intelligence
Module: Metaheuristic Optimization 
Date: 07/11/2021
file: dsouza_melwyn_r00209495.py
"""



import random,math
from Individual import *
import sys
import numpy as np

myStudentNum = 209495     #Melwyn D Souza - R00209495
random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _maxIterations, _popSize, _initPop, _xoverProb, _mutationRate, _trunk,  _elite ):
        """
        Parameters and general variables
    
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = int(_popSize)
        self.genSize        = None
        self.crossoverProb  = float(_xoverProb)
        self.mutationRate   = float(_mutationRate)
        self.maxIterations  = int(_maxIterations)
        self.trunkSize      = float(_trunk) 
        self.eliteSize      = float(_elite) 
        self.fName          = _fName
        self.initHeuristic  = int(_initPop)
        self.iteration      = 0
        self.data           = {}
        self.noCrossover    = 0         
        self.readInstance()
        
        """
        The type of initialization is selected dependiong on users choice
        0 - Random initialization
        1 - NN initialization
        """
        if self.initHeuristic == 0 :
            print("Random")
            self.initPopulationRandom()
        else:
            print("Nearest Neighbour")
            self.initPopulationNN()
        
        
    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (cid, x, y) = line.split()
            self.data[int(cid)] = (int(x), int(y))
        file.close()

    def initPopulationRandom(self):
        """
        Creating random individuals in the population
        """
        self.population = []
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data,[],"random")
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())
    
    def initPopulationNN(self):
        """
        Creating nearest neighbour individuals in the population
        The function from  Individual is imported for the initialization
        Check Individual file to find NN implementaion
        """
        self.population = []
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data,[], "NN")
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())
            
    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def truncationTournamentSelection(self):
        """
        Truncation selection is a rank based selectionn method
        
        """            
        len_pool = len(self.matingPool)
        upper_bound  = math.ceil(len_pool*self.trunkSize)
        indA = self.matingPool[random.randint(0,upper_bound)]
        indB = self.matingPool[random.randint(0,upper_bound)]
        return indA, indB

    def order1Crossover(self, indA, indB):
        """
        Your Order1 Crossover Implementation
        segment selects a random range with length 1/3 to 1/2 of the chromosome indA
        it then deletes the elements in indB chromosome which are also present in segment
        the segment is then appended at the end of indB
        """
        segment = random.randint(int(len(indA)/3), int(len(indA)/2))
        i = random.randint(0, int(len(indA)/2))
        tmpA  = indA[i:i+segment]
        cgenes = indB.copy()
        for element in tmpA:
            if element in cgenes:
                cgenes.remove(element)
        cgenes.extend(tmpA)
        child = Individual(self.genSize, self.data, cgenes,"")
        return child

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        A 
        """
        if random.random() > self.mutationRate:
            return ind
        segment = random.randint(int(len(ind.genes)/3), int(len(ind.genes)/2))
        i = random.randint(0, int(len(ind.genes)/2))
        cgenes = ind.copy()
        cgenes = cgenes.genes[0:i]+cgenes.genes[i:i+segment][::-1]+cgenes.genes[i+segment:]
        child = Individual(self.genSize, self.data, cgenes,"")
        return child
        
    def crossover(self, indA, indB):
        """
        Executes a dummy crossover and returns the genes for a new individual
        """
        midP=int(self.genSize/2)
        p1 =  indA[0:midP]
        cgenes = p1 + [i for i in indB if i not in p1]
        child = Individual(self.genSize, self.data, cgenes,"")
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swapping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        Note this is a survival of the fittest 
        The mating pool is sorted by the chromosome fitness in ascending order
        This makes it easier to use truncate selection (best 25% of the fittest are allowed to mate)
        This makes it easier for elitism (depending on elite percentage, say 10%, 
                        the first 10% of elite population will be retained in the next generation)
        """
        self.matingPool = []
        sort_dict = dict()
        for ind_i in self.population:
            sort_dict[ind_i] = ind_i.getFitness()
        sort_dict = dict(sorted(sort_dict.items(), key=lambda x: x[1]))
        for i in sort_dict.keys():
            self.matingPool.append(i.genes)

    def newGeneration(self):
        """
        Creating a new generation
        1. Truncation Selection
        2. Order 1 Crossover
        3. Inversion Mutation
        
        elite_retain is the upper limit calculated for the percentage specified by the user, it is a replacement strategy
        The idea is not to lose the best performing chromosomes in creation of new generation
        So we retain the percentage of the fittest chromosomes of the previous generation 
        and copy them over without crossover/mutation (the mating pool is sorted by fitness in ascending order, 
                                                       the first i indexes are directly sopied over to new gen)
        
        no_crossover selects random indexes based on the percentage specified by the user, these indexes are then used to skip
        """
        
        elite_retain  = int(self.eliteSize*self.popSize)
        self.noCrossover = int((1-self.crossoverProb)*self.popSize)
        self.noCrossover = random.sample(range(0,self.popSize), self.noCrossover)
        
        for i in range(self.popSize):
            """
            1. Select two candidates - Truncation Selection
            2. Apply Crossover - Order1Crossover 
            3. Apply Mutation - Inversion Mutation
            """
            if i < elite_retain:
                hallOfFame = Individual(self.genSize, self.data, self.matingPool[i],"")
                hallOfFame.computeFitness()
                self.updateBest(hallOfFame)
                self.population[i]=hallOfFame
                continue
            parent1, parent2 = self.truncationTournamentSelection()
            if i in self.noCrossover:
                child = Individual(self.genSize, self.data, parent1,"")
            else:
                child = self.order1Crossover(parent1,parent2)
            child  = self.inversionMutation(child)
            child.computeFitness()
            self.updateBest(child)
            self.population[i]=child

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """
        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print ("Total iterations: ", self.iteration)
        print ("Best Solution: ", self.best.getFitness())
        print()
        return self.best.getFitness()

if len(sys.argv) < 9:
    print ("Error - Incorrect input")
    print ("Expecting python TSP.py [instance] [number of runs] [max iterations] [population size]", 
            "[initialisation method] [xover prob] [mutate prob] [truncation] [elitism] ")
    sys.exit(0)


f, inst, nRuns, nIters, pop, initH, pC, pM, trunkP, eliteP = sys.argv


"""####################################### CONFIG #############################"""
fitness_y = []

fileName = str(inst)
nRuns = int(nRuns)
max_iters = int(nIters)
popSize = int(pop) #100
init_type = int(initH)
pC = float(pC)
pM = float(pM)
trunkP = float(trunkP) #0.25
elitism = float(eliteP)

print()
print("*"*100)
print("My configuratiion")
print("Filename: ",fileName)
print("No of runs: ",nRuns)
print("Maximum Iterations: ",max_iters)
print("Population size is: ",popSize)
print("Crossover probability: {}, Mutation probability: {}".format(pC, pM))
print("Truncation rate: {}, Elitism rate: {}".format(trunkP,elitism))
print("*"*100)
print()

'''
Reading in parameters, but it is up to you to implement what needs implementing
e.g. truncation selection, elitism, initialisation with heuristic vs random, etc'''

avg_fitness = []
for i in range (0,nRuns):
    ga = BasicTSP(fileName, max_iters, popSize, init_type, pC, pM, trunkP, elitism)
    best = ga.search()
    avg_fitness.append(best)
    
avg_fitness = np.mean(avg_fitness)
fitness_y.append(round(avg_fitness,3))
print("The average fitness with elitsm {}, pC {}, pM {} and trunkP {} is {}".format(elitism, pC, pM, trunkP, avg_fitness))

