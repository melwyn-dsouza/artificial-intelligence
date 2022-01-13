# -*- coding: utf-8 -*-
"""
Author: Melwyn D Souza
Student Number: R00209495
Email: melwyn.dsouza@mycit.ie
Course: MSc Artificial Intelligence
Module: Metaheuristic Optimization 
Date: 2/1/2022
"""


import sys, random, copy, time
import matplotlib.pyplot as plt

studentID = 209495


def main():
    
    """uncomment to tune parameters / to run all algorithms for 10 runs on default parameters
    please go to line 359 and follow the comments to tune parameters"""
    # parameterTune()
    
    """Change plotRTD to True if you want to plot run time distribution"""
    plotRTD = False
    plotRTD = True
    
    
    if len(sys.argv) < 8:
        print ("Error - Incorrect input")
        print ("Expecting 'python DSouza_Melwyn_R00209495_MH2.py [instance_file]\
      [algorithm(gsat, gwsat, gtabu, gwalksat)][nRuns] [nBadIters] [nRestarts]\
      [Random walk 'wp'] [Tabu length 'tl']'")
        sys.exit(0)
    
    f, inst_file, alg, nRuns, nBaditers, nRestarts, wp, tl = sys.argv
    
    print("\nYour arguments are:",inst_file, alg, nRuns, nBaditers, nRestarts, wp, tl)
            
    lc = LocalSearch(inst_file, alg, int(nRuns), int(nBaditers), int(nRestarts), float(wp), float(tl), plotRTD)
    lc.run()


"""Class LocalSearch contains attributes to perform few local search techniques"""
class LocalSearch:
    
    """Initialize different paramenters"""
    def __init__(self, _filename, _alg, _nRuns, _nBadIters, _nRestarts, _wp, _tl, _rtd = False):
        self.filename = _filename
        self.nRuns = _nRuns
        self.nVars = 0
        self.nClauses = 0
        self.variables = []
        self.clauses = []
        self.soln = {}
        self.sat, self.unsat = [], []
        self.maxIters = 1000
        self.nBadIters = _nBadIters
        self.nRestarts = _nRestarts
        self.wp = _wp
        self.tl = _tl
        self.alg = _alg
        self.tabuList = [None]*abs(int(self.tl))
        self.plotRTD = _rtd
        
        #Read cnf files and store few things (variables, clauses) in memory
        self.readIns()
    
    """Read CNF files adn extract number of variables and clauses, 
    also list of clauses and variables"""
    def readIns(self):
        f = open(self.filename, 'r')
        cur_clause = []
        #reading each line from CNF file
        for line in f:
            data = line.split()
            if (data[0]=='c') or (len(data) == 0):
                continue
            if data[0] == 'p':
                self.nVars, self.nClaus = int(data[2]), int(data[3])
                continue
            if self.nVars == 0 or self.nClaus == 0:
                print("Please check the CNF file, missing line 'p cnf num_vars num_clauses'")
                sys.exit(0)
            if data[0] == '%':
                break
    
            #all intial lines are processed, lines now are clauses
            for var in data:
                var = int(var)
                if var == 0:
                    self.clauses.append(cur_clause)
                    cur_clause = []
                    continue
                cur_clause.append(var)
                if abs(var) not in self.variables:
                    self.variables.append(abs(var))
            self.variables = sorted(self.variables)       
    
        if self.nVars != len(self.variables):
            print("The total number of variables do not match with 'p cnf num_vars num_clauses'")
            print("nVars is {} but found only {} variables in clauses".format(self.nVars,len(self.variables)))
            print("Variables are:",self.variables)
            sys.exit(0)
        if self.nClaus != len(self.clauses):
            print ("The total number of clauses do not match with 'p cnf num_vars num_clauses'")
            sys.exit(0)
        f.close()   
        return sorted(self.variables), self.clauses
    
    """Randomly assign 0s and 1s creating a solution to start with
    isoln is a  dictionary with variables as keys and assigned value (0/1) as dict.values"""
    def initialSolution(self):
        isoln = {}
        for literal in range(1,len(self.variables)+1):
            isoln[literal] = random.choice([0,1])
        return isoln
    
    """tempSoln is a dict type solution, 
    function isSolution returns lists of SAT and UNSAT clauses"""
    def isSolution(self, tempSoln):
        sat,unsat = [], []
        flag = False
        for clause in self.clauses:
            for literal in clause:
                if literal < 0:
                    if tempSoln[abs(literal)] == 0:
                        sat.append(clause)
                        flag = True
                        break
                else:
                    if tempSoln[literal] == 1:
                        sat.append(clause)
                        flag = True
                        break
            if flag == False:
                unsat.append(clause)
            else:
                flag = False
        return sat, unsat
    
    """Calculate netgain, negative and positive gain of a tempSoln dict type solution
    it compares the SAT and UNSAT clauses of current solution with the previous solution"""
    def calGain(self, tempSoln):
        count = 0
        sat2, unsat2 = self.isSolution(tempSoln)
        for i in self.unsat:
            for j in unsat2:
                if i == j:
                    count += 1
                    break
            posGain = len(self.unsat) - count
        negGain  = len(unsat2) - count
        netGain = posGain - negGain
        # netGain1 = len(self.unsat) - len(unsat2)
        return netGain, posGain, negGain
        
    """Run Local Search algorithms (gsat, gwsat, gtabu, gwalksat) to find best variable to flip"""
    def runAlgorithm(self):
        if self.alg == 'gsat' or 'gtabu':
            cost = {}
            for i in self.soln.items(): #each variable is flipped and scores are saved in dictionary 'cost'
                temp_soln = copy.deepcopy(self.soln)
                flip = 1 if i[1]==0 else 0 
                temp_soln[i[0]] = flip
                
                gain = self.calGain(temp_soln) #calculate net, positive and negative gain
                
                cost[i[0]] = gain[0] #net_gain
            
            if self.alg == 'gsat':
                bestVars = [var for var,Cost in cost.items() if Cost == max(cost.values())]
                bestVar = random.choice(bestVars)
                return bestVar, cost[bestVar]
            
            elif self.alg == 'gtabu':     
                for i in sorted(list(set(cost.values())), reverse = True):
                    bestVars = [var for var,Cost in cost.items() if Cost == i]
                    #check if all the bestVars are in tabulist 
                    if all(variables in self.tabuList for variables in bestVars):
                        continue
                    else: #select randomly a variable and verify its not in tabulist
                        while True:
                            bestVar = random.choice(bestVars)
                            if bestVar not in self.tabuList:
                                self.tabuList.insert(0,bestVar)
                                self.tabuList.pop()
                                return bestVar, cost[bestVar]
        
        if self.alg == 'gwsat' or self.alg == 'gwalksat':
            #randomly select one unsat clause from list of unsat clauses
            randUnsat = random.choice(self.unsat)
            net, pos, neg = {}, {}, {}
            
            for i in randUnsat:
                temp_soln = copy.deepcopy(self.soln)
                flip = 1 if temp_soln[abs(i)]==0 else 0 
                temp_soln[abs(i)] = flip
                gain = self.calGain(temp_soln) #calculate net, positive and negative gain
                net[abs(i)], pos[abs(i)], neg[abs(i)] = gain
            
            if self.alg == 'gwsat':
                #netZeroVars are the variables with 0 negative gain
                netZeroVars = [var for var,Cost in neg.items() if Cost == 0]
                #select a variable from 0 negative gain list, or perform random/gsat with wp probability
                if len(netZeroVars) > 0:
                    bestVar = random.choice(netZeroVars)
                    return bestVar, net[bestVar]
                else:
                    randNum = random.random()
                    if randNum < self.wp:
                        bestVar = abs(random.choice(randUnsat))
                        return bestVar, net[bestVar]
                    else:
                        bestVars = [var for var,Cost in neg.items() if Cost == min(neg.values())]
                        bestVar = random.choice(bestVars)
                        return bestVar, net[bestVar]
            
            #perform random walk with wp probability else gsat
            elif self.alg == 'gwalksat':
                randNum = random.random()
                if randNum < self.wp:
                    bestVar = abs(random.choice(randUnsat))
                    return bestVar, net[bestVar]
                else:
                    bestVars = [var for var,Cost in net.items() if Cost == max(net.values())]
                    bestVar = random.choice(bestVars)
                    return bestVar, net[bestVar]
            
    """Main body of the Search Algorithm"""    
    def mainProcedure(self):
        prevGain, noChange, restart = 0,0,0
        self.soln = self.initialSolution()
        totalTime = 0
        
        #Iterate/flip best variable maxIters times  
        for i in range (self.maxIters):
            steps = 0
            self.sat,self.unsat = self.isSolution(self.soln)
            
            if self.alg == 'gsat' or self.alg == 'gtabu':
                #checks all variables in the cnf file, hence each iteration steps are 20
                steps = i*self.nVars
            elif self.alg == 'gwsat' or self.alg == 'gwalksat':
                #checks one random 3-SAT clause, hence each iteration steps are 3
                steps = i*3 
                
            #check if the solution is SAT 
            if len(self.unsat)==0:
                print("Iteration:{}, nRestarts: {}".format(i,restart))
                print("Solution found: {}".format(self.soln))
                return self.soln, totalTime, steps
            
            start = time.time()
            #run the local search algorithm (gsat, gwsat, gwalksat, gtabu)
            bestVar, netGain = self.runAlgorithm()
            end = time.time()
            
            timeTaken  = end-start
            totalTime  += timeTaken
            
            #bad iterations where gain wont decrease or increase (local optima)
            if prevGain == netGain:
                noChange += 1
                if noChange >= self.nBadIters:
                    # print("\nStuck in a local optima, Restarting at a random point")
                    restart += 1
                    self.soln = self.initialSolution()
                    if restart >= self.nRestarts:
                        break
            else:
                noChange = 0
                
            prevGain = netGain
            #flip the best variable obtained from the search algorithm
            flip = 1 if self.soln[bestVar] == 0 else 0 
            self.soln[bestVar] = flip
            
        print("**Solution not found**")
        return None,None,steps
    
    """run the mainProcedure() and plot RDT"""
    def run(self):
        plt.rcParams['savefig.dpi'] = 500 
        plt.rcParams['figure.dpi'] = 500
        rt = []
        searchSteps = []
        paratuneSteps = []
        
        print("\n"+"*"*10 + self.alg + "*"*10)
        for i in range(self.nRuns):
            random.seed(studentID+(i*100))
            
            print("\nRUN: ",i)

            solution, timetaken, steps = self.mainProcedure()
            print("steps for run {} are {}".format(i,steps))
            #records steps for solution found and "not found"
            paratuneSteps.append(steps)
            
            #only record steps of good solutions for RDT
            if solution is not None:
                rt.append(timetaken)
                searchSteps.append(steps)
        
        """uncomment the lines below to plot different RTD plots"""
        if self.plotRTD == True:
            #nRuns vs run-time
            plt.plot([i for i in range(len(rt))], rt)
            plt.xlabel('nRuns')
            plt.ylabel('Runtime(s)')
            plt.title(self.alg)
            plt.show()
            
            #x,y
            sortrt = sorted(rt)
            jk = [i/self.nRuns for i in range(len(sortrt))]
            plt.plot(sortrt, jk)
            plt.xlabel('Runtime(s)')
            plt.ylabel('P(solve)')
            plt.title(self.alg)
            plt.show()
    
            #log(x),y
            searchSteps = sorted(searchSteps)
            jks = [i/self.nRuns for i in range(len(searchSteps))]
            plt.xscale("log")
            plt.plot(searchSteps, jks)
            plt.xlabel('Runtime(steps)(log scale)')
            plt.ylabel('P(solve)')
            plt.title(self.alg + ' (semi-log)')
            plt.show()
            
            #log(x),log(y)
            sortrt = sorted(rt)
            jk = [i/self.nRuns for i in range(len(sortrt))]
            plt.xscale("log")
            plt.yscale("log")
            plt.plot(sortrt, jk)
            plt.xlabel('Runtime(steps)(log scale)')
            plt.ylabel('P(solve)(log scale)')
            plt.title(self.alg + ' (log-log)')
            plt.show()
    
            #log(x),log(1-(P(solve))
            sortrt = sorted(rt)
            jk = [i/self.nRuns for i in range(len(sortrt))]
            frd = [1-i for i in jk]
            plt.xscale("log")
            plt.yscale("log")
            plt.plot(sortrt, frd)
            plt.xlabel('Runtime(steps)(log scale)')
            plt.ylabel('P(solve)(log scale)')
            plt.title(self.alg + ' (failure rate log-log)')
            plt.show()
        
        return paratuneSteps

"""hyper parameter tuning"""
def parameterTune():
    files = ["inst/uf20-0877.cnf", "inst/uf20-0498.cnf", "inst/uf20-0471.cnf"]
    algs = ['gsat', 'gwsat', 'gtabu', 'gwalksat']
    
    """uncomment to run 10 steps of each local search alg with default parameters on all 3 files"""
    # for alg in algs:
    #     print("\n\n\nALgorithm", alg)
    #     for file in files:
    #         print("File", file)
    #         lc = LocalSearch(file,alg,10,20,50,0.1,5)
    #         steps = lc.run()
    #         print(steps)
        
    for file in files:    
        """Uncomment to tune tabu parameter"""
        # tabuAvg = []
        # for tabulen in list(range(1,10,2)):
        #     lc = LocalSearch(file, 'gtabu',10,20,50,0.1,tabulen)
        #     steps = lc.run()
        #     avgSteps = sum(steps) / len(steps)
        #     tabuAvg.append(avgSteps)
        # print("Algorithm: {}, File: {}, tabuLengths: {}".format('gtabu',file,list(range(1,10,2))))
        # print("Average values of 10 runs for each tabu lenght:")
        # print(tabuAvg)
        
        """uncomment for wp (random walk noise) parameter tuning on gwsat and gwalksat"""
        # for alg in ['gwsat', 'gwalksat']:    
        #     stepsAvg = []
        #     wps =  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        #     for wp in wps:
        #         lc = LocalSearch(file, alg,10,20,50,wp,5)
        #         steps = lc.run()
        #         avgSteps = sum(steps) / len(steps)
        #         stepsAvg.append(avgSteps)
        #     print("Algorithm: {}, File: {}, randomWalkProb: {}".format(alg,file,wps))
        #     print("Average values of 10 runs for each wp:")
        #     print(stepsAvg)
        # print("*"*30)
            
        """uncomment for nRestarts parameter tuning on gsat"""
        # nResAvg = []
        # for nRes in list(range(0,6,1)):
        #     lc = LocalSearch(file, 'gsat', 10, 10, nRes, 0.1, 5)
        #     steps = lc.run()
        #     avgSteps = sum(steps) / len(steps)
        #     nResAvg.append(avgSteps)
        # print("Algorithm: {}, File: {}, nRestarts: {}".format('gsat',file,list(range(0,6,1))))
        # print(nResAvg)
        # print("*"*30)

        """uncomment to tune nBaditers on gsat"""
        # nBadAvg = []
        # for nBad in list(range(0,25,5)):
        #     lc = LocalSearch(file, 'gsat', 10, nBad, 50, 0.1, 5)
        #     steps = lc.run()
        #     avgSteps = sum(steps) / len(steps)
        #     nBadAvg.append(avgSteps)
        # print("Algorithm: {}, File: {}, nBaditers: {}".format('gsat',file,list(range(0,25,5))))
        # print(nBadAvg)
        # print("*"*30)
    print("*"*30)

if __name__=='__main__':
    main()
