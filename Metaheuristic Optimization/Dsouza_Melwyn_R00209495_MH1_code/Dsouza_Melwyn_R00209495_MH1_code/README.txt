TSP (Travelling Salesman Problem) - GA (Genetic Algorithm)

Introduction:
Use genetic algorithm to solve the tsp problem, travelling salesman problem can be described as, 
If a salesman wishes to travel to different cities, what would be the shortest way to travel every 
city only once and return where he started

Steps used in genetic algorithm implementation:
1.	Initialization - Random and NN heuristic
2.	Order 1 Crossover
3.	Inversion Mutation
4.	Truncation Selection
5.	Elitism replacement

Technology used:
1.	Script is implemented in Python 3.8
2.	Compiled used - Spyder IDE  
3.	Python packages – numpy, random, sys, math, matplotlib

Description of programs:
1.	Individuall.py 
		a.	Individual.py contains classes like copy, genDists, insertion_heuristic1_precomp, computeFitness, 
			getFitness, euclideanDistance
		b.	These are used to make a copy of chromosomes, generate distance dictionary containing 2D distance, 
	nearest neighbour calculation, calculate fitness of chromosome etc
2.	Dsouza_Melwyn_R00209495_MH1_code.py
		a.	readInstance - gather info from all tsp data files supplied
		b.	initPopulationRandom – Radom initialization heuristic 
		c.	initPopulationNN – nearest neighbour heuristic
		d.	updateBest – Update the fittest in the population
		e.	truncationTournamentSelection – Rank based election technique 
		f.	order1Crossover – Crossover technique
		g.	inversionMutation – Mutation technique
		h.	updateMatingPool – Mating pool is a sorted by fitness in ascending order 
		i.	newGeneration – create new generation
		j.	GAStep – Uupdate mating pool and run new generation
		k.	Search – run itsrations over GAStep

Each function is commented for better understanding of the code

How to run the script:
1.	Update or install all relevant packages as mentioned above
2.	Use spyder IDE as the script is developed and tested in anaconda spyder env
3.	Place the inst.tsp files in the same directory as the above .py files
4.	Run the file by adding  arguments as shown below: (The arguements can be added in spyder in "command line options" under "run" 
	Command syntax – 
		python Dsouza_Melwyn_R00209495_MH1.py inst_file nRuns nIterations popSize initalisation Pc 
								Pm truncationPercentage elitismPercentatge
	ExampleS – 
		python Dsouza_Melwyn_R00209495_MH1_code.py inst-0.tsp 10 500 100 1 0.8 0.05 0.25 0.1
		python Dsouza_Melwyn_R00209495_MH1_code.py inst-5.tsp 10 500 100 1 0.8 0.05 0.25 0.1	
		python Dsouza_Melwyn_R00209495_MH1_code.py inst-13.tsp 10 500 100 1 0.8 0.05 0.25 0.1	




"results" directory contains all ecperimental results, and graphical representations for inst-0.tsp over different variations