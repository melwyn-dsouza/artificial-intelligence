Name: Melwyn D Souza
Course: Msc in AI
Student ID: R00209495

---------------------------------------------------------------------------------------
Introduction
---------------------------------------------------------------------------------------
Aim of this assignment is to implement Local Search Schemes to solve 3-SAT instances

Local Search methodologies implemented in this assignment:
- GSAT
- GWSAT
- GWALKSAT
- GTABU

---------------------------------------------------------------------------------------
Python File Content
---------------------------------------------------------------------------------------
File name: DSouza_Melwyn_R00209495_MH2_code.py

The project is implemented in Python 3.8
Environment: Anaconda
Compiler: Spyder IDE
Libraries/Packages: random, sys, time,matplotlib


Instructions for running:

1. Download and install all packages mentioned above
2. Spyder IDE is preferred since the project is built and tested in Spyder Anaconda env
3. Place all the instance files in the same directory as the python file
4. Open the script in spyder IDE, follow the comments in the script
5. main():
	-parameterTune(): Uncomment this line to run gsat,gwsat,gwalksat and gtabu on
			  three instance files
			
	-plotRTD = True: Uncomment this line to plot different representations of RTD
6. Select run from options run> configure per file> General settings> command line options
7. Enter "<finename> <algorithm> <nruns> <nBadIters> <nRestarts> <wp> <tl>" and click ok, then run the script

6.To run the script in an external console:
Syntax: python DSouza_Melwyn_R00209495_MH2_code.py <finename> <algorithm> <nruns> <nBadIters> <nRestarts> <wp> <tl>
Example: python DSouza_Melwyn_R00209495_MH2_code.py myfolder/uf20-01.cnf gsat 10 20 50 0.1 5