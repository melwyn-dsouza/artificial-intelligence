# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:18:25 2022

@author: dsouzm3
"""

from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
import copy

def task1():
    names = ["James", "Daniel", "Emily", "Sophie"]
    starters = ["Carpaccio", "Prawn_Cocktail", "Onion_Soup", "Mushroom_Tart"]
    mainCourse = ["Filet_Steak", "Vegan_Pie", "Baked_Mackerel", "Fried_Chicken"]
    drinks = ["Beer", "Coke", "Red_Wine", "White_Wine"]
    deserts = ["Ice_Cream","Chocolate_Cake", "Apple_Crumble", "Tiramisu"]

    model = cp_model.CpModel()
    
    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self, starter, maincourse, dessert, drink):
            cp_model.CpSolverSolutionCallback.__init__(self)    
            self.starter_ = starter
            self.main_course_ = maincourse
            self.drink_ = drink
            self.dessert_ = dessert
            self.solutions_ = 0
    
        def OnSolutionCallback(self):
            self.solutions_ = self.solutions_ + 1
            print("Solution:", self.solutions_ )
            
            for name in names:
                print(" - "+name+":")
                for starter in starters:
                    if (self.Value(self.starter_[name][starter])):
                        print("    - ", starter)
                for mains in mainCourse:
                    if (self.Value(self.main_course_[name][mains])):
                        print("    - ", mains)
                for desert in deserts:
                    if (self.Value(self.dessert_[name][desert])):
                        print("    - ", desert)
                for drink in drinks:
                    if (self.Value(self.drink_[name][drink])):
                        print("    - ", drink)
            print()
    
    #creating necessary decision variables
    person_starter = {}
    for name in names:
        variables = {}
        for starter in starters:
            variables[starter] = model.NewBoolVar(name + starter)
        person_starter[name] = variables
    
    person_mainCourse = {}
    for name in names:
        variables = {}
        for mains in mainCourse:
            variables[mains] = model.NewBoolVar(name + mains)
        person_mainCourse[name] = variables

    person_drinks = {}
    for name in names:
        variables = {}
        for drink in drinks:
            variables[drink] = model.NewBoolVar(name + drink)
        person_drinks[name] = variables

    person_deserts = {}
    for name in names:
        variables = {}
        for desert in deserts:
            variables[desert] = model.NewBoolVar(name + desert)
        person_deserts[name] = variables
                
    for name in names:
       variables = []
       for starter in starters:
           variables.append(person_starter[name][starter])
       model.AddBoolOr(variables)

       variables = []
       for mains in mainCourse:
           variables.append(person_mainCourse[name][mains])
       model.AddBoolOr(variables)

       variables = []
       for desert in deserts:
           variables.append(person_deserts[name][desert])
       model.AddBoolOr(variables)

       variables = []
       for drink in drinks:
           variables.append(person_drinks[name][drink])
       model.AddBoolOr(variables)
       # print(variables)

       """
       Max one item per course per person
       """
       for i in range(4):
           for j in range(i+1,4):
               model.AddBoolOr([
                       person_drinks[name][drinks[i]].Not(), 
                       person_drinks[name][drinks[j]].Not()])
               model.AddBoolOr([
                       person_starter[name][starters[i]].Not(), 
                       person_starter[name][starters[j]].Not()])
               model.AddBoolOr([
                       person_mainCourse[name][mainCourse[i]].Not(), 
                       person_mainCourse[name][mainCourse[j]].Not()])
               model.AddBoolOr([
                       person_deserts[name][deserts[i]].Not(), 
                       person_deserts[name][deserts[j]].Not()])

       """
       Every person has a different item
       """
       for i in range(4):
           for j in range(i+1,4):
               for k in range(4):
                   model.AddBoolOr([
                           person_starter[names[i]][starters[k]].Not(), 
                           person_starter[names[j]][starters[k]].Not()])
                   model.AddBoolOr([person_mainCourse[names[i]][mainCourse[k]].Not(), 
                                   person_mainCourse[names[j]][mainCourse[k]].Not()])
                   model.AddBoolOr([person_deserts[names[i]][deserts[k]].Not(), 
                                   person_deserts[names[j]][deserts[k]].Not()])
                   model.AddBoolOr([person_drinks[names[i]][drinks[k]].Not(), 
                                   person_drinks[names[j]][drinks[k]].Not()])
   
    """
    ----------------------------Constraints----------------------------
    
    1.The carpaccio starter is not combined with the vegan pie as main course
    and the filet steak main course is not followed by ice cream as desert
    a. The carpaccio starter is not combined with the vegan pie as main course
    b. Filet steak main course is not followed by ice cream as desert
    """
    for name in names:
        model.AddBoolAnd([person_starter[name]["Carpaccio"].Not()]).\
            OnlyEnforceIf([person_mainCourse[name]["Vegan_Pie"]])
        model.AddBoolAnd([person_mainCourse[name]["Filet_Steak"].Not()]).\
                      OnlyEnforceIf([person_deserts[name]["Ice_Cream"]])
        
    """
    2. Emily does not have prawn cocktail or onion soup as starter
    none of the men has beer or coke to drink
    a.Emily doesnt have praws or onions
    b.James and Daniel wont have coke or beer
    """
    model.AddBoolAnd([person_starter["Emily"]["Prawn_Cocktail"].Not(), 
                     person_starter["Emily"]["Onion_Soup"].Not()])
    model.AddBoolAnd([person_drinks["James"]["Beer"].Not(), 
                     person_drinks["James"]["Coke"].Not(), 
                     person_drinks["Daniel"]["Beer"].Not(), 
                     person_drinks["Daniel"]["Coke"].Not()])
    
    """
    3. The person having prawn cocktail as starter has baked mackerel as main course and the
    filet steak main course works well with the red wine.
    """
    for name in names:
        model.AddBoolAnd([person_starter[name]["Prawn_Cocktail"]]).\
            OnlyEnforceIf([person_mainCourse[name]["Baked_Mackerel"]])
        model.AddBoolAnd([person_mainCourse[name]["Filet_Steak"]]).\
            OnlyEnforceIf([person_drinks[name]["Red_Wine"]]) 
        
    """
    4. One of the men has white wine as drink and one of the women drinks coke
    """
    model.AddBoolOr([person_drinks["James"]["White_Wine"], 
                     person_drinks["Daniel"]["White_Wine"]])
    model.AddBoolOr([person_drinks["Emily"]["Coke"], 
                     person_drinks["Sophie"]["Coke"]])
    
    """
    5. The vegan pie main always comes with mushroom tart as starter and vice versa; 
    also, the onion soup and filet steak are always served together.
    """
    for name in names:
        model.AddBoolAnd([person_mainCourse[name]["Vegan_Pie"]]).\
            OnlyEnforceIf([person_starter[name]["Mushroom_Tart"]])
        model.AddBoolAnd([person_starter[name]["Onion_Soup"]]).\
            OnlyEnforceIf([person_mainCourse[name]["Filet_Steak"]])
            
    """
    6. Emily orders beer as drink or has fried chicken as main and ice cream as desert; 
    James orders coke as drink or has onion soup as starter and filet steak as main.
    """
    model.AddBoolOr([person_drinks["Emily"]["Beer"],
                    person_mainCourse["Emily"]["Fried_Chicken"]])
    model.AddBoolAnd([person_deserts["Emily"]["Ice_Cream"]])

        
    model.AddBoolOr([person_drinks["James"]["Coke"],
                person_starter["James"]["Onion_Soup"]])
    model.AddBoolAnd([person_mainCourse["James"]["Filet_Steak"]])
        
    """
    7.  Sophie orders chocolate cake but does not drink beer nor likes fried chicken; 
    Daniel orders apple crumble for dessert but has neither carpaccio nor mushroom tart as starter.
    """
    model.AddBoolAnd([person_drinks["Sophie"]["Beer"].Not(),
                      person_mainCourse["Sophie"]["Fried_Chicken"].Not()])
    model.AddBoolAnd([person_deserts["Sophie"]["Chocolate_Cake"]])
    model.AddBoolAnd([person_starter["Daniel"]["Carpaccio"].Not(),
                      person_starter["Daniel"]["Mushroom_Tart"].Not()])
    model.AddBoolAnd([person_deserts["Daniel"]["Apple_Crumble"]])
        
    solver = cp_model.CpSolver()  
    status = solver.SearchForAllSolutions(model, SolutionPrinter(person_starter, person_mainCourse , person_deserts , person_drinks))
    print(solver.StatusName(status))

    for name in names:
        if solver.Value(person_deserts[name]["Tiramisu"]):
            print(name + ' has Tiramisu for dessert')


def task2_sudoku(sud):
    model = cp_model.CpModel()
    sud_size = sud.shape[0]
    sud_dict = {}
    
    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self, sudoku_size, sudoku_dict):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.solutions_ = 0
            self.sudokuDict_ = sudoku_dict
            self.sudoku_size_ = sudoku_size

        def OnSolutionCallback(self):
            self.solutions_ = self.solutions_ + 1
            print("\nSolution: ", self.solutions_)

            result = np.zeros((self.sudoku_size_, self.sudoku_size_)).astype(np.int32)
            for i in range(self.sudoku_size_):
                for j in range(self.sudoku_size_):
                    result[i, j] = (self.Value(self.sudokuDict_[i, j]))

            print(result)
    
    #create Int from 1,9 for non-zero sudoku slots
    for i in range(sud_size):
        for j in range(sud_size):
            if sud[i][j] != 0:
                sud_dict[i,j] = sud[i][j]
            else:
                sud_dict[i,j] = model.NewIntVar(1, sud_size,  f"sudoku_{i}_{j}")
    
    #different numbers in row
    for i in range(sud_size):
        model.AddAllDifferent([sud_dict[i,j] for j in range(sud_size)])
                    
    #differnt number in column
    for j in range(sud_size):
        model.AddAllDifferent([sud_dict[i,j] for i in range(sud_size)])
        
    grid_size = 3
    for i in range(0,sud_size,grid_size):
        for j in range(0,sud_size,grid_size):
            model.AddAllDifferent([sud_dict[i+m,j+n] for m in range(3) for n in range(3)])
    
    solver = cp_model.CpSolver()    
    solver.SearchForAllSolutions(model, SolutionPrinter(sud_size, sud_dict))
    
    


def task3(data, min_profit_margin = 2160):
        
    #part a. Loading data
    projects_df,quotes_df,dependencies_df,value = data['Projects'], data['Quotes'], data['Dependencies'], data['Value']
    #cost of all jobs in projects completed
    cost = 0 

    model = cp_model.CpModel()
    
    #part b. creating decision varialbles
    proj_dict = {}
    for p in projects_df.index.values:
        proj_dict[p] = model.NewBoolVar(p)
    
    
    pc_pair = {}
    #contractor >> month >> jobs during this month belonging to a project dict 
    contractor_project_month = {}
    #project >> month >> contractors eligible to work on the project dict 
    project_month_contractors = {}
    
    #create dict key >> contactors | value >> months(with job/project of each month)
    for contractor in quotes_df.index.values:
        contractor_project_month[contractor] = {}
        for month in projects_df.columns.values:
            contractor_project_month[contractor][month] = []
         
    #create dict key >> projects | value >> months(with contactors/project of each month)
    for p in projects_df.index.values:
        project_month_contractors[p] = {}
        for m in projects_df.columns.values:
            project_month_contractors[p][m] = []
         
    #main loop creates various decision variables (which contractor is working on which project and when)    
    #project/contractor/month/job decision varialbles
    for contractor in quotes_df.index.values:
        for job in quotes_df.columns.values:
            if str(quotes_df.loc[contractor][job]) == 'nan':
                #contractor not qualified, so pass
                pass
            else:
                for project in projects_df.index.values:
                    for month in projects_df.columns.values:
                        if str(projects_df.loc[project][month]) == 'nan':
                            # no project in this month so pass
                            pass
                        else:
                            if projects_df.loc[project][month] == job:
                                #boolean var for job which can be done by a contractor
                                pc_pair[project+'_'+contractor+'_'+month+'_'+job] = model.NewBoolVar(project+'_'+contractor+'_'+month+'_'+job) 
                                #contractors monthly job availability belonging to different projects
                                contractor_project_month[contractor][month].append(pc_pair[project+'_'+contractor+'_'+month+'_'+job])
                                #project jobs of every month and the contractors eligible to do this job 
                                project_month_contractors[project][month].append(pc_pair[project+'_'+contractor+'_'+month+'_'+job])
                                #cost calculation of the projects delivered 
                                cost += int(quotes_df.loc[contractor][job])*pc_pair[project+'_'+contractor+'_'+month+'_'+job]
    
    
    #a contractor cannot work on two jobs/projects at the same time
    for contractor, month_projects_df in contractor_project_month.items():
        # print(contractor, month_projects_df)
        # print(f"{contractor} >>>>>>>>>> {month_projects_df}")
        for m,p in month_projects_df.items():
            # print(m,p)
            model.Add(sum(p) <= 1)
    
    #Two contractors cant work on same project at same time 
    for p, mp in project_month_contractors.items():
        # print(contractor, month_projects_df)
        # print(f"{p} >>>>>>>>>> {mp}")
        for m,ps in mp.items():
            #print(m,ps)
            #If project is going ahead, exactly one contractor works on job
            if len(ps)>0:
                model.Add(sum(ps) == 1).OnlyEnforceIf(proj_dict[p])
                # Part E. Constraint #3 - If project is not taken on then 0 contractors work on any of the jobs
                model.Add(sum(ps) == 0).OnlyEnforceIf(proj_dict[p].Not())

    #dependencies bool varialbles
    for project1 in dependencies_df.index.values:
        for project2 in dependencies_df.columns.values:
            if str(dependencies_df.loc[project1][project2]) == 'required':
                #Project B can only be taken on, if also Project A is taken on
                model.AddBoolAnd([proj_dict[project2]]).OnlyEnforceIf(proj_dict[project1])
            if str(dependencies_df.loc[project1][project2]) == 'conflict':
                #Project B and Project C are mutually exclusive and cannot be both taken on 
                model.AddBoolAnd([proj_dict[project2].Not()]).OnlyEnforceIf(proj_dict[project1])

    #the difference between the value of all delivered projects and the cost of all required subcontractors
    #, is at least €2160 (profit margin)
    total_value = 0
    
    #Value = sum of all projects_df being carried out
    for p in value.index.values:
        total_value += int(value.loc[p]['Value'])*proj_dict[p]
    
    pm = total_value - cost
    model.Add( pm >= min_profit_margin)
          
    
    #CPSAT solver     
    solver = cp_model.CpSolver()    
    status = solver.Solve(model)
    sp = SolutionPrinter_task3(proj_dict, pc_pair, pm)  
    status = solver.SearchForAllSolutions(model, sp)
    print(f"There are {sp.solutions_} solutions")
     
class SolutionPrinter_task3(cp_model.CpSolverSolutionCallback):
    def __init__(self, project, project_job, pm):
        self.solutions_ = 0
        self.prft_ = pm 
        self.projects_ = project
        self.proj_jobj_ = project_job
        cp_model.CpSolverSolutionCallback.__init__(self)
        

    def OnSolutionCallback(self):
        self.solutions_ = self.solutions_ + 1
        print("\n\n\nSolution: ", self.solutions_ )
        print("Projects Contracted:")
        ps = []
        for project in self.projects_.keys():
            if self.Value(self.projects_[project]):
                ps.append(str(self.projects_[project]))

        contractor_jobs = []

        for cj in self.proj_jobj_.keys():
            if self.Value(self.proj_jobj_[cj]):
                d= str(self.proj_jobj_[cj]).split('_')
                contractor_jobs.append(d)    

        for proj in ps:
            print(f'\t--{proj}--')
            for cont_job in contractor_jobs:
                if cont_job[0] == proj:
                    print(f'\t\t- {cont_job[3]} was carried out in month {cont_job[2]} by {cont_job[1]}')
   
        print('\n\nProfit Margin is: ', self.Value(self.prft_))

    
if __name__ == "__main__":
    data = pd.read_excel('Assignment_DA_1_data.xlsx', sheet_name = None, index_col=0)
    
    sudoku_input = np.array([[0, 0, 0, 0, 0, 0, 0, 3, 0],
                            [7, 0, 5, 0, 2, 0, 0, 0, 0],
                            [0, 9, 0, 0, 0, 0, 4, 0, 0],
                            [0, 0, 0, 0, 0, 4, 0, 0, 2],
                            [0, 5, 9, 6, 0, 0, 0, 0, 8],
                            [3, 0, 0, 0, 1, 0, 0, 5, 0],
                            [5, 7, 0, 0, 6, 0, 1, 0, 0],
                            [0, 0, 0, 3, 0, 0, 0, 0, 0],
                            [6, 0, 0, 4, 0, 0, 0, 0, 5]])
    
    task1()    
    task2_sudoku(sudoku_input)
    task3(data) 
