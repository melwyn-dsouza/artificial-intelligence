# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:18:25 2022

@author: dsouzm3
"""

from ortools.sat.python import cp_model
import numpy as np
import pandas as pd

names = ["James", "Daniel", "Emily", "Sophie"]
starters = ["Carpaccio", "Prawn_Cocktail", "Onion_Soup", "Mushroom_Tart"]
mainCourse = ["Filet_Steak", "Vegan_Pie", "Baked_Mackerel", "Fried_Chicken"]
drinks = ["Beer", "Coke", "Red_Wine", "White_Wine"]
deserts = ["Ice_Cream","Chocolate_Cake", "Apple_Crumble", "Tiramisu"]

def logic():

    model = cp_model.CpModel()
    
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
       #At least one item per person for each course
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
        # print(name, person_deserts[name]["Tiramisu"])
        if solver.Value(person_deserts[name]["Tiramisu"]):
            print(name + ' has Tiramisu for dessert')
            
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
   
logic()    
