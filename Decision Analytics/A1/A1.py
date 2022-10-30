# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:18:25 2022

@author: dsouzm3
"""

from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
import copy


def Logic():
    names = ["James", "Daniel", "Emily", "Sophie"]
    starters = ["Carpaccio", "Prawn_Cocktail", "Onion_Soup", "Mushroom_Tart"]
    mainCourse = ["Filet_Steak", "Vegan_Pie", "Baked_Mackerel", "Fried_Chicken"]
    drinks = ["Beer", "Coke", "Red_Wine", "White_Wine"]
    gender = ["Boy", "Girl"]

