# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 21:16:25 2021

Name: Melwyn D Souza
Student Number: R00209495
Date: 24/10/2021
Module: COMP9016 - Knowledge Representation 
Lecturer: Dr Ruairi O'Reilly
Course: MSc in Artificial Intelligence

"""

import os,sys,inspect

current_dir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from agents import *
import random

from search import *
from notebook import psource, heatmap, gaussian_kernel, show_map, final_path_colors, display_visual, plot_NQueens
import time

"""
############################                Question 1                 ############################
"""

class SimpleReflexGoldDigger(Agent):
    location = [0,1]
    direction = Direction("right")
    lives = 3
    points = 20
    
    def moveforward(self, success=True):
        if not success:
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1
    
    def turn(self, d):
        self.direction = self.direction + d

    def collect(self, thing):
        return True

def programSRGD(percepts):
    '''Returns an action based on it's percepts'''
    percept, loc = percepts
    for p in percept: # the agent collects whatever it fins, its a simple reflex dumb agent
        if isinstance(p, (Gold,Pebble,Bomb)):
            return 'collect'
        if isinstance(p,Bump):
            return 'turn'
    return 'moveforward'

class ModelBasedGoldDigger(Agent): #Model checks for tick and guesses bomb location  
    
    location = [0,1]
    direction = Direction("right")
    lives = 3
    points = 20
    
    def moveforward(self, success=True):
        if not success:
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1
    
    def turn(self, d):
        self.direction = self.direction + d

    def collect(self, thing):
        return True

def programMBGD(percepts):
    '''Returns an action based on it's percepts'''
    percept,location = percepts
    for p in percept:
        if isinstance (p, Tick):
            print("There might be bomb around this location {}".format(location))
            return 'located_bomb'
        if isinstance(p, (Gold,Pebble,Bomb)):
            return 'collect'
        if isinstance(p,Bump):
            return 'turn'
    return 'moveforward'

class GoalBasedGoldDigger(Agent): #Model checks for tick and guesses bomb location  
    
    location = [0,1]
    direction = Direction("right")
    lives = 3
    points = 20
    gold_loc = []
    final_loc = [[0,0]]
    
    def moveforward(self, success=True):
        if not success:
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1
    
    def turn(self, d):
        self.direction = self.direction + d

    def collect(self, thing):
        return True

def programGBGD(percepts):
    '''Returns an action based on it's percepts'''
    percept,location = percepts
    for p in percept:
        if isinstance(p, (Gold,Pebble,Bomb)):
            return 'collect'
    return 'move_final_loc'

class Gold(Thing):
    pass

class Pebble(Thing):
    pass

class Bomb(Thing):
    pass

class Bump(Thing):
    pass

class Tick(Thing):
    pass

class GoldMine(GraphicEnvironment):
    bomb_confirmed = []
    agent_path_covered = []
    first = True
    iteration = 0
    
    def percept(self, agent):
        self.iteration += 1
        print("This is {}th step\n".format(self.iteration))
        print("Current location of agent is {}".format(agent.location))
        things = self.list_things_at(agent.location)
        print("Things at this location {}".format(things))
        loc = copy.deepcopy(agent.location) # find out the target location
        #Check if agent is about to bump into a wall
        if agent.direction.direction == Direction.R:
            loc[0] += 1
        elif agent.direction.direction == Direction.L:
            loc[0] -= 1
        elif agent.direction.direction == Direction.D:
            loc[1] += 1
        elif agent.direction.direction == Direction.U:
            loc[1] -= 1
        if loc[0] == size_x:
            print("There is a bump at {}".format(loc))
            things.append(Bump())
        if not self.is_inbounds(loc):
            print("There is a bump at {}".format(loc))
            things.append(Bump())
        agent.points -= 1
        print("Points: {}, Lives: {}".format(agent.points, agent.lives))
        return things, agent.location
    
    def find_gold(self,agent): # goal based agent searches for gold
        things=self.list_things_at(agent.location)
        i=self.width
        j=self.height
        for x in range(i):
            for y in range (j):
                gold_loc = self.list_things_at([x,y])
                if (len(gold_loc) == 1):
                    if self.list_things_at([x,y],tclass=Gold):
#                         print("Nearest gold is at {}".format(gold_loc))
                        agent.gold_loc.append([x,y])
        print("These are the gold locations {}".format(agent.gold_loc))
        return True
       
    def find_path(self,agent):
        path = [[0,0]]+agent.gold_loc
        for i in range(len(path)-1):
            from_loc = path[i]
            to_loc = path[i+1]
            x1=from_loc[0]
            y1=from_loc[1]
            x2=to_loc[0]
            y2=to_loc[1]
            for x in range( abs(x1-x2) ):
                if(x2-x1)>0:
                    agent.location[0]+=1
                    loc = copy.deepcopy(agent.location)
                    agent.final_loc.append(loc)
                elif (x2-x1)<0:
                    agent.location[0]-=1
                    loc = copy.deepcopy(agent.location)
                    agent.final_loc.append(loc)
                else:
                    pass
            for y in range( abs(y1-y2) ):
                if(y2-y1)>0:
                    agent.location[1]+=1
                    loc = copy.deepcopy(agent.location)
                    agent.final_loc.append(loc)
                elif (y2-y1)<0:
                    agent.location[1]-=1
                    loc = copy.deepcopy(agent.location)
                    agent.final_loc.append(loc)
                else:
                    pass
        return True
    
    def execute_action(self, agent, action):
        
        goal = isinstance(agent,GoalBasedGoldDigger)
        if self.first and goal:
            self.find_gold(agent)
            self.find_path(agent)
            self.first = False
        
        if ((agent.points > 0) and (agent.lives >0)):
            
            if (action == 'move_final_loc'):
                if len(agent.final_loc)>0:
                    agent.location = agent.final_loc[0]
                    agent.final_loc.pop(0)
                else:
                    print("\nCollected all golds, going home bye!\n")
                    agent.alive = False
                
            if (action == 'turn'  and agent.direction.direction=='right'):
                agent.location[1] += 1
                agent.direction.direction ='left'
                print('turn and Right at  location:{0}  and action:{1} with direction:{2}'.format(agent.location,action,agent.direction.direction) )

            elif (action == 'turn'  and agent.direction.direction=='left' ):
                agent.location[1] += 1
                agent.direction.direction ='right'
                print('turn and Left at  location:{0}  and action:{1} with direction:{2}'.format(agent.location,action,agent.direction.direction) )

            elif action == 'moveforward':
                print("Moving forward")
                print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
                agent.moveforward()
            
            elif (action == 'located_bomb'  and agent.direction.direction=='right'):
                loc = agent.location
                self.bomb_confirmed.append([loc[0]+1,loc[1]])
                agent.location[1] += 1
                agent.points += 5
                agent.direction.direction ='right'
                
                print("There are bombs in location {}".format(self.bomb_confirmed))
            
            elif (action == 'located_bomb'  and agent.direction.direction=='left'):
                loc = agent.location
                self.bomb_confirmed.append([loc[0]-1,loc[1]])
                agent.location[1] += 1
                agent.points += 5
                agent.direction.direction ='left'
                
                print(self.bomb_confirmed)
                
            elif action == "collect":
                items = self.list_things_at(agent.location, tclass=(Gold,Pebble,Bomb))
                if len(items) != 0:
                    if agent.collect(items[0]):
                        if str(items[0])[1:-1] == "Gold":
                            agent.points += 12
                        if str(items[0])[1:-1] == "Pebble":
                            agent.points -= 4
                        if str(items[0])[1:-1] == "Bomb":
                            agent.points -= 5
                            agent.lives -= 1
                            if agent.lives == 0:
                                print("Gold diger ran out of lives, agent is dead")
                                agent.alive = False
                        print("My points are {} and lives are {}".format(agent.points, agent.lives))
                        print('{} collected {} at location: {}'.format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                        self.delete_thing(items[0])
    
            #keeping a track on path
            self.agent_path_covered.append(agent.location)

        elif agent.points <= 0:
            agent.lives -= 1
            agent.points = 20
            print("Gold digger ran out of points, agent lost one life, points restored to 20")
            print("My points are {} and lives are {}".format(agent.points, agent.lives))
        else:
            print("Gold digger ran out of lives, agent dead")
            print("My points are {} and lives are {}".format(agent.points, agent.lives))
            agent.alive = False

    def is_done(self):
        winner = False
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        for agent in self.agents:
            if(agent.points >= 50):               
                winner = True
                print("I scored 50 points, I win, I wont work anymore")
        return  dead_agents or winner


#this is the size of our gold mine env
size_x = 7
size_y = 7

def q1():
    print("*"*100)    
    print("*"*100)    
    print("\nQuestion 1 - Model types\n")
    print("*"*100)    
    print("*"*100)   
    def run1(steps):
        for i in range (steps):
            gold_mine.step()
            print("----------------------------------------------")
    
    def addthings(golds, pebbles, bombs):
        random.seed(1)
        for thing in ["gold", "pebble", "bomb"]:
            if thing == 'gold':
                for i in range (0,golds):
                    gold_mine.add_thing(Gold(), [random.randint(0,size_x-1),random.randint(0,size_y-1)])
            elif thing  == 'pebble':
                 for i in range (0,pebbles):
                    gold_mine.add_thing(Pebble(), [random.randint(0,size_x-1),random.randint(0,size_y-1)])
            elif thing  == 'bomb':
                for i in range (0,bombs):
                    x = random.randint(0,size_x-1)
                    y = random.randint(0,size_y-1)
                    gold_mine.add_thing(Bomb(), [x,y])
                    gold_mine.add_thing(Tick(), [x-1,y])
                    gold_mine.add_thing(Tick(), [x+1,y])
    
    print("*"*100)    
    print("*"*100)    
    print("\nSimple Reflex Gold Digger Agent\n")
    print("*"*100)    
    print("*"*100)    
    gold_mine = GoldMine(size_x,size_y, color={'SimpleReflexGoldDigger': (0, 0, 255), "Gold": (253, 208, 23),"Bomb": (225, 0, 0), 'Pebble':(43, 27, 23), 'Tick':(128,128,128)})
    gold_digger = SimpleReflexGoldDigger(programSRGD)
    addthings(10,4,4)
    gold_mine.add_thing(gold_digger, [0,0])
    run1(15)
    print("*"*100)
    print("*"*100)        
    print("\nModel Based Reflex Gold Digger Agent\n")
    print("*"*100)    
    print("*"*100)    
    gold_mine = GoldMine(size_x,size_y, color={'ModelBasedGoldDigger': (0, 0, 255), "Gold": (253, 208, 23),"Bomb": (225, 0, 0), 'Pebble':(43, 27, 23), 'Tick':(128,128,128)})
    gold_digger = ModelBasedGoldDigger(programMBGD)
    addthings(10,4,4)
    gold_mine.add_thing(gold_digger, [0,0])
    run1(35)
    print("*"*100)    
    print("*"*100)    
    print("\nGoal Based Reflex Gold Digger Agent\n")
    print("*"*100)    
    print("*"*100)    
    gold_mine = GoldMine(size_x,size_y, color={'GoalBasedGoldDigger': (0, 0, 255), "Gold": (253, 208, 23),"Bomb": (225, 0, 0), 'Pebble':(43, 27, 23), 'Tick':(128,128,128)})
    gold_digger = GoalBasedGoldDigger(programGBGD)
    addthings(10,4,4)
    gold_mine.add_thing(gold_digger, [0,0])
    run1(19)
    

"""
############################                Question 2                 ############################
"""
def q2():    
    print("*"*100)    
    print("*"*100)    
    print("\nQuestion 2 - Search\n")
    print("*"*100)    
    print("*"*100)    
    # fucntion returns all neighbors locations
    #size_x and size_y is the grid dimension,loc_x and loc_y is the current location
    def neighbours(size_x, size_y, loc_x, loc_y):
        X = size_x
        Y = size_y
        neighbors = lambda x, y : [(x1, y1) for x1 in range(x-1, x+2)
                                    for y1 in range(y-1, y+2)
                                    if ((-1 < y <= Y) and (-1 < x <= X) and (x != x1 or y != y1) and (0 <= x1 <= X) and (0 <= y1 <= Y))]
                                       
        return neighbors(loc_x, loc_y)
    
    #Creates a mapped dict for every location
    def gold_mine_final_dict(size_x,size_y):
        mapped=dict()
        for x in range(size_x):
            for y in range(size_y):
                dict_neighbors=dict()
                list_neighbors=neighbours(size_x-1,size_y-1,x,y)
                for a in range(len(list_neighbors)):
                    dict_neighbors[ str(list_neighbors[a][0])+'_'+str(list_neighbors[a][1]) ]=1
                mapped[str(x)+'_'+str(y)]=dict_neighbors
        return mapped
    
    def gold_mine_dict(x,y):
        my_dict=dict()
        for x1 in range(x):
            for y1 in range(y):
                key=str(x1)+'_'+str(y1)
                my_dict[key]=(x1,y1)
        return my_dict
                
    
    size_x=7
    size_y=7
    # print(gold_mine_final_dict(size_x,size_y))
    gold_finding_map = UndirectedGraph(gold_mine_final_dict(size_x,size_y))
    gold_finding_map.locations = gold_mine_dict(size_x,size_y)
    # print("lalalal", gold_finding_map.locations)
    
    # node colors, node positions and node label positions
    node_colors = {node: 'white' for node in gold_finding_map.locations.keys()}
    node_positions = gold_finding_map.locations
    node_label_pos = { k:[v[0],v[1]-10]  for k,v in gold_finding_map.locations.items() }
    edge_weights = {(k, k2) : v2 for k, v in gold_finding_map.graph_dict.items() for k2, v2 in v.items()}
    
    gold_finding_graph_data = {  'graph_dict' : gold_finding_map.graph_dict,
                            'node_colors': node_colors,
                            'node_positions': node_positions,
                            'node_label_positions': node_label_pos,
                              'edge_weights': edge_weights
                          }
    
    show_map(gold_finding_graph_data)
    
    
    #class copied from aima, altered for my needs
    class InstrumentedProblem(Problem):
        def __init__(self, problem):
            self.problem = problem
            self.succs = self.goal_tests = self.states = 0
            self.found = None
        def actions(self, state):
            self.succs += 1
            return self.problem.actions(state)
        def result(self, state, action):
            self.states += 1
            return self.problem.result(state, action)
        def goal_test(self, state):
            self.goal_tests += 1
            result = self.problem.goal_test(state)
            if result:
                self.found = state
            return result
        def path_cost(self, c, state1, action, state2):
            self.pcost = self.problem.path_cost(c, state1, action, state2)
            return self.problem.path_cost(c, state1, action, state2)
        def value(self, state):
            return self.problem.value(state)
        def __getattr__(self, attr):
            return getattr(self.problem, attr)
        def __repr__(self):
            return '<{:4d}/{:4d}/{:4d}/{}>'.format(self.succs, self.goal_tests,
                                                    self.states, str(self.found)[:4])
    
    
    initial_loc='4_0'
    goal_loc ='6_6'
    print("Initial state: {}, Goal state: {}".format(initial_loc,goal_loc))
    godl_mine_problem = GraphProblem(initial_loc, goal_loc, gold_finding_map)
    
    bfts = InstrumentedProblem(godl_mine_problem)  
    s_time = time.time()
    breadth_first_tree_search(bfts)
    bfts_time = time.time()-s_time
    print("\nUninformed Search - Breadth First Tree Search")
    print("Execution Time: {}s".format(bfts_time))
    print("Path Cost: ", bfts.pcost)
    print("Nodes Explored: ",bfts.goal_tests)
    
    dfgs = InstrumentedProblem(godl_mine_problem)
    s_time = time.time()
    depth_first_graph_search(dfgs)
    dfgs_time = time.time()-s_time
    print("\nUninformed Search - Depth First Graph Search")
    print("Execution Time: {}s".format(dfgs_time))
    print("Path Cost: ", dfgs.pcost)
    print("Nodes Explored: ", dfgs.goal_tests)
    
    ids = InstrumentedProblem(godl_mine_problem)
    s_time = time.time()
    iterative_deepening_search(ids)
    ids_time = time.time()-s_time
    print("\nUninformed Search - Iterative Deepening Search")
    print("Execution Time: {}s".format(ids_time))
    print("Path Cost: ", ids.pcost)
    print("Nodes Explored :", ids.goal_tests)
    
    gbfs=InstrumentedProblem(godl_mine_problem)
    s_time = time.time()
    greedy_best_first_graph_search(gbfs, godl_mine_problem.h)
    gbfs_time = time.time()-s_time
    print("\nInformed Search - Greedy Best First Search")
    print("Execution Time: {}s".format(gbfs_time))
    print("Path Cost: ", gbfs.pcost)
    print("Nodes Explored :", gbfs.goal_tests)
    
    astar=InstrumentedProblem(godl_mine_problem)
    s_time = time.time()
    astar_search(astar)
    astar_time = time.time()-s_time
    print("\nInformed Search - A* Search")
    print("Execution Time: {}s".format(astar_time))
    print("Path Cost: ", astar.pcost)
    print("Nodes Explored :", astar.goal_tests)
    
    rbfs=InstrumentedProblem(godl_mine_problem)
    s_time = time.time()
    recursive_best_first_search(rbfs)
    rbfs_time = time.time()-s_time
    print("\nInformed Search - Recursive Best First Search")
    print("Execution Time: {}ms".format(rbfs_time))
    print("Path Cost: ", rbfs.pcost)
    print("Nodes Explored :", rbfs.goal_tests) 


def main():    
    q1()    
    q2()

if __name__=='__main__':
    main()