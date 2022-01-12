# -*- coding: utf-8 -*-
"""
Name: Melwyn D Souza
Student Number: R00209495
Date: 4/12/2021
Module: COMP9016 - Knowledge Representation 
Lecturer: Dr Ruairi O'Reilly
Course: MSc in Artificial Intelligence
Assignment - 2
"""
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from probability import *
from probabilistic_learning import *
from learning import train_test_split
from notebook import *

import pandas as pd
import numpy as np

"""###################################### q 1.2 ##########################################"""
def q12():
    """Creating Bayes Nodes"""
    aiNode = BayesNode('AI', '', 0.25)
    employedNode = BayesNode('Employed', '', 0.75)
    trNode = BayesNode('Traffic',['Employed'],{True : 0.85, False: 0.15})
    ffNode = BayesNode('FossilFuel',['Traffic'],{True : 0.75, False: 0.25})
    reNode =  BayesNode('RenewableEnergy',['AI'],{True : 0.45, False: 0.55})
    gwNode = BayesNode('GlobalWarming', ['FossilFuel', 'RenewableEnergy'],{(True, True): 0.65, (True, False): 0.95, (False, True): 0.1, (False, False): 0.75})
    
    print("\n{}Question 1.2{}".format("*"*20,"*"*20))
    print("Querying bayes nodes with Fossilfuel-True, Traffic-True, Renewable Energy-False, AI-True, Employed-True")    
    print("GLobal warming probabilty:", gwNode.p(True, {'FossilFuel':True, 'Traffic':True, 'RenewableEnergy':False, 'AI':True,'Employed':True}))
    print()
    """Creating Bayes Network"""
    bn = BayesNet([
        ('AI', '', 0.60),
        ('RenewableEnergy','AI',{T : 0.45, F: 0.55}),
        ('Employed', '', 0.85),
        ('Traffic','Employed',{T : 0.85, F: 0.15}),
        ('FossilFuel','Traffic',{T : 0.75, F: 0.25}),
        ('GlobalWarming', 'FossilFuel RenewableEnergy',{(T, T): 0.65, (T, F): 0.95, (F, T): 0.1, (F, F): 0.75}),                                                      
    ])
    
    """querying bayes net"""
    print("Querying bayes nodes with Fossilfuel-True, Traffic-True, Renewable Energy-False, AI-False, Employed-True")    
    ans_dist = enumeration_ask('GlobalWarming', {'FossilFuel': True, 'Traffic': True, 'RenewableEnergy':False, 'Employed':True, 'AI':False}, bn)
    print("Global warming probability:", ans_dist[True])

"""###################################### q 1.3.1 ##########################################"""

def q13(): 
    print("\n{}Question 1.3{}".format("*"*20,"*"*20))
    attributes=['BI_Rads','Age','Shape','Margin','Density' ,'Target']
    
    mamo_data =  DataSet(name = "mammographic_masses",attr_names=attributes)
    mamo_data.remove_examples("?")
    
    mamo_data, mamo_test  =  train_test_split(mamo_data,test_split=0.2)
    mamo_data, mamo_test  = DataSet(examples = mamo_data ,attr_names=attributes), DataSet(examples = mamo_test ,attr_names=attributes)
    # print(len(mamo_data.examples), len(mamo_test.examples))
    
    benign_freq, malign_freq = 0,0
    benign_fv, malign_fv = [], []
    for feature in mamo_data.examples:
        if feature[-1] == 0:
            benign_fv.append(feature)
        elif feature[-1] == 1:
            malign_fv.append(feature)
    total_instances = len(mamo_data.examples)
    
    """prior probabilities, #b = 427, m = 403, total_inst = 830"""
    prior_prob_benign = len(benign_fv)/total_instances
    prior_prob_malign = len(malign_fv)/total_instances
    print("Prior Probabilties")
    print("Benign: {}, Malignant: {}".format(prior_prob_benign, prior_prob_malign))
    
    """find the freq of elements in an array"""
    def freq(arr):
        frequency = defaultdict(int)
        for i in arr:
            frequency[i] += 1
        return frequency
    
    """probabilities of evidence"""
    prob_evidence = []
    print("\nProbability of evidence")
    for attr_name, attr_ind in zip(mamo_data.attr_names,mamo_data.inputs):
        print('Feature: ',attr_name)
        column = [feature[attr_ind] for feature in mamo_data.examples]
        feature_freq = freq(column)
        feature_freq = {k: v/total_instances for k, v in feature_freq.items()}
        prob_evidence.append(feature_freq)
    print(prob_evidence)
    
    """probabilities of likelihood - benign"""
    prob_likelihood_benign = []
    print("\nProbability of likelilhood (Benign)")
    for attr_name, attr_ind in zip(mamo_data.attr_names, mamo_data.inputs):
        print('Feature: ',attr_name)
        column = [feature[attr_ind] for feature in benign_fv]
        feature_freq = freq(column)
        feature_freq = {k: v/len(benign_fv) for k, v in feature_freq.items()}
        prob_likelihood_benign.append(feature_freq)
    print(prob_likelihood_benign)
    
    """probabilities of likelihood - malignant"""
    prob_likelihood_malign = []
    print("\nProbability of likelilhood (Malignant)")
    for attr_name, attr_ind in zip(mamo_data.attr_names, mamo_data.inputs):
        print('Feature: ',attr_name)
        column = [feature[attr_ind] for feature in malign_fv]
        feature_freq = freq(column)
        feature_freq = {k: v/len(malign_fv) for k, v in feature_freq.items()}
        prob_likelihood_malign.append(feature_freq)
    print(prob_likelihood_malign)
    
    # test_vector = [4,63,1,1,3] #benign
    # # test_vector = [4,59,4,4,3] #malign
    # benign_numerator, malign_numerator, denominator = prior_prob_benign, prior_prob_malign, 1
    # for feature, index in zip(test_vector, mamo_data.inputs):
    #     benign_numerator = benign_numerator * prob_likelihood_benign[index][feature] 
    #     malign_numerator = malign_numerator * prob_likelihood_malign[index][feature]
    #     denominator = denominator * prob_evidence[index][feature]
    
    # prob_benign = benign_numerator/denominator
    # prob_malign = malign_numerator/denominator
    # total = prob_benign+prob_malign
    # prob_benign = (prob_benign*100)/(total)
    # prob_malign = (prob_malign*100)/(total)
    
    # print("The test feature instance is bening: {} %".format(round(prob_benign)))
    # print("The test feature instance is malignant: {} %".format(round(prob_malign)))

    """###################################### q 1.3.2 ##########################################"""
    """Naive Bayes Algorithm"""
    prediction = []
    for test_vector in mamo_test.examples:
        benign_numerator, malign_numerator, denominator = prior_prob_benign, prior_prob_malign, 1
        for feature, index in zip(test_vector, mamo_test.inputs):
            try:
                benign_numerator = benign_numerator * prob_likelihood_benign[index][feature] 
                malign_numerator = malign_numerator * prob_likelihood_malign[index][feature]
                denominator = denominator * prob_evidence[index][feature]
            except:
                pass
            prob_benign = benign_numerator/denominator
            prob_malign = malign_numerator/denominator
            total = prob_benign+prob_malign
            prob_benign = (prob_benign*100)/(total)
            prob_malign = (prob_malign*100)/(total)
        if prob_benign > prob_malign:
            prediction.append(0)
            # print("*"*50)
            # print("Test Vector:{}\nTrue Class:{}\nPredicted Class:{}".format(test_vector[:-1],test_vector[-1],0))
        else:
            prediction.append(1)
            # print("*"*50)
            # print("Test Vector:{}\nTrue Class:{}\nPredicted Class:{}".format(test_vector[:-1],test_vector[-1],0))
    
    count = 0
    for test_v, pred in zip(mamo_test.examples, prediction):
        if (test_v[-1]==pred):
            count += 1
        
    print("*"*50)
    print("Number of train instances:",len(mamo_data.examples))
    print("Number of test instances:",len(mamo_test.examples))
    print("The Naive Bayes model accuracy is {} %".format((count/len(prediction))*100))
    
def main():
    q12()
    q13()
    
if __name__=='__main__':
    main()
