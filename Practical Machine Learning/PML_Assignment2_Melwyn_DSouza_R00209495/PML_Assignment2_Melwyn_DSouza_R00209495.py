# -*- coding: utf-8 -*-
"""
Name: Melwyn D Souza
Student Number: R00209495
Date: 23/12/2021
Module: Practical Machine Learning 
Lecturer: Dr Ted Scully
Course: MSc in Artificial Intelligence

"""

#basic libraries
from collections import Counter
import pandas as pd
import numpy as np
import sklearn, random, copy
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer

#preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest

#Models and experimentation
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV

#Sampling Imports
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn import pipeline

random.seed(1)

"""Main function, Uncomment particular functions to pre-process, rebalance, feature select and evaluate"""

def main():

    """Reading the data"""
    weather = pd.read_csv("weatherAUS.csv")
    
    """uncomment to plot imbalances before pre-processing data"""
    # print(weather['RainTomorrow'].value_counts())
    # colors = ["#0101DF", "#DF0101"]
    # sns.countplot('RainTomorrow', data=weather, palette=colors)
    # plt.title('Class Distributions \n (0 - No rain | 1 - Rain)', fontsize=14)
    # plt.ylabel('Counts')
    
    """Part 1 : pre-processing data and splitting as train and test sets"""
    print("\nPre processing data...")
    X,y = pre_processing(weather) #numpy arrays
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.20, random_state=1)
    
    """Experiment Part2: modiified pre_processing techniques | comment the first pre-processor if you want to test this function"""
    # print("\nPre processing data...")
    # X,y = modified_pre_processing(weather)
    # print(X.shape)
    # train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.20, random_state=1)

    """Part 1: Uncomment the lines below to test different models and find best 3 models"""
    # print("\nTesting all models pre rebalance and feature selection...")
    # best_models = test_all_models(train_features, test_features, train_labels, test_labels)
    # print("Best models are {}".format(best_models))

    """Part 2: rebalance data using random under sampler before trainign the models"""
    print("\nRebalancing data")
    train_features, train_labels = rebalanceData(train_features, train_labels, 'rus')
    
    """Part 2: feature selection | select 10 best features"""
    no_feat = 10
    indx = feature_selection(train_features, train_labels, no_feat)
    train_features, test_features = train_features[:,indx], test_features[:,indx] 

    """Part 1: Uncomment the lines below to test different models and find best 3 models after rebalancing and feature selction"""
    # print("\nTesting all models with Tomek rebalance and feature selection...")
    # best_models = test_all_models(train_features, test_features, train_labels, test_labels)
    # print("Best models are {}".format(best_models))
    
    """research Part 3: test all rebalnce methods on all models and plot f1 scores of all models"""
    # testRebalance(train_features, test_features, train_labels, test_labels)
    
    """Part 1: uncomment the code below for hyper parameter training the top best models - SVC, SGDClassifier, RandomForestClassifier and LogisticRegression""" 
    # model = SGDClassifier()
    # param_grid = {
    #         'max_iter': [200,400,600,800,1000,1200,1400],
    #         'penalty': ['l2','l1'],
    #         'n_jobs': [-1],
    #         'alpha' : [0.0001, 0.00025, 0.0005, 0.00075, 0.001]}
    # hyperOpt(train_features, test_features, train_labels, test_labels, model, param_grid)
        
    # print("Hyper parameter tuning SVC() model...")
    # model = SVC()
    # param_grid = {
    #           'gamma': ['auto','scale'],
    #           'max_iter': [100,200,-1]
    #           }
    # hyperOpt(train_features, test_features, train_labels, test_labels, model, param_grid)

    # model = RandomForestClassifier()
    # param_grid = {
    #         'max_depth': [80, 100,  None],
    #         'max_features': ['sqrt', 'log2'],
    #         'min_samples_leaf': [1,2],
    #         'n_estimators': [100, 500, 1000]}
    # hyperOpt(train_features, test_features, train_labels, test_labels, model, param_grid)
    
    # model = LogisticRegression()
    # param_grid = {
    #         'solver': ['lbfgs','liblinear','saga'],
    #         'max_iter':[50,100,200],
    #         'n_jobs':[-1]}
    # hyperOpt(train_features, test_features, train_labels, test_labels, model, param_grid)
    
    """top models with best parameters"""
    # topParamModels(train_features, test_features, train_labels, test_labels)

    """uncomment to evaluate the best model - cross fold validation of SVC()"""
    model = SVC(gamma ='scale', max_iter=-1)
    evaluation_cv(train_features,train_labels,model)

#cross fold validation with 6 folds
def evaluation_cv(X,y,model):
    result = []
    totalConfusionMatrix = np.zeros((2,2))
    
    kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=1)
    
    for train_index, val_index in kf.split(X,y):
        
        X_train = copy.deepcopy(X[train_index])
        y_train = copy.deepcopy(y[train_index])
        
        X_train, y_train = rebalanceData(X_train, y_train,technique='tomek')
        
        model.fit(X_train, y_train)
        pred = model.predict(X[val_index]) 
        
        f1  = f1_score(y[val_index],pred,average = 'micro')
        result.append(f1)
        print("F1 score is:", f1)
        confusionMatrix = confusion_matrix(y_true = y[val_index], y_pred = pred )
        totalConfusionMatrix += confusionMatrix
    print("Mean f1 score (accuracy) is {}".format(np.mean(result)))
    print(totalConfusionMatrix)

#experimentation - testing top 3 models with their best parameters found using hyper parameter tuning function
def topParamModels(train_features, test_features, train_labels, test_labels):
    print(type(train_features), train_features.shape)
    print(type(test_features), test_features.shape)
    models = [SGDClassifier(alpha = 0.0005, max_iter = 600, n_jobs = -1, penalty= 'l2'),\
              RandomForestClassifier(max_depth = 100, max_features= 'sqrt', min_samples_leaf= 1, n_estimators= 1000),\
                  LogisticRegression(max_iter=50, n_jobs= -1, solver= 'lbfgs')]
    #print confusion matrix and classification report for 3 best performing models 
    for model in  models:
        m = model.fit(train_features,train_labels)
        preds = m.predict(test_features)
        f1 = f1_score(test_labels,preds,average = 'micro')
        print("The f1 score of {} is {}".format(model,f1))
        print("Confusion matrix is as shown below\n")
        print(confusion_matrix(test_labels, preds))
        print(classification_report(test_labels, preds))
   
#experimentation - feature selection selects top 10 features out of any number of features
def feature_selection(X,y,n):
    rfc =  RandomForestClassifier(n_estimators=250, random_state=0) 
    rfc.fit(X,y) 
    imp = rfc.feature_importances_
    # print(imp)
    for i,v in enumerate(imp):
        print('Feature: %0d, Score: %.5f' % (i,v))
    sortInd = np.argsort(imp)
    largest_indices = list(sortInd[::-1][:n])
    # X_fs = X[:,largest_indices] 
    # print(X_fs)
    # return X_fs,y 
    print(largest_indices)
    return largest_indices

#hyperparameter optimization - using GridSearchCV runs different combinations of parameters and selects the best
def hyperOpt(train_features, test_features, train_labels, test_labels, model, param_grid):
   
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    grid_search = GridSearchCV(model, param_grid, scoring='f1_micro', cv = cv, n_jobs=-1)
    result = grid_search.fit(train_features, train_labels)
    best_model = result.best_estimator_
    predictions = best_model.predict(test_features)
    
    f1 = f1_score(test_labels, predictions, average='micro') 
    print("Best f1 Results: ", f1, "with parameters: ", result.best_params_)  
    
  
#research - test 8 different rebalancing methods on all 8 models | plot f1 scores of all models for each rebalance method
def testRebalance(train_features, test_features, train_labels, test_labels):
    techs = ['rus', 'ros','smote','tomek','enn', 'ada', 'svmsmote', 'smoteenn']
    names = ['SGD', 'LR', 'RFC', 'ADA', 'DTC', 'KNC', 'SVC', 'GNB']
    all_score = []
    for tech in techs:
        X_res, y_res = rebalanceData(train_features,train_labels,technique = tech)        
        best_model, n, scr = test_all_models(X_res, test_features, y_res, test_labels)
        all_score.append(scr)
    
    
    for scores,lbl in zip(all_score,techs):
        plt.plot(names, scores, label = lbl )
    
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500   
    plt.xlabel('Model Names')
    plt.ylabel('f1 scores')
    plt.title('Comparing Rebalance Techniques')
    plt.legend()
    plt.show()
       
        
#step1.1 Dealing with imablanced dataset  
def rebalanceData(X,y,technique='tomek'):
    
    # print("Before resampling data shape %s"% Counter(y))
    techs = { 'rus': RandomUnderSampler(random_state=40), 'ros': RandomOverSampler(random_state=40),\
             'smote': SMOTE(random_state=40), 'tomek':TomekLinks(n_jobs = -1),\
                 'ada': ADASYN(random_state=40), 'svmsmote':  SVMSMOTE(random_state=40),\
               'smoteenn': SMOTEENN(random_state=40), 'enn': EditedNearestNeighbours()}
        
    m = techs[technique]
    X_res, y_res = m.fit_resample(X, y)
    # print("{} sampled dataset shape {}".format(techs[technique], Counter(y_res)))
    return X_res, y_res

    
# step2 - running few models to select the best 3 models - F1 scores and confusion matrix is printed of best 3 models
def test_all_models(train_features, test_features, train_labels, test_labels):
    
    best_models = []
    
    models = [SGDClassifier(),LogisticRegression(max_iter = 500),RandomForestClassifier(),\
                AdaBoostClassifier(),DecisionTreeClassifier(),KNeighborsClassifier(),SVC(),GaussianNB()]
    
    model_scores = {} 
    scores = []
    names = []
    #iterate through models, save f1 scores
    for model in models:
        mdl = model.fit(train_features, train_labels)
        preds = mdl.predict(test_features)
        f1Score = f1_score(test_labels, preds, average='micro')
        model_scores[model] = f1Score
        scores.append(f1Score)
        names.append(type(model).__name__)
    
    model_scores =  dict(sorted(model_scores.items(),reverse=True, key=lambda item: item[1]))
    print(model_scores)
    
    #print confusion matrix and classification report for 3 best performing models 
    for i in range (3):
        temp = list(model_scores.items())[i]
        best_models.append(temp[0])
        print("Rank {} model is {} with f1 score of {}".format(i, temp[0], temp[1]))
        print("Confusion matrix is as shown below\n")
        preds = temp[0].predict(test_features)
        print(confusion_matrix(test_labels, preds))
        print(classification_report(test_labels, preds))
   
    return best_models,names,scores
    
#Step1 - pre-processing the data
def pre_processing(data):

    pd.set_option('display.max_columns', None)

    # categorical features converted to int values using label encoder
    data.drop('Date', axis='columns', inplace=True)
    data.drop('Location', axis='columns', inplace=True)
    data = copy.deepcopy(data[data['RainTomorrow'].notna()])
    
    le = LabelEncoder()
    data["RainToday"] = le.fit_transform(data["RainToday"])
    data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])
    data['WindDir9am'] = le.fit_transform(data['WindDir9am'])
    data['WindDir3pm'] = le.fit_transform(data['WindDir3pm'])
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])

    #drop rows with missing values (NA values) only when a row contains less than 20 non NA values
    data = data.dropna(thresh=20,axis=0)
    
    #multivariate imputer, this is used to add missing values after completion of above step with thresh=20
    imputer = IterativeImputer(random_state=0)
    values = imputer.fit_transform(data)
    data = pd.DataFrame(data = values, columns = data.columns)
    
    X = data.drop('RainTomorrow',axis=1)
    y = data['RainTomorrow']
    # print(X.head())
    # print(y.head())   
    
    # print(X.iloc[:,0:3])
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500
    
    #uncomment to plot imbalanced in pre-processed data 
    # p = sns.boxplot(data=X.iloc[:,15:20])
    # print(data['RainTomorrow'].value_counts())
    # colors = ["#0101DF", "#DF0101"]
    # sns.countplot('RainTomorrow', data=data, palette=colors)
    # plt.title('Class Distributions \n (0 - No rain | 1 - Rain)', fontsize=14)
    
    #scaling the data
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X) 
    X = pd.DataFrame(data = X_scaled, columns = X.columns)
    # norm_scaler = MinMaxScaler()
    # X_scaled = norm_scaler.fit_transform(X)

    #Dealing with Outliers
    clf = IsolationForest(contamination=0.01).fit(X) #create Isolation Forest object
    result = clf.predict(X) #Finding outliers, returns array of 1 (non outliers) and -1 (outliers)
    X = X[result==1] #filter all good instances
    y = y[result==1]     

    # p_clf = sns.boxplot(data=X.iloc[:,15:19])

    X = X.to_numpy()
    y = y.to_numpy()
    
    
    # print(type(X), X.shape)
    # print(type(y), y.shape)
    
    return X,y

#Modified pre-processing techniques - experimentation
def modified_pre_processing(data):
    
    pd.set_option('display.max_columns', None)
    
    #completely drop all nan rows, retained 50% of original data
    data = data.dropna()
    data = data.drop(['Date','Location'],axis=1)
    
    #manual encoding of yes no values
    le = LabelEncoder()
    data["RainToday"] = le.fit_transform(data["RainToday"])
    data["RainTomorrow"] = le.fit_transform(data["RainTomorrow"])
    
    #one-hot-encoding wind directions and concatenating
    df1 = pd.get_dummies(data["WindGustDir"], prefix='WindGustDir')
    data = pd.concat([data,df1],axis=1)
    data = data.drop(['WindGustDir'],axis=1)
    
    df1 = pd.get_dummies(data["WindDir9am"], prefix='WindDir9am')
    data = pd.concat([data,df1],axis=1)
    data = data.drop(['WindDir9am'],axis=1)
    
    df1 = pd.get_dummies(data["WindDir3pm"], prefix='WindDir3pm')
    data = pd.concat([data,df1],axis=1)
    data = data.drop(['WindDir3pm'],axis=1)


    # print(data.head())
    # print(data.shape)


    #X_train, y_target
    X = data.drop(['RainTomorrow'],axis=1)
    y = data['RainTomorrow'].astype(float)  
    
    #scale few columns in the dataframe
    ct = ColumnTransformer([
        ('std_scaler', MinMaxScaler(), ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',\
                                          'Sunshine',  'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',\
                                              'Humidity9am', 'Humidity3pm', 'Pressure9am',\
                                                  'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'])], remainder='passthrough')
    
    X_scaled = ct.fit_transform(X)
    X = pd.DataFrame(data = X_scaled, columns = X.columns)
    
    #Dealing with Outliers
    clf = IsolationForest(contamination=0.02).fit(X) #create Isolation Forest object
    result = clf.predict(X) #Finding outliers, returns array of 1 (non outliers) and -1 (outliers)
    X = X[result==1] #filter all good instances
    y = y[result==1]


    X = X.to_numpy()
    y = y.to_numpy()
    
    
    # print(type(X), X.shape)
    # print(type(y), y.shape)
    
    return X,y
 
if __name__=='__main__':
    main()