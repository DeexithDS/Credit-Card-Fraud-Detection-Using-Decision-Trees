# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:13:03 2020

@author: DEEXITH REDDY
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV


##Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn import ensemble



##Loading the dataset and checking data types, null values.

df=pd.read_csv("C:/Users/DEEXITH REDDY/Desktop/Projects/Credit Card Decision Trees/Credit_Card.csv")
df.info()

df.info()
df.isna().sum()
 df.isnull().sum().sum()

##There are no null values. Hence, we go on with analysis directly.

##Boxplot visulaization:

sns.set(style="whitegrid")
ax = sns.boxplot(x=df["LIMIT_BAL"])
ax = sns.boxplot(x=df["AGE"])
ax = sns.boxplot(x="SEX", y="LIMIT_BAL", data=df)
ax = sns.boxplot(x="MARRIAGE", y="LIMIT_BAL", data=df)
ax = sns.boxplot(x="AGE", y="LIMIT_BAL", data=df)
ax = sns.lineplot(x="AGE", y="LIMIT_BAL", data=df)

##Setting the index
df.set_index('ID')
df=df.drop('ID',axis=1)
 
 
##Checking value counts:
 
 df['EDUCATION'].value_counts()
 
 ##We see there are 14 "0" values. We delete them.
 ##Since 5 and 6 are unknown, we replace 6 with 5
 
df=df[df.EDUCATION !=0]
df['EDUCATION'] = df['EDUCATION'].replace([6],5)

##Deleting 0 values from marriage
df['MARRIAGE'].value_counts()
df=df[df.MARRIAGE !=0]

##Visualizing the payment status
sns.countplot(x="PAY_0",data=df)
sns.countplot(x="PAY_2",data=df)
sns.countplot(x="PAY_3",data=df)
sns.countplot(x="PAY_4",data=df)
sns.countplot(x="PAY_5",data=df)
sns.countplot(x="PAY_6",data=df)

##We see that majority are using revolving credit. Followed by payin.g duly and no consumption

##We see that that all values are withn 9
df['PAY_0'].value_counts()
df['PAY_2'].value_counts()
df['PAY_3'].value_counts()
df['PAY_4'].value_counts()
df['PAY_5'].value_counts()
df['PAY_6'].value_counts()

##Exploring Bill amount:

df['BILL_AMT1'].describe()

##There are negative values, which we delete.

 df = df[(df['BILL_AMT1']>0) & (df['BILL_AMT2']>0) & (df['BILL_AMT3']>0) & (df['BILL_AMT4']>0) & (df['BILL_AMT5']>0) & (df['BILL_AMT6']>0)]
 
 ##Checking that the bill amount is not more than credit of i million
 
 df['BILL_AMT2'].describe()
 df['BILL_AMT3'].describe()
 df['BILL_AMT4'].describe()
 df['BILL_AMT5'].describe()
 df['BILL_AMT6'].describe()
 
 
 ##Hence, we finally have 22341 rows.
 
 ##We will use decision trees
 
 ##Separating the columns of target variable and predictor variables
 
 y=df['default.payment.next.month']
 df=df.drop('default.payment.next.month',axis=1)
 
 ##Train test splits 
 
 X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=42)


##Basic Tree

tree_clf = DecisionTreeClassifier(max_depth=4)
tree_clf.fit(X_train,y_train)
y_pred=tree_clf.predict(X_test)
accuracy_score(y_test, y_pred)

##0.82

names=df.columns


export_graphviz(
tree_clf,
out_file=None,
feature_names=names,
class_names['Not default','Default'],
filled=True
)


for name, importance in zip(df.columns, tree_clf.feature_importances_):
 print(name, importance)


##Perform grid search
 
param_grid = {'max_depth': np.arange(3, 10),
             'criterion' : ['gini','entropy'],
             'max_leaf_nodes': [5,10,20,100],
             'min_samples_split': [2, 5, 10, 20]}

# create the grid

grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5, scoring= 'accuracy')


grid_tree.fit(X_train, y_train)

##Printing best estimator
print(grid_tree.best_estimator_)

##Best score of that tree
print(np.abs(grid_tree.best_score_))

##0.82

##Random tree classifier

param_grid1 = {'n_estimators': [200, 400, 600, 1000],
              'criterion': ['entropy', 'gini'],
              'class_weight' : ['balanced'], 'n_jobs' : [-1]}
grid_forest = GridSearchCV(RandomForestClassifier(), param_grid1, scoring = 'accuracy', cv=5)
grid_forest.fit(X_train, y_train)
print(grid_forest.best_estimator_)
print(np.abs(grid_forest.best_score_))

#0.823

##Gradient Booster
param_gri2 = {'n_estimators': [200,300],
              'learning_rate' : [0.5, 0.75, 1.0]}
grid_gbc = GridSearchCV(GradientBoostingClassifier(),param_gri2, scoring = 'accuracy', cv=5)
grid_gbc = grid_gbc.fit(X_train, y_train)
print(grid_gbc.best_estimator_)
print(grid_gbc.best_score_)

##0.80 

##Adaboost

param_gri3 = {'n_estimators': [200,300]}          
   
grid_ada = GridSearchCV(AdaBoostClassifier(),param_gri3, scoring = 'accuracy', cv=5)
grid_ada = grid_ada.fit(X_train, y_train)
print(grid_ada.best_estimator_)
print(grid_ada.best_score_)   

#0.822

##Extra Trees Classifier

param_gri4 = {'max_depth': np.arange(3, 10),
             'criterion' : ['gini','entropy'],
             'max_leaf_nodes': [5,10,20,100],
             'min_samples_split': [2, 5, 10, 20]}
grid_etc = GridSearchCV(ExtraTreesClassifier(),param_gri3, scoring = 'accuracy', cv=5)
grid_etc = grid_etc.fit(X_train, y_train)
print(grid_etc.best_estimator_)
print(grid_etc.best_score_)

#0.81



##Hence, Random Forest is the best.

tree_clf = grid_forest.best_estimator_
tree_clf.fit(X_train,y_train)
y_pred=tree_clf.predict(X_test)
accuracy_score(y_test, y_pred)

##0.83 Accuracy

for name, importance in zip(df.columns, tree_clf.feature_importances_):
 print(name, importance
       
#Payment at first month is most important
 
