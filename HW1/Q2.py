#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  Behrad Moniri  
@title: Causal Inference HW 1 - Question 2 
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
import hsic

#%% Load Graph Data
dag_data = np.genfromtxt('dag.csv', delimiter=',')


#%%    X -> Z <- Y
clf = SVR()
clf.fit(dag_data[1:,2:4], dag_data[1:,4])
residue = clf.predict(dag_data[1:,2:4]) - dag_data[1:,4]

plt.figure
plt.subplot(121)
plt.suptitle('X -> Z <- Y')
plt.scatter(residue, dag_data[1:,2])
plt.xlabel('X')
plt.ylabel('Residue')
plt.subplot(122)
plt.scatter(residue, dag_data[1:,3])
plt.xlabel('Y')
plt.ylabel('Residue')
plt.show()

#%%    X -> Z -> Y

clf = SVR()
clf.fit(dag_data[1:,3:4], dag_data[1:,4])
residue = clf.predict(dag_data[1:,3:4]) - dag_data[1:,4]


plt.figure
plt.title('X -> Z -> Y')
plt.scatter(residue, dag_data[1:,2])
plt.xlabel('X')
plt.ylabel('Residue')
plt.show()

#%%    X <- Z <- Y

clf = SVR()
clf.fit(dag_data[1:,2:3], dag_data[1:,4])
residue = clf.predict(dag_data[1:,2:3]) - dag_data[1:,4]


plt.figure
plt.title('X <- Z <- Y')
plt.scatter(residue, dag_data[1:,3])
plt.xlabel('Y')
plt.ylabel('Residue')
plt.show()


#%% W -> X
clf = SVR()
clf.fit(dag_data[1:,1:2], dag_data[1:,2])
residue = clf.predict(dag_data[1:,1:2]) - dag_data[1:,2]

plt.figure
plt.scatter(residue, dag_data[1:,1:2])
plt.title('W -> X')
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()


#%% W -> Y
clf = SVR()
clf.fit(dag_data[1:,1:2], dag_data[1:,3])
residue = clf.predict(dag_data[1:,1:2]) - dag_data[1:,3]

plt.figure
plt.scatter(residue, dag_data[1:,1:2])
plt.title('W -> Y')
plt.xlabel('W')
plt.ylabel('Residue')
plt.show()

#%% X -> W <- Y

clf = SVR()
clf.fit(dag_data[1:,2:4], dag_data[1:,1])
residue = clf.predict(dag_data[1:,2:4]) - dag_data[1:,1]

plt.figure
plt.scatter(residue, dag_data[1:,2])
plt.suptitle('X -> W <- Y')
plt.xlabel('x')
plt.ylabel('Residue')
plt.show()

plt.figure()
plt.suptitle('X -> W <- Y')
plt.scatter(residue, dag_data[1:,3])
plt.xlabel('y')
plt.ylabel('Residue')
plt.show()

