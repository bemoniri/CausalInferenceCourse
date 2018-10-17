#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  Behrad Moniri  
@title: Causal Inference HW 1 - Question 1
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import seaborn as sns; sns.set(style="white", color_codes=True)
import csv
import hsic


#%% Linear Gaussian
X = np.random.normal(0, 0.5, 300000)
N = np.random.normal(0, 0.5, 300000)
Y = X + N

g = sns.jointplot(X, Y)

# P(Y|X=+-1)

Condp = Y[abs(X[:] - 1) < 0.1]
Condn = Y[abs(X[:] + 1) < 0.1]
plt.figure()
plt.hist(Condn, 50, density=1)
plt.hist(Condp, 50, density=1)
plt.legend(['P(Y|X=+1)','P(Y|X=-1)'])
plt.title('Linear and Gaussian')
plt.show()

# P(X|Y=+-2)


Condp = X[abs(Y[:] - 1) < 0.1]
Condn = X[abs(Y[:] + 1) < 0.1]
plt.figure()
plt.hist(Condn, 50, density=1)
plt.hist(Condp, 50, density=1)
plt.legend(['P(X|Y=+1)','P(X|Y=-1)'])
plt.title('Linear and Gaussian')
plt.show()

#%% Non-linear Gaussian
X = np.random.normal(0, 1, 300000)
N = np.random.normal(0, 10, 300000)
q = 1
b = 5
Y = X + b *X ** 3 + np.sign(N) * abs(N) ** q

g = sns.jointplot(X, Y)

# P(Y|X=+-1)

Condp = Y[abs(X[:] - 1) < 0.1]
Condn = Y[abs(X[:] + 1) < 0.1]
plt.figure()
plt.hist(Condn, 50, density=1)
plt.hist(Condp, 50, density=1)
plt.legend(['P(Y|X=+1)','P(Y|X=-1)'])
plt.title('Linear and Gaussian')
plt.show()

# P(X|Y=+-20)


Condp = X[abs(Y[:] - 20) < 1]
Condn = X[abs(Y[:] + 20) < 1]
plt.figure()
plt.hist(Condn, 50, density=1)
plt.hist(Condp, 50, density=1)
plt.legend(['P(X|Y=+20)','P(X|Y=-20)'])
plt.title('Linear and Gaussian')
plt.show()


#%% ‌Backwards
test_reverse = []

X = np.linspace(-3, 3, 300)
N = np.random.normal(0, 1, 300)
q = 1
b = 1
Y = X + b *X ** 3 + np.sign(N) * abs(N) ** q

lin_regressor = LinearRegression()
poly = PolynomialFeatures(5)

Y_transform = poly.fit_transform(Y.reshape(-1,1))
lin_regressor.fit(Y_transform,X) 

x_preds = lin_regressor.predict(Y_transform)
residue_reverse = X - x_preds

plt.figure()
plt.plot(x_preds, Y)
plt.scatter(X, Y)
plt.show()

plt.figure()
plt.scatter(Y, residue_reverse)
plt.show()


#%% Forward

test = []

X = np.linspace(-3, 3, 300)
N = np.random.normal(0, 1, 300)
q = 1
b = 1
Y = X + b *X ** 3 + np.sign(N) * abs(N) ** q

lin_regressor = LinearRegression()
poly = PolynomialFeatures(5)

X_transform = poly.fit_transform(X.reshape(-1,1))
lin_regressor.fit(X_transform,Y) 

Y_preds = lin_regressor.predict(X_transform)
residue = Y - Y_preds

plt.figure()
plt.scatter(X, Y)
plt.plot(X, Y_preds, color='red')
plt.show()



plt.figure()
plt.scatter(X, residue)
plt.show()



#%% Real Data - Old Faithfull

d = {}
eruptions = []
waiting = []
with open('Old_Faithful_Geyser_Data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        data = row[:][0].split()
        eruptions.append(float(data[1]))
        waiting.append(float(data[2]))
eruptions = np.asarray(eruptions)
waiting = np.asarray(waiting)


plt.figure()
plt.scatter(eruptions, waiting)
plt.ylabel('Eruption Time')
plt.xlabel('Waiting')
plt.title('Old Faithful Geyser Data')

#%% if eruption time is cause

lin_regressor = LinearRegression()
poly = PolynomialFeatures(4)
eruptions_transform = poly.fit_transform(eruptions.reshape(-1,1))
lin_regressor.fit(eruptions_transform, waiting)
waiting_preds = lin_regressor.predict(eruptions_transform)
residue1 = waiting_preds - waiting

plt.figure()
plt.scatter(eruptions, waiting_preds, color = 'red')
plt.scatter(eruptions, waiting)
plt.title('Regression Assuming Eruption Time is the Cause')
plt.ylabel('Waiting Time')
plt.xlabel('Eruption Time')


plt.figure()
plt.scatter(eruptions, residue1, color = 'red')
plt.title('Regression Assuming Eruption Time is the Cause')
plt.ylabel('Error')
plt.xlabel('Eruption Time')
plt.show()
print(hsic.hsic_gam(residue1.reshape(-1,1), eruptions.reshape(-1,1), 0.02))


#%% if waiting is cause

lin_regressor = LinearRegression()
poly = PolynomialFeatures(4)
waiting_transform = poly.fit_transform(waiting.reshape(-1,1))
lin_regressor.fit(waiting_transform, eruptions) 
eruptions_pred = lin_regressor.predict(waiting_transform)
residue2 = eruptions_pred - eruptions
plt.figure()
plt.scatter(eruptions_pred, waiting, color = 'red')
plt.scatter(eruptions, waiting)
plt.title('Regression Assuming Waiting Time is the Cause')
plt.xlabel('Eruptions')
plt.ylabel('Waiting Time')

plt.figure()
plt.scatter(waiting, residue2, color = 'red')
plt.title('Regression Assuming Waiting Time is the Cause')
plt.xlabel('Waiting Time')
plt.ylabel('Error')
plt.show()
print(hsic.hsic_gam(residue2.reshape(-1,1), waiting.reshape(-1,1), 0.02))


#%% Real data - abalone

d = {}
Length = []
Rings = []
with open('abalone.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        Length.append(float(row[1]))
        Rings.append(int(row[-1]))
Rings = np.asarray(Rings)
Length = np.asarray(Length)

plt.figure()
plt.scatter(Rings, Length)
plt.show()


#%% if number of rings is cause


clf = SVR()
clf.fit(Rings.reshape(-1,1), Length)
residue1 = clf.predict(Rings.reshape(-1,1)) - Length
Length_preds = clf.predict(Rings.reshape(-1,1))
plt.figure()
plt.scatter(Rings, Length, color = 'red')
plt.scatter(Rings, Length_preds)
plt.title('Regression Assuming Number of Rings is the Cause')
plt.ylabel('Length')
plt.xlabel('Number of Rings')


plt.figure()
plt.scatter(Rings, residue1, color = 'red')
plt.title('Regression Assuming Number of Rings is the Cause')
plt.ylabel('Error')
plt.xlabel('Number of Rings')
plt.show()


#%% if length is cause
clf = SVR()
clf.fit(Length.reshape(-1,1), Rings)
residue2 = clf.predict(Length.reshape(-1,1)) - Rings
Rings_preds = clf.predict(Length.reshape(-1,1))


plt.figure()
plt.scatter(Rings, Length, color = 'red')
plt.scatter(Rings_preds, Length)
plt.ylabel('Length')
plt.xlabel('Number of Rings')


plt.figure()
plt.scatter(Length, residue2, color = 'red')
plt.ylabel('Error')
plt.xlabel('Length')
plt.show()

#%% HSIC

a = hsic.hsic_gam(residue2.reshape(-1,1), Length.reshape(-1,1), 0.05)
b = hsic.hsic_gam(residue1.reshape(-1,1), Rings.reshape(-1,1), 0.05)
print(a)
print(b)