import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
import hsic

#%%

test1 = []
test2 = []

test1_r = []
test2_r = []

for i in range(100):
    X = np.linspace(-3, 3, 300)
    N = np.random.normal(0, 1, 300)
    q = 1
    b = 1
    Y = X + b *X ** 3 + np.sign(N) * abs(N) ** q

    clf = SVR(kernel='rbf')
    clf.fit(Y.reshape(-1, 1), X)
    Xpred = clf.predict(Y.reshape(-1, 1))
    residue2 = X - Xpred

    a = hsic.hsic_gam(Y.reshape(-1, 1), residue2.reshape(-1, 1), 0.02)
    test1_r.append(a[0])
    test2_r.append(a[1])

    print(i)
    print(test2_r)

for i in range(100):
    X = np.linspace(-3, 3, 300)
    N = np.random.normal(0, 1, 300)
    q = 1
    b = 1
    Y = X + b * X ** 3 + np.sign(N) * abs(N)

    clf = SVR(kernel='rbf')
    clf.fit(X.reshape(-1, 1), Y)
    Ypred = clf.predict(X.reshape(-1, 1))

    residue1 = Y - Ypred
    a = hsic.hsic_gam(X.reshape(-1, 1), residue1.reshape(-1, 1), 0.02)
    test1.append(a[0])
    test2.append(a[1])

    print(i)
    print(test2_r)


plt.figure()
plt.scatter(X, residue1)
#plt.plt(X, test2_r)
plt.show()

#%%
plt.figure()
plt.scatter(X, residue2)
#plt.plt(X, test2_r)
plt.show()

plt.figure()
plt.hist(Y, density=1)
plt.figure()
plt.scatter(X, Y)
plt.plot(X, Ypred, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Data', 'Regression'])
plt.title('Q1')
plt.show()
