'''
these are some notes from udemy lazyprogrammer on numpy stack (which is numpy, pandas, matplotlib, scipy)
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
r = np.random.randn(10000)
r
plt.hist(r, bins=1000)
plt.hist(r, bins=1000)
plt.hist(r, bins=100)
r = 10*np.rand.randn(10000) +5
r = 10*np.random.randn(10000) +5
r
#if we want random from guassian but need mean different than 0 and std differnt than one use
#this where 10 is the mean and 5 is the std
r = 10*np.random.randn(10000) +5
plt.hist(r, bins=100)
#add more dimensions
r = np.random.randn(10000, 2)
r
r.shape
#verify with scatter plot
plt.scatter(r[:,0], r[:,1])
r[:,0]
r[:,1]
r.shape
plt.scatter(r[:,0], r[:,1])
r[:,1] = 5*r[:,1] +2
plt.scatter(r[:,0], r[:,1])
plt.axis('equal')
#coverience and the plot
cov = np.array([[1,0.8], [0.8, 3]])
cov
cov.shape
mu = nparray([0,2])
mu = np.array([0,2])
mu
r = mvn.rvs(mean=mu, cov=cov, size=1000)
r = scs.rvs(mean=mu, cov=cov, size=1000)
import scipy as sc
r = sc.rvs(mean=mu, cov=cov, size=1000)
from scipy.stats import multivariate_normal as mvn
r = mvn.rvs(mean=mu, cov=cov, size=1000)
r
plt.scatter(r[:,0], r[:,0])
r_numpy = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
r_numpy = r
r_numpy = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
plt.scatter(r_numpy[:,0], [:,1])
plt.scatter(r_numpy[:,0], r_numpy[:,1])
#exercise 1
A = np.array([[.3, .6, .1], [ .5, .2, .3], [.4, .1, .5]])
A
v = [1/3, 1/3, 1/3]
vA**25
v*A**25
r
r.shape
x = np.array([[np.random.randint(1,101) for i in range(5)]])
x
y = np.array([[np.random.randint(1,101) for i in range(5)]])
plt.plot(x,y)
y
plt.scatter(x,y)
x = np.linspace(0,10, 100)
y = sin(x)
plt.plot(x,y)
x = np.linspace(0,10, 5)
y = sin(x)
plt.plot(x,y)
plt.plot(x,y)
plt.scatter(x,y)
plt.scatter(x,y)
save scipyLearn.py ~0/
save scipyLearn ~0/
get_ipython().magic('save scipyLearn ~0/')
