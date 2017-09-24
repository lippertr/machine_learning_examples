# coding: utf-8
import numpy as np
L = [1,2,3]
A = np.array([1,2,3,])
get_ipython().magic('whos ')
l2 = [4,5,6]
l2
np.append(A, l2)
A.append(l2)
#no append like there is for lists in numpy instead think of it as vectore
'''
we must use for loops with lists [] or comprehensions which is just a for
loop shorthand and these are expensive
'''
get_ipython().magic('history ')
get_ipython().magic('whos ')
A
A**2
A*4
#most functions work element wise
np.sqrt(A)
np.log(A)
np.exp(A)
#list would require a for loop
#dot products
a = np.array([1,2]}
a = np.array([1,2])
b = np.array([2,1])
np.dot(a,b)
a.dot(b)
a*b
#element wise multiplication as expedt so arrays must be same size
np.sum(a*)
np.sum(a*b)
(a*b).sum()
np.dot(a,b)
amag = np.sqrt((a*a).sum())
amag
amag = np.linalg.norm(a)
amag
cosangle = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(cosangle)
angle
m = np.array([[1,2],[3,4]]]
m = np.array([[1,2],[3,4]]])
m = np.array([[1,2],[3,4]])
m
l = [[1,2],[3,4]]
l
m
m[0,0]
m2 = np.matrix([[1,2],[3,4]])
m2
A = np.array(m2)
A
A.T
z = np.zeros(10)
z
Z = np.zeros((10,10))
z
Z
one = np.ones((10,10))
one
np.random.random((10,10))
r =np.random.random((10,10))
r
g = np.random.randn(10,10)
g
g.mean()
g.var()
g = np.random.randn(10,10)
g.mean()
g.var()
g = np.random.randn(1000,1000)
g
g.var()
g.mean()
get_ipython().magic('save numpy_lesson ~0/')
