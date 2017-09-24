#matrix inverse
A = np.array([[1,2], [3,4]])
aInv = np.linalg.inv(a)
aInv
aInv.dot(a)
'''
determinant
'''
np.linalg.det(a)

#diagonal
np.diag(a)
b = np.array([11,12],[23,24]])
b = np.array([[11,12],[23,24]])

#you can insert stuff if you do it carefully and you CANNOT say b.insert(.....)
np.insert(b, [2],[44,55], axis =0)
b = np.insert(b, [2],[44,55], axis =0)
np.diag(b)
np.diag([1,2])
vec1 = np.array([1,2])
vec2 = np.array([3,4])
vec1
np.inner(vec1,vec2)
vec1.dot(vec2)
np.outer(vec1, vec2)
vec1
#trace of 2d vector or more is the sum of the diagnol 
#we can
vec1
a
a.diag().sum()
np.diag(a).sum()
np.trace(a)
X = np.random.randn(100,3)
X
#coveriant
np.cov(X)
cov = np.cov(X)
cov.shape
#so that's wrong since we exped 3 so do again the 'right' way now
cov = np.cov(X.T)
cov.shape
cov
#eighen values and vectors
np.linalg.eigh(cov)
np.linalg.eig(cov)
#solving a system of equations
A
b
b = np.array([1,2])
b
x = np.linalginv(A).dot(b)
x = np.linalg.inv(A).dot(b)
x
#so common there's a function for it
# USE this function instead of other methods
x = np.linalg.solve(A,b)
x
x = np.linalg.inv(A).dot(b)
x
x = np.linalg.inv(A).dot(b)
x
x = np.linalg.solve(A,b)
x
A = np.array([[1,1],[1.5,4]])
B = np.array([2200,5050])
np.linalg.solve(A,B)

#poor way to import a csv but it works
x = []
x = []
for line in open('data_2d.csv'):
    row = line.split(',')
    print(row)
    sample = [float(x) for x in row]
    print(sample)
    x.append(sample)
