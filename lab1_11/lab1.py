import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])

def normalize(a):
    return (a - a.mean(axis=0)) / a.std(axis=0)

print(normalize(a))

#print('a / 9 =', a / 9, sep='\n')