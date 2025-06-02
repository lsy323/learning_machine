import numpy as np

arr = np.zeros(3)

def f(arr):
    print(id(arr))
    arr[0] = 1
    
print(arr)
f(arr)
print(arr)