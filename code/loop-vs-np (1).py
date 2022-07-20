# copyright 2022 moshe sipper
# www.moshesipper.com

import time
import numpy as np

a = np.random.rand(10 ** 6)
b = np.random.rand(10 ** 6)

tic = time.time()
dot = 0
for i in range(10 ** 6):
    dot += a[i] * b[i]
print(f'for loop, dot product is {dot}')
toc = time.time()
loop = toc - tic

tic = time.time()
dot = np.dot(a, b)
toc = time.time()
print(f'np.dot, dot product is {dot}')
npdot = toc - tic

print(f'for loop time: {loop}')
print(f'np.dot time: {npdot}')
print()
print(f'For loop is {loop/npdot} times slower.')
