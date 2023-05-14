import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn

# first 100 digits of pi

HUNDRED_PI = np.array( [1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,1,6,9,3,9,9,3,7,5,1,0,5,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8,1,6,4,0,6,2,8,6,2,0,8,9,9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9] )
x = np.array( [x for x in range(1, 101)] )


# Visualize our data

plt.scatter(x, HUNDRED_PI, c="r")
plt.plot(x, HUNDRED_PI)
plt.show()