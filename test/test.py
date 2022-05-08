class A():
    def __init__(self):
        self.a = 10

    def __call__(self, *args, **kwargs):
        print(self.a)


# A()()
# a = [1, 2, 3, 4, 5, 6]
# print(a[:-2])

import torch

a = torch.Tensor([1, 2, 3, 4, 5, 6, 1, 2, 4, 5, 3, 4])
b = torch.unique(a)

# print(a)
# print(b)
import numpy as np

c = np.zeros((10,2))
kappa = []
for i in range(10):
    c[i,:] = [0.11,0.11]
    kappa.append([0.11,0.11])

print(c)
print(np.array(kappa))
