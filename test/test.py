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

print(a)
print(b)

pass
