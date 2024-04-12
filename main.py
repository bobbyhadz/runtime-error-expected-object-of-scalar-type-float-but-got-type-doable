# RuntimeError: Expected scalar type Float but found Double

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

kernel = torch.from_numpy(np.random.random_sample((1, 3))).double()
input_ = torch.from_numpy(np.random.random_sample((1, 1, 1, 6))).double()

print(kernel)
print(input_)

model = nn.Sequential(
    nn.Conv2d(1, 1,
              (1, 3),
              stride=1,
              padding=0,
              bias=False)
)

model[0].weight.data.copy_(kernel)
A = Variable(input_)

# ✅ Call float() method in call to model()
out = model(A.float()) # 👈️
print(out.data)