import numpy as np

import math
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt

  
def noisy_sign(x):
  sigma=np.sqrt(1/3)
  n = tdist.Normal(torch.tensor([0.0]), torch.tensor([sigma])).sample([x.nelement()]).reshape(x.shape)
  return torch.sign(x + n)

def sr_sign(x):
  n = tdist.Uniform(-1,1).sample([x.nelement()]).reshape(x.shape)
  return torch.sign(x + n)
  
def cast_int2(x):
    A = x.int()
    A = torch.relu(A+2)
    A = torch.relu(3-A)
    A = 1 - A
    A = A.float()
    return A

def cast_intn(x,n):
    a = (2 << n) >> 3
    b = a << 1
    c = b << 1
    A = (x*a).int()
    A = torch.relu(A+b)
    A = torch.relu(c-1-A)
    A = b-1 - A
    A = A.float()
    A = A/(a*1.0)
    return A
  
def cast_even_intn(x,n):
    a = (2 << n) >> 3
    b = a << 1
    c = b << 1
    A = (x*a).int()
    A = torch.relu(A+b)
    A = torch.relu(c-A)
    A = b - A
    A = A.float()
    A = A/(a*1.0)
    return A

##runfile('C:/Users/Yaniv/Downloads/np_calc.py', "-T 50 -c -p 0.5,1.3,1,1 -I 1,1")