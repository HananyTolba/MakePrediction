# %%
import gpasdlm  

# %%
from gpasdlm.gp import GaussianProcessRegressor as GPR
from gpasdlm.kernels import *

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
m = 500
x = np.linspace(0,5,m)
y = Periodic(length_scale = .7, period = 1).simulate(x)
#y = np.sin(3*x)*np.sin(x)
y_noisy = y + .3*np.random.randn(x.size)
#y_noisy = 100000 + y_noisy*10000


# %%
plt.figure(figsize=(10,5))
plt.plot(x,y_noisy,'ko')
plt.plot(x,y,'b',lw=3)
plt.show()

# %%
gpr = GPR(x,y_noisy)
#gpr.kernel_choice = "Matern52"

# %%


# %%
kernelExpression = " Matern52  + Periodic*RBF   + RBF * Matern32*Linear + Periodic"
list_kernel = kernelExpression.replace(" ", "").split("+")
#print(list_kernel)
list_kernel = [x.replace(" ", "").split("*") if '*' in x else x for x in list_kernel]
list_kernel

import importlib
module_ = importlib.import_module("gpasdlm.kernels")

kernels = []
for ker_name in list_kernel:
    if isinstance(ker_name,list):
        prod_kernel_list = []
        for k in ker_name:
            class_ = getattr(module_, k)
            instance = class_()
            prod_kernel_list.append(instance)
        prod_kernel = np.prod(prod_kernel_list)
        #print(prod_kernel.label())
        kernels.append(prod_kernel)
            
    else:
        class_ = getattr(module_, ker_name)
        instance = class_()
        #print(instance.__class__.__name__)

        kernels.append(instance)
ker = sum(kernels)
print(ker)
ker

# %%
yker = ker.simulate(x)
plt.plot(x,yker)

# %%
K = RBF()*RBF() + Periodic()
K.__class__.__name__

# %%
y = RBF(length_scale=.5).simulate2d(x)
fig = plt.figure(figsize=(10,5) )
plt.imshow(y,cmap='jet')
plt.show()

# %%
Y = RBF().simulate3d(x,time = np.linspace(0,3,50),time_kernel=Periodic(length_scale=1) + RBF())
#Y = Y + .5*np.random.randn(*Y.shape)
print(Y.shape)

# %%
plt.plot(Y.mean(axis=0).mean(axis=0))
#plt.plot(Y.mean(axis=2).mean(axis=1))
#plt.plot(Y.mean(axis=2).mean(axis=0))

# %%

import time
from IPython.display import clear_output

for i in range(Y.shape[2]):
    clear_output(wait=True)
    fig = plt.figure(figsize=(8,5) )
    plt.imshow(Y[:,:,i],cmap ="jet",vmin=Y.min(),vmax=Y.max())
    plt.title("Gaussian Process 3d simulation (spatiotemporal)")
    plt.colorbar()
    plt.clim(vmin=-3,vmax=3)

    plt.show() 
    time.sleep(0.01)

# %%
