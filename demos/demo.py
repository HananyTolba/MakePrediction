#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Author = "Hanany Tolba"
#01/02/2021

# __author__ = "Hanany Tolba"



from makeprediction.quasigp import QuasiGPR as qgpr
from makeprediction.invtools import date2num
from makeprediction.kernels import *
import datetime
import pandas as pd
import numpy as np

# %%
f = lambda dt:  100*np.sin(2*np.pi*dt/500)*np.sin(2*np.pi*dt/3003)  + 500  
x = pd.date_range(start = datetime.datetime(2021,1,1), periods=1000, freq = '3s' )
time2num = date2num(x)

y = f(time2num) + 7*np.random.randn(x.size)


# %%
trainSize = int(x.size *.7)
xtrain,ytrain = x[:trainSize], y[:trainSize]
xtest,ytest = x[trainSize:], y[trainSize:]

# %%
model = qgpr(xtrain,ytrain, RBF()) 
model.plotly()

# %%
model.fit()


# %%
model.predict(xtest)
model.plotly(ytest)

# %%
#prediction with update
ypred = []
for i in range(xtest.size):
    yp,_ = model.predict(xtest[i],return_value = True)
    ypred.append(yp)
    data = {'x_update': xtest[i], 'y_update': ytest[i],}
    model.update(**data)
    


# %%
import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
plt.plot(xtest,ytest,'b', label ='Test')
plt.plot(xtest,ypred,'r',label='Prediction')
plt.legend()
plt.grid()
plt.savefig('fig_pred.svg', dpi=700)

# %%
#The simple way:
model.predict(xtest,ytest[:-1])
model.plotly(ytest)


# %%
### save the model:
model_path = 'saved_model'
model.save(model_path)



# %%
#serving the saved_model 
#################################
# execute realtime_db.py before executing the following cells. 
saved_model.deploy2dashbord('live_db.csv')

#We can change the prediction horizon :
###  for example :  saved_model.deploy2dashbord('live_db.csv',prediction_horizon = 3)

# %%
