
<!-- ![alt text](assets/logo.png)
 -->
<img src="assets/logo.png" alt="makeprediction logo" width="300px"/>
<!-- <img src="assets/logo_1.png" alt="makeprediction logo" width="300px"/>
 -->


MakePrediction is a package for building Gaussian process models in Python. It was originally created by [Hanany Tolba].
 
 * MakePrediction is an open source project. If you have relevant skills and are interested in contributing then please do contact us (hananytolba@yahoo.com).*

Gaussian process regression (GPR):
=====================================
The advantages of Gaussian processes are:

* The prediction interpolates the observations.
* The prediction is probabilistic (Gaussian) so that one can compute empirical confidence intervals and decide based on those if one should refit (online fitting, adaptive fitting) the prediction in some region of interest.
* Versatile: different kernels can be specified. Common kernels are provided, but it is also possible to specify custom kernels.

In addition to standard scikit-learn estimator API,
* The methods proposed here are much faster than standard scikit-learn estimator API.
* The prediction method here (**predict**) is very complete compared to scikit-learn gaussian process API with many options such as:
the *sparse* context and the automatic online update of prediction.

   


## What does makeprediction do?
* Modelling and analysis time series.

* Automatic time-series prediction (forecasting) using Gaussian processes model.
* Real-Time time series prediction.
* Deploy on production the fitted (or saved) makeprediction model.

### Applications:
* Energy consumption prediction. 
* Energy demand prediction.
* Stock price prediction.
* Stock market prediction.
* ...
### Latest release from PyPI

* pip install makeprediction

### Latest source from GitHub

*Be aware that the `master` branch may change regularly, and new commits may break your code.*

[MakePrediction GitHub repository](https://github.com/HananyTolba/MakePrediction.git), run:

* pip install .

Example
==========================

Here is a simple example:

```python
from makeprediction.quasigp import QuasiGPR as qgpr
from makeprediction.invtools import date2num
from makeprediction.kernels import *
import datetime
import pandas as pd
import numpy as np

#generate time series
###############################
  
x = pd.date_range(start = datetime.datetime(2021,1,1), periods=1000, freq = '3s' )
time2num = date2num(x)

# f(x)
f = lambda dt:  100*np.sin(2*np.pi*dt/500)*np.sin(2*np.pi*dt/3003)  + 500
# f(x) + noise
y = f(time2num) + 7*np.random.randn(x.size)

# split time serie into train and test
trainSize = int(x.size *.7)
xtrain,ytrain = x[:trainSize], y[:trainSize]
xtest,ytest = x[trainSize:], y[trainSize:]

# Create an instance of the class qgpr as model and plot it with plotly:
#########################################
model = qgpr(xtrain,ytrain, RBF()) 
model.plotly()
```
<img src="assets/fig1.svg" alt="makeprediction logo" width="700px"/>

```python
#fit the model
model.fit()
```


```python
#predict with model and plot
model.predict(xtest)
model.plotly(ytest)
```
<img src="assets/model_predict.png" alt="makeprediction logo"/>


```python

#Online prediction with update
ypred = []
for i in range(xtest.size):
    yp,_ = model.predict(xtest[i],return_value = True)
    ypred.append(yp)
    data = {'x_update': xtest[i], 'y_update': ytest[i],}
    model.update(**data)


#plot 

import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
plt.plot(xtest,ytest,'b', label ='Test')
plt.plot(xtest,ypred,'r',label='Prediction')
plt.legend()
plt.savefig('fig_pred.png', dpi=300)
```
<img src="assets/fig_pred.png" alt="makeprediction logo" width="700px"/>

The previous prediction with updating, can be obtained simply by the "predict" method as follows:

```python
#prediction with update 
model.predict(xtest,ytest[:-1])
#And ploly 
model.plotly(ytest)
```
<img src="assets/model_predict_with_update.png" alt="makeprediction logo" width="700px"/>

```python
# Errors of prediction
model.score(ytest)

{'train_errors': {'MAE': 5.525659848832947,
  'MSE': 48.75753482298262,
  'R2': 0.9757047695585449},
 'test_errors': {'MAE': 6.69916209795533,
  'MSE': 68.7186589422385,
  'R2': 0.9816696384584944}}
```

