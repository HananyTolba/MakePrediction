
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

# creat an instance of qgpr as model and plot with plotly:
#########################################
model = qgpr(xtrain,ytrain, RBF()) 
model.plotly()
<img src="assets/model.png" alt="makeprediction logo" width="600px"/>



