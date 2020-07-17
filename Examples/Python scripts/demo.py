import numpy as np
import matplotlib.pyplot as plt
from gpbytf.gaussianprocess import GaussianProcessRegressor as GP
from gpbytf.kernels import *
#import gpbytf
# np.random.RandomState(314)

m = 10
x = np.linspace(0, 5, m)

# np.random.seed(314)
mdl =   RBF()
print(mdl)
#mdl = RBF()

# print(mdl)
# y = mdl.simulate(x)

x = np.linspace(-3,3,100)
f = lambda s: np.sin(s)
y = f(x).ravel()

from numpy.testing import assert_almost_equal

gpr = GP()
gpr.kernel_choice = "Matern32"
#
# Test the interpolating property for different kernels.
gpr.fit(x, y)
gpr.std_noise =.0001
y_pred, y_cov = gpr.predict(x,y)
#print(y_pred,"y : ", y)
print(y_pred - y)
#assert_almost_equal(y_pred, y,decimal = 5)   
#yn = y + 0 * np.random.randn(x.size)
#gp = GP()

# res = gp.model_expression_peridct(x,yn,True)
# print(res)
# #gp.kernel_choice = "Periodic"

# gp.fit(x,yn)
# print(gp)
# #xs = np.linspace(0,12,1000)
# xs = x
# yfit, _ = gp.predict(x, yn,xs)

plt.plot(x,y,'.k')
plt.plot(x,y_pred,'r')
plt.show()

#gp.model_expression_peridct(x,yn)
# #yn = 10* yn + 100
# train_size = int(m * .7)

# (xtrain, ytrain) = x[:train_size], yn[:train_size]
# (xtest, ytest) = x[train_size:], yn[train_size:]
# model1 = GP()

# model1.kernel_choice = "periodic"
# #model1.kernel_choice = "periodic"
# #model1.fit(xtrain, ytrain)
# model1.fit(xtrain,ytrain)

# print(model1)
# yfit, _ = model1.predict(xtrain, ytrain, xtrain)
# # xs = np.linspace(30,35,300)
# xs = np.linspace(x.min(), x.max(), 1000)

# ypred, _ = model1.predict(xtrain, ytrain, xtest,ytest)
# plt.figure(figsize=(10, 5))
# plt.plot(x, yn, 'k', label="Data")
# plt.plot(x, y, 'g', label="Data")

# plt.plot(xtrain, yfit, 'b', lw=2, label="Fitted GP")
# plt.plot(xtest, ypred, 'r', label="Prediction")
# plt.title("Prediction with {}".format(model1))
# plt.legend()
# plt.show()
