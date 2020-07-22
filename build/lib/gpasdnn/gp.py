#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Hanany Tolba"
# __copyright__ = "Copyright 2020, Guassian Process by Deep Learning Project"
# __credits__ = ["Hanany Tolba"]
# __license__ = "Apache License 2.0"
# __version__ = "0.0.1"
# __maintainer__ = "Hanany Tolba"
# __email__ = "hananytolba@yahoo.com"
# __status__ = "Production/stable"


'''This module is for Gaussian Process Regression simulation fitting and prediction.'''

#import matplotlib.pyplot as plt
import importlib
import copy


from gpasdnn.invtools import fast_pd_inverse as pdinv
from gpasdnn.invtools import inv_col_add_update, inv_col_pop_update
import gpasdnn.kernels as kernels
from gpasdnn.kernels import *

import inspect
import pandas as pd
from collections import Counter
import os
import glob
import sys
#import site
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error as mse
#from sklearn.metrics import r2_score
import numpy as np
#from numpy import loadtxt
import tensorflow as tf
from tensorflow.keras.models import load_model

# tf mp
from scipy.interpolate import interp1d
#from tensorflow import keras
#


from tensorflow.keras import backend as K
from scipy import signal
from tqdm import tqdm
from scipy.signal import correlate
from numpy import argmax, mean, diff, log, nonzero
from termcolor import *
import colorama
colorama.init()




# cprint('hello'.upper(), 'green')


__all__ = ["GaussianProcessRegressor","RBF","Matern52",
"Matern32", "Matern12", "Periodic", "Polynomial","Periodic"]

#import site


#path = site.getsitepackages()[0]

#path_list = list(filter(lambda x: x.endswith('site-packages') ,sys.path))

import sysconfig
#path_list_1 = site.getsitepackages()
#print("path_list_1",path_list_1)
path = sysconfig.get_paths()["purelib"]
#path = path_list[0]
#print("path", path)

path = os.path.join(path,'gpasdnn/gpTFModels')
#path = path + '/keras_600'


#path = "../keras_600"
file_list = glob.glob(path + "**/*.hdf5", recursive=True)
#file_name = os.listdir(path)
file_name = [f for f in os.listdir(path) if not f.startswith('.')]

#file_name.remove('.DS_Store')
class_names = ['Linear', 'Linear + Periodic', 'Periodic', 'Polynomial',
       'Polynomial + Periodic', 'Polynomial + Periodic + Stationary',
       'Polynomial + Stationary', 'Stationary', 'Stationary + Linear + Periodic',
       'Stationary + Periodic']


#path_predict_model = path + '/predict_gpr_model.hdf5'
#model_expression = keras.models.load_model("/Users/tolba/Desktop/gpasdnn/keras_600/predict_gpr_model.hdf5")

#model_expression = load_model(path_predict_model)
#probability_model = tf.keras.Sequential([model_expression, tf.keras.layers.Softmax()])

path_periodic = os.path.join(path, "periodic_1d.hdf5")
#print(path_periodic)
K.clear_session()
newModel = load_model(path_periodic)


path_periodic_noise = os.path.join(path, "iid_periodic_300.hdf5")
#print(path_periodic)
K.clear_session()
model_periodic_noise = load_model(path_periodic_noise)



# import inspect
# import kernels
# K_list = [m for m in inspect.getmembers(kernels, inspect.isclass) if
# m[1].__module__ == 'kernels']

# Kernels_class = [Linear(),
#                  Periodic(),
#                  RBF(),
#                  Matern12(),
#                  Matern32(),
#                  Matern52(),
#                  Exponential(),
#                  Cosine(),
#                  ]

# Kernel_names = list(map(lambda x: x.__class__.__name__.lower(),
# Kernels_class))

Kernels = inspect.getmembers(kernels, inspect.isclass)
Kernels_class_instances = [m[1]() for m in Kernels]
Kernels_class_names = [m[0].lower() for m in Kernels]
# print("instances",Kernels_class_instances,"names",Kernels_class_names)

# print(Kernels_class)


class GaussianProcessRegressor():
    

    """
    Gaussian process regression (GPR)::
    =====================================
    This implementation use a tensorflow pretrained model  to estimate the Hyperparameters of 
    a GPR model and then fitting the data with.

    The advantages of Gaussian processes are:
        * The prediction interpolates the observations.
        * The prediction is probabilistic (Gaussian) so that one can compute empirical confidence intervals and decide based on those if one should refit (online fitting, adaptive fitting) the prediction in some region of interest.
        * Versatile: different kernels can be specified. Common kernels are provided, but it is also possible to specify custom kernels.

    In addition to standard scikit-learn estimator API,
       * The methods proposed here are much faster than standard scikit-learn estimator API.
       * The prediction method here "predict" is very complete compared to scikit-learn estimator API with many options such as:
         "sparse" and the automatic online update of prediction.

   
    Attributes::
    ==================
    xtrain : array-like of shape (n_samples, 1) or (n_samples, ) list of object
        Feature vectors or other representations of training data (also
        required for prediction).
    ytrain : array-like of shape (n_samples, 1) or (n_samples, ) or list of object
        Target values in training data (also required for prediction)
    kernel : 
        Kernel instance, the default is RBF instance.
    sigma_n : 
        Noise standard deviation (std) of the gaussian white noise, default is 0.01.
    model :
        The pretrained tensorflow model, which corresponds to the choice of the kernel function, by default it is that of RBF.


    Methods::
    ==========================
    The class 'GaussianProcessRegressor', is a model of Gaussian process regression which has several  methods. The most important of its methods are: 
    **fit**, **predict**, **simulate** and **kernel_choice**.
    The method **fit** :  estimates the hyperparameters of the kernel function of the model.
    The method **predict** : allows prediction with the GaussianProcessRegressor model.
    The method **simulate** : allows the simulation of the realizations of GaussianProcess according to the various kernels.
    The method **kernel_choice** : allows to choose a kernel as apriori. By default it is the RBF function which is considered.


    Examples:
    ===============
            >>> from gpasdnn.gp import GaussianProcessRegressor as GPR
            >>> from gpasdnn.kernels import RBF, Periodic
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> import time

            >>> x = np.linspace(0,8,1000)
            >>> y = np.sin(x)*np.sin(2*x)+ np.cos(5*x)
            >>> yn = y  + .2*np.random.randn(x.size)

            >>> plt.figure(figsize=(10,6))
            >>> plt.plot(x,yn,'ok',label="Data")
            >>> plt.plot(x,y,'b',lw=2,label='True gaussian process')
            >>> plt.legend()
            >>> plt.show()


            >>> #Defining a Gaussian process model
            >>> model = GPR(x,yn)
            >>> start = time.time()

            >>> #Fit a Gaussian process model
            >>> model.fit()
            >>> xs = np.linspace(8,12,200)

            >>> #Prediction::
            >>> yfit,_ = model.predict() # same as  yfit,_ = model.predict(x)
            >>> ypred,_ = model.predict(xs)

            >>> #Get time taken to run fit and predict
            >>> elapsed_time_lc=(time.time()-start)
            >>> print(f"The time taken to run fit and predict methods is {elapsed_time_lc} seconds")

            >>> plt.figure(figsize=(10,6))
            >>> plt.plot(x,yn,'ok',label="Data")
            >>> plt.plot(x,y,'b',label='True gaussian process')
            >>> plt.plot(x,yfit,'r--',lw=2,label="Training")
            >>> plt.plot(xs,ypred,'r',label='Prediction')
            >>> plt.legend()
            >>> plt.show()
            >>> print(model)




    """

#     path_list = list(filter(lambda x: x.endswith('site-packages') ,sys.path))
#     path = path_list[0]
#     path = path + '/gpasdnn/keras_600'
#     #path = path + '/keras_600'

#     file_list = glob.glob(path + "**/*.hdf5", recursive=True)
#     file_name = [f for f in os.listdir(path) if not f.startswith('.')]

#     Kernels = inspect.getmembers(kernels, inspect.isclass)
#     Kernels_class_instances = [m[1]() for m in Kernels]
#     Kernels_class_names = [m[0].lower() for m in Kernels]


#     class_names = ['Linear', 'Linear + Periodic', 'Periodic', 'Polynomial',
#        'Polynomial + Periodic', 'Polynomial + Periodic + Stationary',
#        'Polynomial + Stationary', 'Stationary', 'Stationary + Linear + Periodic',
#        'Stationary + Periodic']


#     path_predict_model = path + '/predict_gpr_model.hdf5'
# #model_expression = keras.models.load_model("/Users/tolba/Desktop/gpasdnn/keras_600/predict_gpr_model.hdf5")

#     model_expression = keras.models.load_model(path_predict_model)
#     probability_model = tf.keras.Sequential([model_expression, tf.keras.layers.Softmax()])



    def __init__(self,xtrain,ytrain, kernel=RBF(), model=None, sigma_n=.01):
        '''
        Constructor of the Gaussian process regression class:<
        It has five attributes:
        - _kernel: an instance of a kernels class (RBF,Matern32,...)
        - _model: is a pretrained tensorflow model
        - _sigma_n: is the standard deviation of the gaussian white noise.
        '''
        self._xtrain = xtrain
        self._ytrain = ytrain

        self._kernel = kernel
        path = file_list[file_name.index('rbf_1d.hdf5')]
        K.clear_session()
        best_model = load_model(path)
        self._model = best_model
        self._sigma_n = sigma_n
        # self._pred = pred




    def __repr__(self):
        return "Instance of class '{}'".format(self.__class__.__name__)
    
    def __str__(self):
        message_print = "GPR model with kernel {} and noise-std = {}"
        return message_print.format(self._kernel,round(float(self._sigma_n),4))

    @property
    def kernel_choice(self):
        '''
        kernel_choice is for choose the kernel function.
        '''
        return self._kernel

    @property
    def std_noise(self):
        return self._sigma_n

    @std_noise.setter
    def std_noise(self, sigma_n):
        self._sigma_n = sigma_n

    def get_hyperparameters(self):
        d = self._kernel.__dict__
        parms = dict()
        for cle,valeur in d.items():
            if cle != "_hyperparameter_number":
                #print(cle.lstrip('_') + " = ", valeur)
                parms[cle.lstrip('_')] = valeur
        return parms

        ##return getattr(self._kernel)

    def set_hyperparameters(self,dic):
        for cle in self._kernel.__dict__.keys():
            if cle != "_hyperparameter_number":
                setattr(self._kernel, cle, dic[cle.lstrip('_')])

    hyperparameters = property(get_hyperparameters,set_hyperparameters)



    @kernel_choice.setter
    def kernel_choice(self, kernel):
        '''
        This method allows to choose the type of the covariance or kernel function. For the moment only the functions:
                 "Linear",
                 "Periodic",
                 "RBF",
                 "Matern12",
                 "Matern32",
                 "Matern52",
                 "Polynomial",
        are available. Other kernels functions  and composition of kernel, will be added in the next version of this package.
        '''

        kernel = kernel.lower()
        if kernel not in Kernels_class_names:

            raise_alert = "'{}' is not a valid kernel choice, You must choose a valid kernel function.".format(
                kernel)
            raise ValueError(raise_alert)
        else:
            location = Kernels_class_names.index(kernel)
            str_kernel = kernel + '_1d.hdf5'
            path_model = file_list[file_name.index(str_kernel)]
            K.clear_session()  # pour accelerer keras model load
            best_model = load_model(path_model)
            #try:
                #best_model = load_model(path_model)
            #except:
            #    best_model = keras.models.load_model(path_model)

            self._kernel = Kernels_class_instances[location]
            self._kernel.set_length_scale(1)
            if self._kernel.__class__.__name__ == "Periodic": 
                self._kernel.set_period(1)


            self._model = best_model


    # @classmethod
    # def model_change(cls,model_path,kernel_name="RBF"):
    #     str_kernel = kernel_name.lower() + '_1d.hdf5'

    #     cls.file_list[cls.file_name.index(str_kernel)] = model_path


    #@staticmethod
    def line_transform(self,Y):
        '''
        This function transforms any line or segment [a, b] to segment [-3, 3] and
         then returns the parameters of the associated model.
        '''
        names_cls = self._kernel.__class__.__name__
        #Y = self._xtrain
        if names_cls == "Periodic":
            X = np.linspace(-1, 1, Y.size)
        #elif names_cls == "ChangePointLinear":
        #    X = np.linspace(0, 1, Y.size)

        else:
            X = np.linspace(-3, 3, Y.size)

        modeleReg = LinearRegression()

        modeleReg.fit(Y, X)
        res = modeleReg.predict(Y)
        return res, float(modeleReg.intercept_), float(modeleReg.coef_)


    

    
    @staticmethod
    def check_x(x):

        if isinstance(x, np.ndarray):
            
            if x.ndim != 1:
                raise ValueError(
                    "The input (space or time)  must be a 'numpy' vector type.")
        
        elif isinstance(x, list):
            x = np.array(x)
        else:
            raise ValueError(
                "The input (space or time)  must be a 'numpy' vector (1d) or list.")

        return

    @staticmethod
    def x_type(x):  # check_x(x):
        x = np.array(x)
        return x.ravel()

    @staticmethod
    def _sorted(x,y,index = False):
        x = np.array(x)
        y = np.array(y)

        ind = np.argsort(x.ravel(), axis=0)
        if index:
            return x[ind], y[ind], ind
        else:
            return x[ind], y[ind]






    






    def p_fit(self,x,y):




        #x,y = self._xtrain, self._ytrain

        ystd = y.std()
        y = (y - y.mean()) / y.std()

        
        n = y.size
#=================================================================
        x_interp = np.linspace(-1, 1, 300)

        x_transform, a, b = self.line_transform(x.reshape(-1, 1))

        y_interp = np.interp(x_interp, x_transform, y)

        period_est_ = newModel.predict(y_interp.reshape(1,x_interp.size)) 
        period_est_ = period_est_.ravel()

        #===============noise std===========
        noise_std = model_periodic_noise.predict(y_interp.reshape(1,x_interp.size))
        noise_std = noise_std.ravel()
        self._sigma_n = noise_std[0]*ystd
        #print("period_est_ (methode1) :",period_est_)
#=================================================================
        # x_interp = np.linspace(-1,1,300)
        # y_interp = np.interp(x_interp, x, y)

        
        # parms_est_list = newModel.predict(y_interp.reshape(1,x_interp.size)) 
        # parms_est_list = parms_est_list.ravel()

        #plt.plot(x_transform,y)
        #plt.plot(x_interp,y_interp)
        #plt.show()
#=================================================================
        #print("period_est_ (methode2) :",parms_est_list)

        #parms_est_list = parms_est_list[-1]
        #return parms_est_list

        return period_est_



    def period_est(self):
        pp = []
        x,y = self._xtrain, self._ytrain

        hyp = self.p_fit(x,y)
        if int(.001*x.size >10):
            quantile_size = np.linspace(.001,1,200)


        elif int(.01*x.size >10):
            quantile_size = np.linspace(.01,1,200)
        elif int(.1*x.size >10):
            quantile_size = np.linspace(.1,1,200)

        else:
            quantile_size = np.linspace(.2,1,100)


        quantile_size = x.size * quantile_size

        ytest_list = list(map(lambda s: y[:int(s)], quantile_size))
        period_list = []
        for yt in ytest_list:
            pEstimated = self.p_fit(np.linspace(-1,1,yt.size),yt)[-1]*yt.size/x.size

            #print(f"Estimated period is : {pEstimated}")
            period_list.append(pEstimated)

        #plt.plot(period_list)
        #plt.show()
        #print(period_list_[:2])
        #period_list = list(map(lambda s: s[-1], period_list_))
        

        

     
        period_list = np.array(period_list)
        
        diff_test = np.abs(np.diff(period_list))
            
            #diff_test = diff_test[diff_test < np.quantile(diff_test, 0.01)]
        pest = period_list[np.argmin(diff_test)]
        zz = period_list[np.abs(period_list-pest)<.01]
        #print(hyp)
        #hyp[-1] = round(zz.mean(),3)
        hyp[-1] = round(pest,3)

        

        hyp_dict = dict(zip(["length_scale","period"],hyp))
        hyp_dict["variance"] = y.std()**2
        self.set_hyperparameters(hyp_dict)


    def fit(self):
        '''
        This method allows the estimation of the hyperparameters of the GPR model.
        '''
        xtrain, ytrain = self._xtrain, self._ytrain

        xtrain,ytrain = self._sorted(xtrain,ytrain)

        # cprint("Fit a Gaussian Process to data ...".upper(), 'green')
        xtrain = self.x_type(xtrain)
        ytrain = self.x_type(ytrain)


        meany, stdy = ytrain.mean(), ytrain.std()
        ytrain = (ytrain - meany) / stdy


        if self._kernel.__class__.__name__ == "Periodic":

            self.period_est()
            #self.fit_periodic(xtrain, ytrain,robust=True,p0=p0)
            #self._kernel._variance = ytrain.std()
            #self._kernel.set_length_scale(parmsfit_by_sampling[0])
           # self._kernel.set_period(parmsfit_by_sampling[1])

           # ls, p, std_noise = periodic_fit(xtrain,ytrain)
           # #self.get_hyperparameters()
           # hyparms_ = {"length_scale":ls,"period":p,"variance":stdy**2}
           # self.set_hyperparameters(hyparms_)
           # #self._sigma_n = std_noise
           # self._sigma_n = std_noise * stdy


        else:

            #xtrain = self.x_type(xtrain)
            #ytrain = self.x_type(ytrain)

            
            # self._kernel, self.model = self.kernel_choice(kernel = kernel)

            if self._kernel.__class__.__name__ == "Linear":
                x_interp = np.linspace(-3, 3, 600)
                #x_interp = np.linspace(x1[0], x1[-1], 600)
            elif self._kernel.__class__.__name__ == "Polynomial":
                x_interp = np.linspace(-3, 3, 300)

            # elif self._kernel.__class__.__name__ in ["Cosine","Exponential"]:
            #     x_interp = np.linspace(-3, 3, 300)



            else:
                x_interp = np.linspace(-3, 3, 600)

            
            xtrain_transform, a, b = self.line_transform(xtrain.reshape(-1, 1))

            y_interp = np.interp(x_interp, xtrain_transform, ytrain)
            #y_interp = y_interp.ravel()

            #try:
            #    parmsfit_by_sampling = self._model.predict(y_interp.reshape(y_interp.size, 1))
            #except:
            parmsfit_by_sampling = self._model.predict(y_interp.reshape(1,y_interp.size))


            # try:
            #     parms_pred = self._model.predict(
            #         y_interp.reshape(y_interp.size, 1))
            # except BaseException:
            #     parms_pred = self._model.predict(
            #         y_interp.reshape(1, y_interp.size))
            #_l = parms_pred.tolist()
            parmsfit_by_sampling = parmsfit_by_sampling.ravel()
            if self._kernel.__class__.__name__ == "Linear":
                #self._kernel.set_length_scale(parmsfit_by_sampling[0]) # mon modif nouvelle
                #yp,_ = self.predict(xtrain,ytrain)
                #sig = (ytrain - yp).std()
                self.std_noise = parmsfit_by_sampling[0]*stdy
                self._kernel._variance = stdy**2


                #self.std_noise = parmsfit_by_sampling[1]
            # elif self._kernel.__class__.__name__ == "ChangePointLinear":
            #     self._kernel.set_hyperparameters({'length_scale': parmsfit_by_sampling[0],
            #         'length_scale1': parmsfit_by_sampling[1],
            #         'steepness':.0001,
            #         'location': parmsfit_by_sampling[2]})

            elif self._kernel.__class__.__name__ == "Polynomial":
                self._kernel.set_length_scale(parmsfit_by_sampling[0])
                #yp,_ = self.predict(xtrain,ytrain)
                #sig = (ytrain - yp).std()
                #self.std_noise = sig
                self.std_noise = parmsfit_by_sampling[1]*stdy

                self._kernel._variance = stdy**2

            # elif self._kernel.__class__.__name__ in ["Cosine", "Exponential"]:
            #     self._kernel.set_length_scale(parmsfit_by_sampling[0])
               




            else:
                self._kernel.set_length_scale(parmsfit_by_sampling[0])
                self.std_noise = parmsfit_by_sampling[1]*stdy
                self._kernel._variance = stdy**2

            # if self._kernel.__class__.__name__ == "Linear":
                # self._kernel.set_length_scale(parmsfit_by_sampling[0])
                # self.std_noise = parmsfit_by_sampling[1]
            # else:
                # self._kernel.set_length_scale(parmsfit_by_sampling[0])
                # self.std_noise = parmsfit_by_sampling[1]




            #parmsfit_by_sampling = _l[0]
           


        #return parmsfit_by_sampling  # ,   y_interp.shape

    def predict_periodic(self, xtest):
        '''
        This method allows prediction via a periodic kernel model.
        it will be called when the "predict" method is used.
        '''
        xtrain, ytrain = self._xtrain, self._ytrain

        xtrain = self.x_type(xtrain)
        ytrain = self.x_type(ytrain)
        xtest = self.x_type(xtest)


        meany, stdy = ytrain.mean(), ytrain.std()

        ytrain = (ytrain - meany) / stdy

        xtrain_transform, a, b = self.line_transform(
            xtrain.reshape(-1, 1))

        xtest_transform = b * xtest + a

        K_noise = self._kernel.count(
                    xtrain_transform,
                    xtrain_transform)

        np.fill_diagonal(K_noise, K_noise.diagonal() + self._sigma_n**2)

        invK_noise = pdinv(K_noise)


        Kstar = self._kernel.count(
                xtest_transform,
                xtrain_transform)




        y_pred_test = Kstar.T @ invK_noise @ ytrain
        ypred = (stdy * y_pred_test + meany)

        std_stars = self._kernel.count(
                xtest_transform,
                xtest_transform).T

        std_pred_test = std_stars - Kstar.T @ invK_noise @ Kstar

        ypred = (stdy * y_pred_test + meany)

        std_pred_test = np.sqrt(std_pred_test.diagonal())

        return ypred, std_pred_test

    def mean_predict(self, xtest):
        '''
        This method allows long term prediction over an xtest position vector (time or space) via GPR model.
        It will be called when the "predict" method is used. It doesn't need to have updates of new data at regular horizon i.e. (ytest not necessary).
        '''

        xtrain, ytrain = self._xtrain, self._ytrain


        xtrain = self.x_type(xtrain)
        ytrain = self.x_type(ytrain)
        xtest = self.x_type(xtest)

        if self._kernel.__class__.__name__ == "Periodic":

            (ypred, std_pred_test) = self.predict_periodic(xtrain, ytrain, xtest)
        else:

            meany, stdy = ytrain.mean(), ytrain.std()
            # meanyt, stdyt = yt.mean(), yt.std()

            ytrain = (ytrain - meany) / stdy

            xtrain_transform, a, b = self.line_transform(xtrain.reshape(-1, 1))

            xtest_transform = b * xtest + a

            K_noise = self._kernel.count(
                    xtrain_transform,
                    xtrain_transform)

            np.fill_diagonal(K_noise, K_noise.diagonal() + self._sigma_n**2)

            invK_noise = pdinv(K_noise)

            

            Kstar = self._kernel.count(
                xtest_transform,
                xtrain_transform)



            y_pred_test = Kstar.T @ invK_noise @ ytrain
            ypred = (stdy * y_pred_test + meany)
            std_stars = self._kernel.count(
                xtest_transform,
                xtest_transform).T

            std_pred_test = std_stars - Kstar.T @ invK_noise @ Kstar

            ypred = (stdy * y_pred_test + meany)
# a verifier l'expression de variance ici
            std_pred_test = np.sqrt(std_pred_test.diagonal())

        return ypred, std_pred_test

    

  






    def predict_by_block(self, xt, yt, fast=True, option=None):
        '''
        This method uses the "invupdate" function for the fast calculation of the GPR prediction.
        It will be called in the "predict" method if "yt" is no None.
        By default "fast" equals True. If "fast" equals False, "invupdate", will not be used and the classic
        numpy matrix inverse function will be used.
        '''

        x_train, y_train = self._xtrain, self._ytrain


        # if yt is None:
        #     #message = "forecasting started ..."
        #     #cprint(message.title(), "green")

        #     (res, std_pred) = self.mean_predict(x_train, y_train, xt)
        #     res = res.tolist()
        #     std_pred = std_pred.tolist()
        # else:

        x_train = self.x_type(x_train)
        y_train = self.x_type(y_train)
        xt = self.x_type(xt)
        yt = self.x_type(yt)

        meany, stdy = y_train.mean(), y_train.std()
        # meanyt, stdyt = yt.mean(), yt.std()

        y_train = (y_train - meany) / stdy
        yt = (yt - meany) / stdy

        xtrain_transform, a, b = self.line_transform(
            x_train.reshape(-1, 1))

        xtest_transform = b * xt + a
        n = xt.size

        res = []
        std_pred = []
        # x_inds = xt.argsort()
        # xt = xt[x_inds[0::]]
        # yt = yt[x_inds[0::]]
        #K = self._kernel.count(xtrain_transform, xtrain_transform) + \
        #    self._sigma_n**2 * np.eye(xtrain_transform.size)

        K = self._kernel.count(
                    xtrain_transform,
                    xtrain_transform)

        np.fill_diagonal(K, K.diagonal() + self._sigma_n**2)



        invK = pdinv(K)
        # cprint("Gaussian Process forecasting in progress ...".upper(), 'green')
        # message = "forecasting started ..."

        for i in tqdm(range(n)):
            # for i in range(n):

            x = self._kernel.count(xtest_transform[i], xtrain_transform)
            # print(x.shape)
            r = self._kernel.count(xtest_transform[i], xtest_transform[i])
            y_pred_test = np.dot(np.dot(x.T, invK), y_train)
            #print(f"la dimension de la matrice Kernel est {invK.shape}")


            std_stars = self._kernel.count(xtest_transform[i],xtest_transform[i]).T





            std_pred_test = std_stars - np.dot(np.dot(x.T, invK), x)

            # print(y_pred_test)
            # j'ai modifiée ceci

            xtrain_transform = np.hstack((xtrain_transform, xtest_transform[i]))
            #x_train = np.hstack((x_train[1:], xt[i]))
            if (i<n-1):
                y_train = np.hstack((y_train, yt[i]))
            
            # xtrain_transform = np.hstack(
            #     (xtrain_transform, xtest_transform[i]))
            # if (i<n-1):
            #     y_train = np.hstack((y_train, yt[i]))
            

            # M=np.block([[SquareExponentialKernel(xtrain, xtrain, sin_sigma_f,
            # sin_l),x.reshape(-1,1)],[x.reshape(1,-1), r + sigma_n**2]])
            if fast:
                invK = inv_col_add_update(invK, x, r + self._sigma_n**2)
                if option is None:
                    option = False
                if option:
                    xtrain_transform = xtrain_transform[1:]
                    if (i<n-1):
                        y_train = y_train[1:]
                    invK = inv_col_pop_update(invK,0)

            else:

                K = np.block([[K, x], [x.T, r + self._sigma_n**2]])
                invK = np.linalg.inv(K)



            #ce que j'ai ajouter    
            #invK = self._inv_remove_update(invK,0)
            #self.fit(x_train,y_train)
            ####

            res.append(stdy * y_pred_test + meany)

            std_pred.extend(np.sqrt(std_pred_test))

        return np.array(res), np.array(std_pred)

    def predict(self, xt=None, yt=None, horizon=None,option=None, sparse = None, sparse_size=None):
        '''


        As we know the GPR model can do prediction and interpolation. This method calculates prediction (extrapolation) as well as interpolation. Are used:
        Case 1 :
        self.predict(x_train, y_train, xt) gives us the prediction of the data on the xt location vector.
 
        Case 2: According to the prediction horizon horizon =1,2,3,... in this case we want to calculate
        the predictions of the GPR model on the vector of xt leases, with the difference that we will make regular updates (autoregressive) each time a future data yt is observed.   In this case the syntax is :
         
        self.predict(x_train, y_train, xt, yt, horizon=h)

        '''
        x_train, y_train = self._xtrain, self._ytrain

        if sparse is None:
            sparse = False

        if  sparse:
            try:
                if sparse_size is None:
                    sparse_size = max(600,int(.2*x_train.size))
                Index = np.random.choice(y_train.size, sparse_size, replace=False)
                xtrain = x_train
                Index = np.sort(Index.ravel(), axis=None)

                x_train, y_train = x_train[Index], y_train[Index] 
                self._xtrain, self._ytrain = x_train, y_train
            except ValueError as e:
                print(f"The data size is very small '{y_train.size}'. The sparse option requires the data size to be larger than 1000.")



        #x_train,y_train, index = self._sorted(x_train,y_train,index = True)


        if xt is None:
            if sparse:
                xt = np.copy(xtrain)
            else:
                xt = x_train

        x_train = self.x_type(x_train)
        y_train = self.x_type(y_train)
        xt = self.x_type(xt)

        if self._kernel.__class__.__name__ == "Periodic":

            res, std_pred = self.predict_periodic(xt)
            # res = res.tolist()
            # std_pred = std_pred.tolist()
            y_pred, std_pred = np.array(res), np.array(std_pred)

        elif self._kernel.__class__.__name__ in ["Linear","Polynomial"]:

            (res, std_pred) = self.mean_predict(xt)
            y_pred, std_pred = np.array(res), np.array(std_pred)




        else:

            if yt is None:
                message = "Long term forecasting started ..."
                # cprint(message.title(), "green")

                (res, std_pred) = self.mean_predict(xt)
                res = res.tolist()
                std_pred = std_pred.tolist()
            else:

                yt = self.x_type(yt)


                if horizon is None or horizon == 1:
                    horizon = 1
                    message = "Satrt a one-step ahead forecasting with GPR model (kernel={}).".format(
                        self._kernel.__class__.__name__)
                    cprint(message, "green")
                    res, std_pred = self.predict_by_block(xt, yt,option=option)

                else:

                    #meany, stdy = y_train.mean(), y_train.std()
                # meanyt, stdyt = yt.mean(), yt.std()

                    #y_train = (y_train - meany) / stdy

                    #yt = (yt - meany) / stdy
                    n = xt.size
                    res = []
                    std_pred = []

                    message = "Satrt a multi-step ahead forecasting (horizon={}) with GPR model (kernel={}).".format(
                        horizon, self._kernel.__class__.__name__)

                    chunks_xt = [xt[h:h + horizon]
                                 for h in range(0, len(xt), horizon)]
                    chunks_yt = [yt[h:h + horizon]
                                 for h in range(0, len(yt), horizon)]
                    
                    #list_size = [len(ll) for ll in chunks_xt]

                    #print("list of size is : ",list_size)

                    #hyp_list = []
                    for i in tqdm(
                        range(
                            len(chunks_xt)), cprint(
                            message, "green")):
                        #self.fit(x_train, y_train)
                        ###self._kernel._variance = stdy*self._kernel._variance
                        #hyp_list.append(self.hyperparameters)
                       # print(self._xtrain.shape,self._ytrain.shape)

                        (res_h, std_pred_h) = self.mean_predict(chunks_xt[i])


                        

                        if (i < len(chunks_xt) -2):
                            x_train = np.hstack((x_train, chunks_xt[i]))

                            y_train = np.hstack((y_train, chunks_yt[i]))
                            self._ytrain = y_train
                            self._xtrain = x_train


                        #res.extend(stdy * res_h + meany)
                        res.extend(res_h)
                        std_pred.extend(std_pred_h)

            y_pred, std_pred = np.array(res), np.array(std_pred)



        return (y_pred.ravel(), std_pred.ravel())


if __name__ == "__main__":
    from gpasdnn.kernels import *
    from gpasdnn.gp import GaussianProcessRegressor as GP
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    x = np.linspace(0,3,1000)

  
    kernel =  RBF(length_scale = .5)
    kernel =  RBF(length_scale = .5, variance = 1.5)
    

    #
    y0 = kernel.simulate(x,seed = np.random.seed(0))

    y = y0 + .2*np.random.randn(x.size)



    plt.figure(figsize=(10,5))
    plt.plot(x,y,'ok',label= "Data")
    plt.plot(x,y0,'r',label = "Simulated gp with '{}' kernel function".format(kernel.label()))
    plt.legend()
    plt.show()

    