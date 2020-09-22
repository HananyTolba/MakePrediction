#!/usr/bin/env python
# -*- coding: utf-8 -*-

# __author__ = "Hanany Tolba"
# __copyright__ = "Copyright 2020, Guassian Process as Deep Learning Model Project"
# __credits__ = "Hanany Tolba"
# __license__ = "GPLv3"
# __version__ ="0.0.3"
# __maintainer__ = "Hanany Tolba"
# __email__ = "hananytolba@yahoo.com"
# __status__ = "4 - Beta"





'''This module is for Gaussian Process Regression simulation fitting and prediction.'''

import matplotlib.pyplot as plt
import importlib
import copy
import joblib 


from makeprediction.invtools import fast_pd_inverse as pdinv
from makeprediction.invtools import inv_col_add_update, inv_col_pop_update
import makeprediction.kernels as kernels
from makeprediction.kernels import *
#####from makeprediction.kernels import date2num

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
#from tensorflow.keras.models import load_model

# tf mp
from scipy.interpolate import interp1d
#from tensorflow import keras
import scipy.signal 



#from tensorflow.keras import backend as K
from scipy import signal
from tqdm import tqdm
from scipy.signal import correlate
from numpy import argmax, mean, diff, log, nonzero
from termcolor import *
import colorama
colorama.init()

def date2num(dt):
    if np.issubdtype(dt.dtype, np.datetime64):
        x = dt.astype(int).values/10**9/3600/24
    elif isinstance(dt, np.ndarray):
        if dt.ndim == 1:
            x = dt
        elif 1 in dt.shape:
            x = dt.ravel()
        else:
            raise ValueError('The {} must be a one dimension numpy array'.format(dt))
    else:
        raise TypeError('The {} must be a numpy vector or pandas DatetimeIndex'.format(dt))
    return x

# cprint('hello'.upper(), 'green')
  
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 
    
# def get_parms_from_tfModel(model,y):
#     infer = model.signatures["serving_default"]
#     labeling = infer(tf.constant(y.reshape(1,-1).astype('float32')))#[model.output_names[0]]
#     res = list(labeling.keys())[0] 
#     return labeling[res].numpy().ravel()

import json
import requests


SMALL_SIZE  = 300
LARGE_SIZE  = 600

Ports = list(range(8501,8509))
kernels = ["rbf_1d","matern12_1d","matern32_1d","matern52_1d",
          "linear_1d",
          "polynomial_1d",
          "periodic_1d",
          "iid_periodic_300",
          ]
PORTS = dict(zip(kernels,Ports))



def get_parms_from_api(y,kernel=None):
    y = y.reshape(1,-1)
    data = {"inputs":y.tolist()}
    
    if kernel is None:
        if y.size == LARGE_SIZE:
            kernel="rbf"
        elif y.size== SMALL_SIZE:
            kernel = "periodic"
    
    
    try:
        port = PORTS[kernel]
    except:
        kernel = kernel.lower() + "_1d"
        port = PORTS[kernel]
        
    url_ec2 = "http://www.makeprediction.com:" + str(port) + "/v1/models/"

    url_ec2 = url_ec2 + kernel + ":predict"
    #print("requests :" ,url_ec2)
    r = requests.post(url_ec2, data=json.dumps(data))
    return np.array(r.json()["outputs"][0])
    

# def period_(y):
#     n = y.size
#     p=1
#     parms = []
#     if (n<SMALL_SIZE ):
#         print("Choisir une autre méthode car length < SMALL_SIZE  points.")
        
        
#     while int(n*p)>=SMALL_SIZE :
#         m = n*p
#         yre12 = scipy.signal.resample(y[:int(m)],SMALL_SIZE )
#         p_est_12 = mdlPeriodic.predict(yre12.reshape(1,-1)).ravel()
#         p_est_12[-1] =p_est_12[-1]*int(m)/y.size
#         if ((p==1)&(p_est_12[-1]>.95)):
#             return p_est_12
#         else:
#             p = p - .005
#             if (p<=0):
#                 break
#             parms.append(p_est_12)
#     if len(parms)>1:
#         parms = np.array(parms)
#         List = np.round(parms[:,1],3).tolist()
#         period_est = most_frequent(List)
#         print("Estimated period is: ",period_est)
        
#         Est = parms[0,:]
#         Est[-1] = period_est
        
#         return Est
        
        




# __all__ = ["GaussianProcessRegressor","RBF","Matern52",
# "Matern32", "Matern12", "Periodic", "Polynomial","Periodic"]

# #import site


#path = site.getsitepackages()[0]

#path_list = list(filter(lambda x: x.endswith('site-packages') ,sys.path))

# import sysconfig, os, glob
# #path_list_1 = site.getsitepackages()
# #print("path_list_1",path_list_1)
# path = sysconfig.get_paths()["purelib"]


# path = os.path.join(path,'makeprediction/SavedModels')




# file_name = [os.path.join(path,f) for f in os.listdir(path) if not f.startswith('.')]


# path_periodic = os.path.join(path, "periodic_1d")
# path_periodic_noise = os.path.join(path, "iid_periodic_300")


#print(path_periodic in file_name)

#print(path_periodic_noise in file_name)

#file_name.remove('.DS_Store')
class_names = ['Linear', 'Linear + Periodic', 'Periodic', 'Polynomial',
       'Polynomial + Periodic', 'Polynomial + Periodic + Stationary',
       'Polynomial + Stationary', 'Stationary', 'Stationary + Linear + Periodic',
       'Stationary + Periodic']


#path_predict_model = path + '/predict_gpr_model'
#model_expression = keras.models.load_model("/Users/tolba/Desktop/makeprediction/keras_LARGE_SIZE /predict_gpr_model")

#model_expression = load_model(path_predict_model)
#probability_model = tf.keras.Sequential([model_expression, tf.keras.layers.Softmax()])

#path_periodic = os.path.join(path, "periodic_1d")
#print(path_periodic)
#K.clear_session()
#newModel = load_model(path_periodic)
#newModel = tf.saved_model.load(path_periodic)
#infer = loaded_model.signatures["serving_default"]


#path_periodic_noise = os.path.join(path, "iid_periodic_300")
#print(path_periodic)
#K.clear_session()
#model_periodic_noise = load_model(path_periodic_noise)
#model_periodic_noise = tf.saved_model.load(path_periodic_noise)


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
import makeprediction.kernels as kernels_module
Kernels = inspect.getmembers(kernels_module, inspect.isclass)
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
            >>> from makeprediction.gp import GaussianProcessRegressor as GPR
            >>> from makeprediction.kernels import RBF, Periodic
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




    def __init__(self,xtrain=None,ytrain=None, kernel=RBF(), model=None, sigma_n=.01):
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
        #path = file_list[file_name.index('rbf_1d')]
        #path = list(filter(lambda x: 'rbf_1d' in x, file_name))[0]
        #K.clear_session()
        #best_model = tf.saved_model.load(path)


        #best_model = load_model(path)
        #self._model = best_model
        self._sigma_n = sigma_n
        # self._pred = pred


    @classmethod
    def from_dataframe(cls, args):
        if isinstance(args, pd.DataFrame): 
            if args.shape[1]>=2:
                x1,y1 = args.iloc[:, 0].values, args.iloc[:, 1].values
                return cls(x1,y1)

            else:
                x1, y1 = args.index, args.iloc[:, 0].values
                return cls(x1,y1)
        
        else:
            raise ValueError("Invalid args, list, tuple, dict or dataframe.")





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
            str_kernel = kernel + '_1d'
            #path_model = list(filter(lambda x: str_kernel in x, file_name))[0]
            #path_model = file_list[file_name.index(str_kernel)]
            #K.clear_session()  # pour accelerer keras model load

            #best_model = load_model(path_model)
            #best_model = tf.saved_model.load(path_model)

            #try:
                #best_model = load_model(path_model)
            #except:
            #    best_model = keras.models.load_model(path_model)

            self._kernel = Kernels_class_instances[location]
            self._kernel.set_length_scale(1)
            if self._kernel.__class__.__name__ == "Periodic": 
                self._kernel.set_period(1)


            #self._model = best_model


    # @classmethod
    # def model_change(cls,model_path,kernel_name="RBF"):
    #     str_kernel = kernel_name.lower() + '_1d'

    #     cls.file_list[cls.file_name.index(str_kernel)] = model_path
    

    def choice(self, ker):
        self.kernel_choice = ker


    def save_model(self,filename):
        if "_model" in self.__dict__.keys():
            self.__dict__.pop("_model")
        joblib.dump(self, filename + '.joblib')
        
    def load_model(self,path):
        return joblib.load(path) 


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




        #x = date2num(x)


        #x,y = self._xtrain, self._ytrain

        ystd = y.std()
        y = (y - y.mean()) / y.std()

        
        n = y.size
#=================================================================
        x_interp = np.linspace(-1, 1, SMALL_SIZE )

        x_transform, a, b = self.line_transform(x.reshape(-1, 1))

        y_interp = np.interp(x_interp, x_transform, y)
        #print("shape_periodic : ",y_interp.shape)
        period_est_ = get_parms_from_api(y_interp,self._kernel.label())


        #period_est_ = newModel.predict(y_interp.reshape(1,x_interp.size)) 
        #period_est_ = period_est_.ravel()

        #===============noise std===========
        noise_std = get_parms_from_api(y_interp,"iid_periodic_300")
        #noise_std = model_periodic_noise.predict(y_interp.reshape(1,x_interp.size))
        #noise_std = noise_std.ravel()
        self._sigma_n = noise_std[0]*ystd
        #print("period_est_ (methode1) :",period_est_)
#=================================================================
        # x_interp = np.linspace(-1,1,SMALL_SIZE )
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
    

    def periodicFitByRandomSampling(self):
        y = self._ytrain
        ystd = y.std()
        y = (y - y.mean()) / y.std()



        n = y.size
        parms = []
        if (n<310):
            print("The 'Split' method was automatically chosen because the data size is very small for the 'Resampling' method.")

            return self.periodicFitBySplit()
        
        parmsList = []
        for i in range(100):
            I = np.random.choice(y.size, SMALL_SIZE , replace=False)
            I = np.sort(I)
            yI = y[I]

            #yre12 = scipy.signal.resample(y[:int(m)],SMALL_SIZE )



            #p_est_12 = newModel.predict(yI.reshape(1,-1)).ravel()
            p_est_12 = get_parms_from_api(yI,"periodic")

            p_est_12[-1] =p_est_12[-1]*SMALL_SIZE /y.size

            parmsList.append(p_est_12)

        npArray = np.array(parmsList)
        #print(npArray.shape)
        #plt.plot(npArray)
        #plt.show()

        hyp = npArray.mean(axis=0)

        #print(hyp)

        hyp_dict = dict(zip(["length_scale","period"],hyp))
        hyp_dict["variance"] = ystd**2
        self.set_hyperparameters(hyp_dict)


        

    def periodicFitByResampling(self):
        y = self._ytrain
        ystd = y.std()
        y = (y - y.mean()) / y.std()



        n = y.size
        p=1
        parms = []
        if (n<350):
            print("The 'Split' method was automatically chosen because the data size is very small for the 'Resampling' method.")

            return self.periodicFitBySplit()
            
            
        while int(n*p)>=SMALL_SIZE:
            m = n*p
            yre12 = scipy.signal.resample(y[:int(m)],SMALL_SIZE )
            p_est_12 = get_parms_from_api(yre12,"periodic")

            #p_est_12 = newModel.predict(yre12.reshape(1,-1)).ravel()
            p_est_12[-1] =p_est_12[-1]*int(m)/y.size
            #noise_std = model_periodic_noise.predict(yre12.reshape(1,yre12.size))
            noise_std = get_parms_from_api(yre12,"iid_periodic_300")

            noise_std = noise_std.ravel()
            self._sigma_n = noise_std[0]*ystd


            if ((p==1)&(p_est_12[-1]>.95)):
                hyp = p_est_12
                hyp_dict = dict(zip(["length_scale","period"],hyp))
                hyp_dict["variance"] = ystd**2
                self.set_hyperparameters(hyp_dict)
                break


                
            else:
                p = p - .01
                if (p<=0):
                    break
                parms.append(p_est_12)
        if len(parms)>1:
            parms = np.array(parms)
            #plt.plot(parms[:,1])
            #plt.show()
            L = parms[:,1]
            periodEst = L[np.argmin(np.abs(np.diff(L)))]


            
            List = np.round(parms[:,1],3).tolist()
            period_est___ = most_frequent(List)
            #print("Estimated period is: ",period_est___)
            
            Est = parms[0,:]
            #Est[-1] = periodEst
            Est[-1] = period_est___
            hyp = Est
            hyp_dict = dict(zip(["length_scale","period"],hyp))
            hyp_dict["variance"] = ystd**2
            self.set_hyperparameters(hyp_dict)

            
            
            


    


    def periodicFitByRegular(self):
        x,y = self._xtrain, self._ytrain



        x = date2num(x)


        hyp = self.p_fit(x,y)
        hyp_dict = dict(zip(["length_scale","period"],hyp))
        hyp_dict["variance"] = y.std()**2
        self.set_hyperparameters(hyp_dict)


    def periodicFitByInterSplit(self):
        x,y = self._xtrain, self._ytrain
        
        

        x = date2num(x)
        x_plus = np.linspace(x[0],  x[-1],x.size*5) 
        y_plus = np.interp(x_plus, x, y)

        #plt.plot(x_plus,y_plus)
        #plt.show()


        #hyp = self.p_fit(x_plus,y_plus)

        self._xtrain, self._ytrain = x_plus, y_plus

        
        self.periodicFitBySplit()

        self._xtrain, self._ytrain = x, y

    def periodicFitByInterResampl(self):
        x,y = self._xtrain, self._ytrain
        
        

        x = date2num(x)
        x_plus = np.linspace(x[0],  x[-1],x.size*5) 
        y_plus = np.interp(x_plus, x, y)

        #plt.plot(x_plus,y_plus)
        #plt.show()


        #hyp = self.p_fit(x_plus,y_plus)

        self._xtrain, self._ytrain = x_plus, y_plus

        
        self.periodicFitByResampling()

        self._xtrain, self._ytrain = x, y
    
    def periodicFitByInterRegular(self):
        x,y = self._xtrain, self._ytrain
        
        

        x = date2num(x)
        x_plus = np.linspace(x[0],  x[-1],x.size*5) 
        y_plus = np.interp(x_plus, x, y)

        #plt.plot(x_plus,y_plus)
        #plt.show()


        #hyp = self.p_fit(x_plus,y_plus)

        self._xtrain, self._ytrain = x_plus, y_plus

        
        self.periodicFitByRegular()

        self._xtrain, self._ytrain = x, y
        

        #hyp_dict = dict(zip(["length_scale","period"],hyp))
        #hyp_dict["variance"] = y.std()**2
        #self.set_hyperparameters(hyp_dict)


    





    def periodicFitBySplit(self):
        pp = []
        x,y = self._xtrain, self._ytrain

        x = date2num(x)

        hyp = self.p_fit(x,y)

        L = list()
        m = 100
        for i in range(m):
            r = int((i+1)/m*x.size)
            if r>=100:
                L.append(self.p_fit(np.linspace(-1,1,r),y[:r])[-1]*r/x.size)
        periodEst = L[np.argmin(np.abs(np.diff(L)))]
        



        hyp[-1] = round(periodEst,3)

        

        hyp_dict = dict(zip(["length_scale","period"],hyp))
        hyp_dict["variance"] = y.std()**2
        self.set_hyperparameters(hyp_dict)

    def fitInter(self,method=None):
        x,y = self._xtrain, self._ytrain
        
        

        x = date2num(x)
        x_plus = np.linspace(x[0],  x[-1],x.size*5) 
        y_plus = np.interp(x_plus, x, y)
        self._xtrain, self._ytrain = x_plus, y_plus


        self.fit(method)
        self._xtrain, self._ytrain = x, y






    

    def fit(self,method = None):
        '''
        This method allows the estimation of the hyperparameters of the GPR model.
        '''
        xtrain, ytrain = self._xtrain, self._ytrain

        xtrain = date2num(xtrain)

        xtrain,ytrain = self._sorted(xtrain,ytrain)

        # cprint("Fit a Gaussian Process to data ...".upper(), 'green')
        xtrain = self.x_type(xtrain)
        ytrain = self.x_type(ytrain)


        meany, stdy = ytrain.mean(), ytrain.std()
        ytrain = (ytrain - meany) / stdy


        if self._kernel.__class__.__name__ == "Periodic":
            if method is None:
                self.periodicFitBySplit()
            elif method == "resample":
                self.periodicFitByResampling()
            elif method == "split":
                self.periodicFitBySplit()
            elif method == "intersplit":
                self.periodicFitByInterSplit()
            elif method == "interresample":
                self.periodicFitByInterResampl()
            elif method == "interregular":
                self.periodicFitByInterRegular()

            elif method == "random":

                self.periodicFitByRandomSampling()

            elif method == "regular":
                self.periodicFitByRegular()

            #elif method == "search":
            #    self.periodicFitBySearch()

            else:
                raise ValueError("Error: '{}' unknown method name.".format(method))



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
                x_interp = np.linspace(-3, 3, LARGE_SIZE )
                #x_interp = np.linspace(x1[0], x1[-1], LARGE_SIZE )
            elif self._kernel.__class__.__name__ == "Polynomial":
                x_interp = np.linspace(-3, 3, SMALL_SIZE )

            # elif self._kernel.__class__.__name__ in ["Cosine","Exponential"]:
            #     x_interp = np.linspace(-3, 3, SMALL_SIZE )



            else:
                x_interp = np.linspace(-3, 3, LARGE_SIZE )

            
            xtrain_transform, a, b = self.line_transform(xtrain.reshape(-1, 1))

            y_interp = np.interp(x_interp, xtrain_transform, ytrain)
            #y_interp = y_interp.ravel()

            #try:
            #    parmsfit_by_sampling = self._model.predict(y_interp.reshape(y_interp.size, 1))
            #except:
            #parmsfit_by_sampling = self._model.predict(y_interp.reshape(1,y_interp.size))
            

            parmsfit_by_sampling = get_parms_from_api(y_interp,self._kernel.label())


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

        xtrain = date2num(xtrain)
        xtest = date2num(xtest)


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
        
        xtrain = date2num(xtrain)


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

        x_train = date2num(x_train)
        xt = date2num(xt)

        # if yt is None:
        #     #message = "prediction started ..."
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
        # cprint("Gaussian Process prediction in progress ...".upper(), 'green')
        # message = "prediction started ..."

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

        x_train = date2num(x_train)


        if sparse is None:
            sparse = False

        if  sparse:
            try:
                if sparse_size is None:
                    sparse_size = max(LARGE_SIZE ,int(.2*x_train.size))
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
        else:
            xt = date2num(xt)

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
                message = "Long term prediction started ..."
                # cprint(message.title(), "green")

                (res, std_pred) = self.mean_predict(xt)
                res = res.tolist()
                std_pred = std_pred.tolist()
            else:

                yt = self.x_type(yt)


                if horizon is None or horizon == 1:
                    horizon = 1
                   # message = "Satrt a one-step ahead prediction with GPR model (kernel={}).".format(
                   #     self._kernel.label())
                    message = "Start a one-step ahead prediction ..."
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

                    #message = "Satrt a multi-step ahead prediction (horizon={}) with GPR model (kernel={}).".format(
                    #    horizon, self._kernel.label())
                    message = "Start a multi-step ahead prediction (horizon={}) ...".format(
                        horizon)
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
    from makeprediction.kernels import *
    from makeprediction.gp import GaussianProcessRegressor as GP
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

    