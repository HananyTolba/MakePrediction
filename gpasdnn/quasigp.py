#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Author = "Hanany Tolba"
#01/07/2020

# __author__ = "Hanany Tolba"
# __copyright__ = "Copyright 2020, Guassian Process as Deep Learning Model Project"
# __credits__ = "Hanany Tolba"
# __license__ = "GPLv3"
# __version__ ="0.0.3"
# __maintainer__ = "Hanany Tolba"
# __email__ = "hananytolba@yahoo.com"
# __status__ = "4 - Beta"




import pandas as pd 


from gpasdnn.kernels import *
from gpasdnn.gp import GaussianProcessRegressor as GPR
import numpy as np
from copy import copy

from tqdm import tqdm
from termcolor import *
import colorama
import joblib 

colorama.init()

class QuasiGPR():
    '''
    This class implements the quasi GaussianProcessRegressor approach
    which allows the modeling and prediction of time series as sums 
    of several GaussianProcesses.
    '''

    def __init__(self,xtrain=None,ytrain=None,kernel = RBF(), yfit = None,std_yfit=None, modelList = None, components=None):
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._kernel = kernel
        self._modelList = modelList #if self._kernel.__class__.__name__!='KernelSum' else \
                            #[GPR().choice(ker) for ker in self._kernel.recursive_str_list()]
        self.components = components
        self._yfit = yfit
        self._std_yfit = std_yfit


    @classmethod
    def from_dataframe(cls, args):
        if isinstance(args, pd.DataFrame): 
            if args.shape[1]>=2:
                x1,y1 = args.iloc[:, 0], args.iloc[:, 1].values
                return cls(x1,y1)

            else:
                x1, y1 = args.index, args.iloc[:, 0].values
                return cls(x1,y1)



    def __repr__(self):
        return "Instance of class '{}'".format(self.__class__.__name__)
    
    def __str__(self):
        message_print = "Quasi Gaussian Process Regressor model with kernel: {}."
        return message_print.format(self._kernel.label())

    def get_hyperparameters(self):
        hyp = []
        if isinstance(self._modelList, list):
            for mdl in self._modelList:
                if mdl is not None:
                    hyp.append((mdl._kernel.label(),mdl.get_hyperparameters()))
                else:
                    hyp.append((None,None))

            #return list(map(lambda s: (s._kernel.label(),s.get_hyperparameters()),self._modelList))
            return hyp
        elif self._modelList is None:
        #else:
            return list(zip(self._kernel.label().split(' + '),self._kernel.get_hyperparameters())) \
                    if self._kernel.__class__.__name__ == "KernelSum" else [self._kernel.label(), self._kernel.get_hyperparameters()]
        else:
           return self._modelList.get_hyperparameters()

    def set_hyperparameters(self,hyp):
        model_new =  self._modelList
        k=0
        for h in hyp:
            if self._modelList[k]._kernel.label() == h[0]:
                model_new[k].set_hyperparameters(h[1])
                k+=1
            else:
                raise ValueError("Error in kernel name choice '{} != {}'.".format(self._modelList[k]._kernel.label(),h[0]))
    
    def save_model(self,filename):
        #if "_model" in self.__dict__.keys():
        #    self.__dict__.pop("_model")
        joblib.dump(self, filename + '.joblib')
        
    def load_model(self,path):
        return joblib.load(path) 



    def fit(self,method=None):
        xtrain = self._xtrain
        ytrain = self._ytrain
        kernel_expr = self._kernel
        #models = []
        if isinstance(method, list):
            l = kernel_expr.recursive_str_list()
            methods_list =[]
            k=0
            for mtd in l:
                if mtd == "Periodic":
                    methods_list.append(method[k])
                    k += 1
                else:
                    methods_list.append(None)
        else:
            methods_list = method

        



        if kernel_expr.__class__.__name__ == "KernelSum":

            list_models = []
            comp = []
            sig_list = []
            kernel_names = kernel_expr.recursive_str_list()
            for ii in range(len(kernel_names)):
                ker = kernel_names[ii]
                model = GPR(xtrain,ytrain)
                model.kernel_choice = ker
                if isinstance(methods_list, list):
                    model.fit(methods_list[ii])
                else:
                    model.fit(methods_list)


                copy_kernel_model = copy(model._kernel)
                model._kernel = copy_kernel_model
                if "_model" in model.__dict__.keys():
                    model.__dict__.pop("_model")
                list_models.append(model)
                yf,sig = model.predict()
                comp.append(yf)
                sig_list.append(sig)
                #plt.plot(ytrain,'k')
                #plt.plot(yf,'r')
                ytrain = ytrain - yf

                
                #plt.plot(ytrain)
                #model._ytrain = ytrain
                    #yfit = yfit + yf
            self._modelList = list_models
            self.components = comp
            self._yfit = sum(comp)
            self._std_yfit = sum(sig_list)
        elif kernel_expr.__class__.__name__  =="KernelProduct":
           # self._ytrain = self._ytrain 
            raise ValueError("The kernel must be a sum of kernels or a simple kernel.")

        else:
            model = GPR(xtrain,ytrain)
            model.kernel_choice = kernel_expr.label()
            model.fit(method)
            if "_model" in model.__dict__.keys():
                model.__dict__.pop("_model")
            #model.__dict__.pop("_model")
            self._yfit, self._std_yfit = model.predict()

            self._modelList = model

        #return models
    def predict(self, xt=None, yt=None, horizon=1,option=True, sparse = None, sparse_size=None, components=None):


        if xt is None:
            ypred_,std_ = self._yfit, self._std_yfit
            cmps = self.components
        else:

            if self._modelList.__class__.__name__ == "GaussianProcessRegressor": 
                ypred_, std_ = self._modelList.predict(xt,yt,horizon,option)
                components = False
            else:
                models = self._modelList
                yt_std_list = []
                yt_pred_list = []
                for mdl in models:
                    yt_pred, yt_std = mdl.predict(xt,yt,horizon, option, sparse, sparse_size)
                  
                    if yt is not None:
                        yt = yt - yt_pred[:-horizon]

                    yt_pred_list.append(yt_pred)
                    yt_std_list.append(yt_std)
    # Il faut absolument changer ce code car dans le cas où xt est None
    # La prédiction est dèja calculer lors de fit.                
                #if xt is None:
                   # ypred_, std_  = yt_pred, yt_std
                #else:
                ypred_ = sum(yt_pred_list)
                std_ = sum(yt_std_list)
                cmps = np.array(yt_pred_list)

        if components:
            return ypred_,std_, cmps
        else:
            return ypred_,std_