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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

colorama.init()

class QuasiGPR():
    '''
    This class implements the quasi GaussianProcessRegressor approach
    which allows the modeling and prediction of time series as sums 
    of several GaussianProcesses.
    '''

    def __init__(self,xtrain=None,ytrain=None,kernel = RBF(), yfit = None,std_yfit=None, modelList = None, components=None
        ,xtest=None,ypred=None,std_ypred=None):
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._kernel = kernel
        self._modelList = modelList #if self._kernel.__class__.__name__!='KernelSum' else \
                            #[GPR().choice(ker) for ker in self._kernel.recursive_str_list()]
        self.components = components
        self._yfit = yfit
        self._std_yfit = std_yfit
        self._xtest = xtest
        #self._ytest = ytest
        self._ypred = ypred
        self._std_ypred = std_ypred





    def plot(self,prediction=None,ci=None):
        if prediction is None:
            prediction =False
        if ci is None:
            ci = True


        if prediction:
            if ci:
                plt.figure(figsize=(12,5))
                plt.plot(self._xtrain,self._ytrain,'k',lw=1,label="Training data")
                #plt.plot(self._xtest,self._ytest,'g',lw=3,label="test data")
                plt.plot(self._xtest,self._ypred,'b',lw=2,label="Prediction")
                plt.plot(self._xtrain,self._yfit,'r',lw=2,label="Model")
                plt.fill_between(self._xtest, (self._ypred - 1.96*self._std_ypred), (self._ypred + 1.96*self._std_ypred),color="b", alpha=0.2,label='Confidence Interval 95%')
                plt.legend()
                plt.show()
            else:
                plt.figure(figsize=(12,5))
                plt.plot(self._xtrain,self._ytrain,'k',lw=1,label="Training data")
                #plt.plot(self._xtest,self._ytest,'g',lw=3,label="test data")
                plt.plot(self._xtest,self._ypred,'b',lw=2,label="Prediction")
                plt.plot(self._xtrain,self._yfit,'r',lw=2,label="Model")
                #plt.fill_between(self._xtest, (self._ypred - 1.96*stdpred), (self._ypred + 1.96*stdpred),color="b", alpha=0.2,label='Confidence Interval 95%')
                plt.legend()
                plt.show()

        else:
            plt.figure(figsize=(12,5))
            plt.plot(self._xtrain,self._ytrain,'k',lw=1,label="Training data")
            #plt.plot(xtest,(ytest),'g',lw=3,label="test data")
            #plt.plot(xs,(yp),'b',lw=2,label="Prediction")
            plt.plot(self._xtrain,self._yfit,'r',lw=2,label="Model")
            plt.fill_between(self._xtrain, (self._yfit - 1.96* self._std_yfit), (self._yfit + 1.96* self._std_yfit),color="b", alpha=0.2,label='Confidence Interval 95%')
            plt.legend()
            plt.show()


    def components_plot(self):
        kernel_list = self._kernel.label().split(' + ')
        m = len(kernel_list)
        if m>1:
            fig,ax = plt.subplots(m,1,figsize=(10,10),sharex=True)
            for i in range(m):
                ax[i].plot(self.components[i],'b')
                ax[i].set_title("The {}-th component ({})".format(i,kernel_list[i]))
            plt.show()

    def score(self,ytest = None):
        if ytest is None:
            L_score = [mean_absolute_error(self._ytrain,self._yfit) ,
            mean_squared_error(self._ytrain,self._yfit) ,
             r2_score(self._ytrain,self._yfit)]
            d = dict(zip(["MAE","MSE","R2"],L_score))

            
        else:
            L_score = [mean_absolute_error(ytest,self._ypred) ,
            mean_squared_error(ytest,self._ypred) ,
             r2_score(ytest,self._ypred)]
            d = dict(zip(["MAE","MSE","R2"],L_score))
            

        return d


        


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

        self._xtest = xt

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


        self._ypred = ypred_
        self._std_ypred = std_

        if components:

            return ypred_,std_, cmps
        else:
            return ypred_,std_