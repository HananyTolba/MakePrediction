from gpasdnn.kernels import *
from gpasdnn.gp import GaussianProcessRegressor as GPR
import numpy as np
from copy import copy

class QuasiGPR():
    '''
    This class implements the quasi GaussianProcessRegressor approach
    which allows the modeling and prediction of time series as sums 
    of several GaussianProcesses.
    '''

    def __init__(self,xtrain,ytrain,kernel = RBF(), yfit = None,std_yfit=None, modelList = None, components=None):
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._kernel = kernel
        self._modelList = modelList
        self.components = components
        self._yfit = yfit
        self._std_yfit = std_yfit

    def __repr__(self):
        return "Instance of class '{}'".format(self.__class__.__name__)
    
    def __str__(self):
        message_print = "Quasi Gaussian Process Regressor model with kernel: {}."
        return message_print.format(self._kernel.label())

        
    def fit(self):
        xtrain = self._xtrain
        ytrain = self._ytrain
        kernel_expr = self._kernel
        #models = []

        if kernel_expr.__class__.__name__ == "KernelSum":

            list_models = []
            comp = []
            sig_list = []
            kernel_names = kernel_expr.recursive_str_list()
            for ker in kernel_names:
                model = GPR(xtrain,ytrain)
                model.kernel_choice = ker
                model.fit()
                copy_kernel_model = copy(model._kernel)
                model._kernel = copy_kernel_model
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
            model.fit()
            self._yfit, self._std_yfit = model.predict()

            self._modelList = model
        #return models
    def predict(self, xt=None, yt=None, horizon=None,option=None, sparse = None, sparse_size=None, components=None):


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