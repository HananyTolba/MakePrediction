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

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#import plotly.graph_objs as go

import datetime
import time 
import logging
import requests
import json
from copy import copy, deepcopy 
import csv
import multiprocessing

logging.basicConfig(level=logging.DEBUG)

import pandas as pd 
import itertools

from makeprediction.exceptions import *
from makeprediction.kernels import *
from makeprediction.gp import GaussianProcessRegressor as GPR

from makeprediction.gp import get_parms_from_api
from numpy.linalg import cholesky, det, lstsq
from makeprediction.url import kernel2url


import numpy as np
import scipy
from scipy.signal import resample
from scipy import interpolate
from copy import copy

from tqdm import tqdm
from termcolor import *
import colorama
import joblib 
import matplotlib.pyplot as plt


import dash, signal
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc    
from dash.dependencies import Input, Output

from dash_table import DataTable



style_text  = {
               "backgroundColor":"#e5ecf6",
               "textAlign": "center",
               'padding': '10px 0px 10px 0px',
               'fontfamily': 'sans-serif',
              }


style_logo_left = {
             
               'padding': '5px 0px 0px 75px',
               
               'marging': '0px 0px 0px 0px'
              }

style_logo_right = {
               
               'padding': '5px 0px 0px 0px',
               
               'marging': '0px 0px 0px 0px',
              }


styleCell = {'font-family':'sans-serif','textAlign': 'center',}








colorama.init()


import makeprediction.kernels as kernels_module
import inspect
Kernels = inspect.getmembers(kernels_module, inspect.isclass)
Kernels_class_instances = [m[1]() for m in Kernels]
Kernels_class_names = [m[0].lower() for m in Kernels]
Kernels_class_names


SMALL_SIZE = 300


class_names = ['Linear', 'Linear + Periodic', 'Periodic', 'Polynomial',
       'Polynomial + Periodic', 'Polynomial + Periodic + Stationary',
       'Polynomial + Stationary', 'Stationary', 'Stationary + Linear + Periodic',
       'Stationary + Periodic']


class_names_new = ['Linear',
 'Periodic',
 'RBF',
 'Periodic + Linear',
 'Periodic + Linear + RBF',
 'Periodic + RBF',
 ]
simple_class = ['Linear', 'Matern12', 'Periodic', 'RBF', 'WN']


def mean_squared_error(Y_true,Y_pred):
    Y_true = Y_true.ravel()
    Y_pred = Y_pred.ravel()

    return np.square(np.subtract(Y_true,Y_pred)).mean() 

def mean_absolute_error(Y_true,Y_pred):
    Y_true = Y_true.ravel()
    Y_pred = Y_pred.ravel()
    return np.abs(np.subtract(Y_true,Y_pred)).mean() 

def r2_score(Y_true,Y_pred):
    Y_true = Y_true.ravel()
    Y_pred = Y_pred.ravel()
    return np.corrcoef(Y_true, Y_pred)[0, 1]**2 



class QuasiGPR():
    '''
    This class implements the quasi GaussianProcessRegressor approach
    which allows the modeling and prediction of time series as sums 
    of several GaussianProcesses.
    '''

    def __init__(self,xtrain=None,ytrain=None,kernel = None, yfit = None,std_yfit=None, modelList = None, components=None
        ,xtest=None,ypred=None,std_ypred=None, transform = None, deploy_file = None, score = None):
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
        self.transform = transform
        #self._deploy = deploy
        #self.deploy_file = deploy_file

        self._score = score


        if transform is not None:
            self._ytrain = transform(ytrain)



    def get_kernel(self):
        return self._kernel

    def set_kernel(self,kernel_exp):
        if isinstance(kernel_exp,str):
            kernel_exp_list = kernel_exp.split('+')
            kernel_exp_list = [ker.replace(" ","").lower() for ker in kernel_exp_list]
            if all([ker in Kernels_class_names for ker in kernel_exp_list]):
                kernel_sum_list = []
                for ker in kernel_exp_list:
                    location = Kernels_class_names.index(ker)
                    kernel_sum_list.append(Kernels_class_instances[location])
            else:
                raise ValueError("The kernel must be a sum of kernels or a simple kernel.")

            kernel_sum = sum(kernel_sum_list)
            self._kernel = kernel_sum 
        elif kernel_exp.__class__.__name__.lower() in Kernels_class_names:
            kernel_sum = kernel_exp
            self._kernel = kernel_sum

        else:
            raise ValueError("The kernel {} is not valid.".format(kernel_exp))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x)
        return e_x / e_x.sum()

    
    def kernel_predict(self,result = None):
    
        #ytrain = self._ytrain
        #ytrain = (ytrain - ytrain.mean())/ytrain.std()
        #y_resample= scipy.signal.resample(ytrain,SMALL_SIZE)
        #y_resample = (y_resample - y_resample.mean())/y_resample.std()
        predictions = self.kernel_predict_from_apiREST()
        prob_predictions = self.softmax(predictions)

        #prob_predictions = probability_model.predict(y_resample.reshape(1,-1))
        prob_predictions = prob_predictions.ravel()
        pred_test = np.argmax(prob_predictions)
        class_ = class_names[pred_test]
        res = dict(zip(class_names, prob_predictions.tolist())) 
        df = pd.DataFrame.from_dict(res,orient='index',columns=["Probability"])
        df = df.round(4)
        df.sort_values(by=['Probability'], inplace=True,ascending=False)
        
        if result is None:
            return class_
        elif result == "dict":
            return class_, res
        elif result == "df":
            return class_, df

        
    def kernel_predict1(self,result = None):
    
        ytrain = self._ytrain
        ytrain = (ytrain - ytrain.mean())/ytrain.std()
        y_resample= scipy.signal.resample(ytrain,SMALL_SIZE)
        #y_resample = (y_resample - y_resample.mean())/y_resample.std()
        prob_predictions = get_parms_from_api(y_resample,"gp_kernel_predict_300")
        #prob_predictions = probability_model.predict(y_resample.reshape(1,-1))
        prob_predictions = prob_predictions.ravel()
        #print(prob_predictions)
        pred_test = np.argmax(prob_predictions)
        class_ = class_names_new[pred_test]
        res = dict(zip(class_names_new, prob_predictions.tolist())) 
        df = pd.DataFrame.from_dict(res,orient='index',columns=["Probability"])
        df = df.round(4)
        df.sort_values(by=['Probability'], inplace=True,ascending=False)
        
        if result is None:
            return class_
        elif result == "dict":
            return class_, res
        elif result == "df":
            return class_, df


    def simple_kernel_predict(self,result = None):
    
        ytrain = self._ytrain
        ytrain = (ytrain - ytrain.mean())/ytrain.std()
        y_resample= scipy.signal.resample(ytrain,SMALL_SIZE)
        #y_resample = (y_resample - y_resample.mean())/y_resample.std()
        prob_predictions = get_parms_from_api(y_resample,"gp_kernel_predict_simple_300")
        #prob_predictions = probability_model.predict(y_resample.reshape(1,-1))
        prob_predictions = prob_predictions.ravel()
        #print(prob_predictions)
        pred_test = np.argmax(prob_predictions)
        class_ = simple_class[pred_test]
        res = dict(zip(simple_class, prob_predictions.tolist())) 
        df = pd.DataFrame.from_dict(res,orient='index',columns=["Probability"])
        df = df.round(4)
        df.sort_values(by=['Probability'], inplace=True,ascending=False)
        
        if result is None:
            return class_
        elif result == "dict":
            return class_, res
        elif result == "df":
            return class_, df
    



    def kernel_predict_from_apiREST(self,model = None):
        y = self._ytrain
        y = scipy.signal.resample(y,SMALL_SIZE)
        y = (y-y.mean())/y.std()

        y = y.reshape(1,-1)
        data = {"inputs":y.tolist()}
        if model is None:
            model = "model_expression_predict"
        url_ec2 = kernel2url(model)

        r = requests.post(url_ec2, data=json.dumps(data),timeout=5)
        res = np.array(r.json()["outputs"][0])
        
        return res
    

    def kernel_predict_from_apiREST_split(self):
        y = self._ytrain
        sizes = np.linspace(.1,1,30)
        url_ec2 = kernel2url("model_expression_predict")
        
        result = []
        for s in sizes:
            ys = scipy.signal.resample(y[:int(s*y.size)],SMALL_SIZE)
            ys = (ys-ys.mean())/ys.std()
            ys = ys.reshape(1,-1)
            data = {"inputs":ys.tolist()}
            r = requests.post(url_ec2, data=json.dumps(data),timeout=5)
            res = np.array(r.json()["outputs"][0])
            result.append(res)
        
        return result


    




    def log_lh_stable(self, theta, noise):

        x,y = self._xtrain, self._ytrain

        
        #noise = .1
        model_resample = GPR(np.linspace(-1,1,SMALL_SIZE),scipy.signal.resample(y,SMALL_SIZE),Periodic())
        X_train, Y_train = model_resample._xtrain, model_resample._ytrain

        
        def ls(a, b):
            return lstsq(a, b, rcond=-1)[0]
        
        kernel = model_resample._kernel
        d = {'variance': 1, 'length_scale': 1, 'period': theta}
        kernel.set_hyperparameters(d)
        
        K = kernel.count(model_resample._xtrain) + noise**2 * np.eye(model_resample._xtrain.size)
        

        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.dot(ls(L.T, ls(L, Y_train))) + \
               0.5 * len(X_train) * np.log(2*np.pi)







    def plot(self,prediction=None,ci=None,xtest=None,ytest=None):
        if prediction is None:
            prediction =False
        if ci is None:
            ci = False

        test_None = all(elem is not None for elem in [xtest,ytest])

        if prediction:
            if ci:
                plt.figure(figsize=(12,5))
                if test_None:
                    plt.plot(xtest,ytest,'g',lw=2,label="Test data")
   
                plt.plot(self._xtrain,self._ytrain,'-.k',lw=2,label="Train data")
                plt.plot(self._xtest,self._ypred,'b',lw=2,label="Prediction")
                plt.plot(self._xtrain,self._yfit,'r',lw=2,label="Model")
                plt.fill_between(self._xtest, (self._ypred - 1.96*self._std_ypred), (self._ypred + 1.96*self._std_ypred),color="b", alpha=0.2,label='Confidence Interval 95%')
                plt.fill_between(self._xtrain, (self._yfit - 1.96* self._std_yfit), (self._yfit + 1.96* self._std_yfit),color="b", alpha=0.2)
                plt.grid()

                plt.legend()
                plt.show()
            else:
                plt.figure(figsize=(12,5))
                if test_None:
                    plt.plot(xtest,ytest,'g',lw=2,label="Test data")
   
                plt.plot(self._xtrain,self._ytrain,'.-k',lw=2,label="Training data")
                #plt.plot(self._xtest,self._ytest,'g',lw=3,label="test data")
                plt.plot(self._xtest,self._ypred,'b',lw=2,label="Prediction")
                plt.plot(self._xtrain,self._yfit,'r',lw=2,label="Model")
                #plt.fill_between(self._xtest, (self._ypred - 1.96*stdpred), (self._ypred + 1.96*stdpred),color="b", alpha=0.2,label='Confidence Interval 95%')
                plt.legend()
                plt.grid()

                plt.show()


        else:
            plt.figure(figsize=(12,5))
            plt.plot(self._xtrain,self._ytrain,'.-k',lw=1,label="Training data")
            if test_None:
                plt.plot(xtest,ytest,'ok',lw=1,label="Test data")
   
            #plt.plot(xtest,(ytest),'g',lw=3,label="test data")
            #plt.plot(xs,(yp),'b',lw=2,label="Prediction")
            plt.plot(self._xtrain,self._yfit,'r',lw=2,label="Model")
            if ci:
                plt.fill_between(self._xtrain, (self._yfit - 1.96* self._std_yfit), (self._yfit + 1.96* self._std_yfit),color="b", alpha=0.2,label='Confidence Interval 95%')
            plt.legend()
            plt.grid()
            plt.show()


    def components_plot(self):
        kernel_list = self._kernel.label().split(' + ')
        m = len(kernel_list)
        if m>1:
            fig,ax = plt.subplots(m,1,figsize=(10,10),sharex=True)
            for i in range(m):
                ax[i].plot(self._xtrain,self.components[i],'b')
                ax[i].set_title("The {}-th component ({})".format(i+1,kernel_list[i]))
            plt.show()


    def components_plotly(self,save=False, filename = None,template = "plotly_white"):

        kernels_list = self._kernel.label().split(" + ")
        
        kernels_list = kernels_list + ["Noise"]

        kernels_list = ["{}:  {}-th component.".format(kernels_list[i],i+1) for i in range(len(kernels_list))]

        if len(kernels_list)== 2:
            fig = make_subplots(rows=len(kernels_list), cols=1, subplot_titles = kernels_list)
            fig.append_trace(go.Scatter(
            x=self._xtrain,
            y=self._yfit), row=1, col=1)
            fig.append_trace(go.Scatter(
                    x=self._xtrain,
                    y=self._ytrain - self._yfit),
                    row=2, col=1)
            fig.update_layout(template=template)
            fig.update_layout(showlegend=False)
            fig.update_yaxes(automargin=True)
            fig.show()



        else:
            fig = make_subplots(rows=len(kernels_list), cols=1, subplot_titles = kernels_list)
            for i in range(len(kernels_list)):
                if i< len(self.components):
                    fig.append_trace(go.Scatter(
                    x=self._xtrain,
                    y=self.components[i]), row=i+1, col=1)
                else:
                    fig.append_trace(go.Scatter(
                    x=self._xtrain,
                    y=self._ytrain - self._yfit),
                    row=i+1, col=1)
            #fig.update_layout(legend= {'itemsizing': 'constant'})
            #fig.update_layout(height=700, width=900)
            fig.update_layout(template=template)
            fig.update_layout(showlegend=False)
            fig.update_yaxes(automargin=True)


            fig.show()
        if save:
            #fig.html
            if filename is None:
                filename = "fig"
            

            fig.write_html(filename + ".html")










    def plotly(self,ytest = None, template = "plotly_white",save = False, filename = None, return_fig = False, test_only = False):


        if test_only:
            #print('pass')
            xs_list = self._xtest.tolist()
            xs_rev = xs_list[::-1]
            yp_list = self._ypred.tolist()
            yp_upper = self._ypred + 1.96*self._std_ypred
            yp_upper = yp_upper.tolist()
            yp_lower = self._ypred - 1.96*self._std_ypred
            yp_lower = yp_lower.tolist()
            yp_lower_rev = yp_lower[::-1]
            
                        
            fig = go.Figure()


            # fig.add_trace(go.Scatter(
            # x=x_list, y=y_list,
            # line_color='rgba(0,0,0, 1)',
            # name='Training data',))

            # fig.add_trace(go.Scatter(
            # x=x_list, y=yf_list,
            # line_color='rgba(255,0,0, 1)',
            # name='Model',))
            
           

            fig.add_trace(go.Scatter(
            x=self._xtest, y=yp_list,
            line_color='rgba(0,0,255, .8)',
            name='Prediction',))
            # fig.add_trace(go.Scatter(x=x_list+x_rev,y=y_upper+y_lower_rev,
            # fill='toself',
            # fillcolor='rgba(0,0,255,.1)',
            # line_color='rgba(255,255,255,0)',
            # name='Confidence Interval (95%)',))
            
            fig.add_trace(go.Scatter(
            x =xs_list+xs_rev,
            y=yp_upper+yp_lower_rev,
            fill='toself',
            fillcolor='rgba(0,0,255,.1)',
            line_color='rgba(255,255,255,0)',
            name='Confidence Interval (95%)',
            showlegend=True))
            fig.update_traces(mode='lines')
            fig.update_layout(template=template)
            # fig.update_layout(legend=dict(yanchor="top",
            # y=0.99,
            # xanchor="left",
            # x=0.01))
            fig.update_layout(legend=dict(orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",x=1))
            fig.update_yaxes(automargin=True)

            if return_fig:
                return fig
            else:
                fig.show()

        else:

            if self.transform is not None:
                if ytest is not None:
                    ytest = self.transform(ytest)

            mode_value = 'lines + markers'

            x_list = self._xtrain.tolist()
            x_rev = x_list[::-1]
            y_list = self._ytrain.tolist()
            
            if self._yfit is None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_list, y=y_list,line_color='rgba(0,0,0, 1)',name='Training data',showlegend=True))
                fig.update_traces(mode='lines')
                fig.update_layout(template=template)
                # fig.update_layout(legend=dict(yanchor="top",
                #         y=0.99,
                #         xanchor="left",
                #         x=0.01))
                fig.update_layout(legend=dict(orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",x=1))
                fig.update_yaxes(automargin=True)

                if return_fig:
                    return fig
                else:
                    fig.show()

                #fig.update_layout(template=template, title="www.makeprediction.com")

                #fig.update_layout(autosize=False)
                #fig.show()


            else:
                if  ytest is None:
                    
                    
                    yf_list = self._yfit.tolist()

                    y_upper = self._yfit + 1.96*self._std_yfit

                    y_upper = y_upper.tolist()
                    #y_lower = [s - .2 for s in y_list]
                    y_lower = self._yfit - 1.96*self._std_yfit
                    y_lower = y_lower.tolist()
                    y_lower_rev = y_lower[::-1]




                    fig = go.Figure()
                    
                    if self._ypred is None:
                        fig.add_trace(go.Scatter(
                        x=x_list, y=y_list,
                        line_color='rgba(0,0,0, 1)',
                        name='Training data',))

                        fig.add_trace(go.Scatter(
                        x=x_list, y=yf_list,
                        line_color='rgba(255,0,0, 1)',
                        name='Model',))
                        
                       

                        
                        # fig.add_trace(go.Scatter(x=x_list+x_rev,y=y_upper+y_lower_rev,
                        # fill='toself',
                        # fillcolor='rgba(0,0,255,.1)',
                        # line_color='rgba(255,255,255,0)',
                        # name='Confidence Interval (95%)',))
                        
                        
                        fig.update_traces(mode='lines')
                        fig.update_layout(template=template)
                        # fig.update_layout(legend=dict(yanchor="top",
                        # y=0.99,
                        # xanchor="left",
                        # x=0.01))
                        fig.update_layout(legend=dict(orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",x=1))
                        fig.update_yaxes(automargin=True)

                        if return_fig:
                            return fig
                        else:
                            fig.show()
                        #fig.update_layout(autosize=False)
                        #fig.show()
                    else:
                        xs_list = self._xtest.tolist()
                        xs_rev = xs_list[::-1]
                        yp_list = self._ypred.tolist()
                        yp_upper = self._ypred + 1.96*self._std_ypred
                        yp_upper = yp_upper.tolist()
                        yp_lower = self._ypred - 1.96*self._std_ypred
                        yp_lower = yp_lower.tolist()
                        yp_lower_rev = yp_lower[::-1]
                        
                        


                        fig.add_trace(go.Scatter(
                        x=x_list, y=y_list,
                        line_color='rgba(0,0,0, 1)',
                        name='Training data',))

                        fig.add_trace(go.Scatter(
                        x=x_list, y=yf_list,
                        line_color='rgba(255,0,0, 1)',
                        name='Model',))
                        
                       

                        fig.add_trace(go.Scatter(
                        x=self._xtest, y=yp_list,
                        line_color='rgba(0,0,255, .8)',
                        name='Prediction',))
                        # fig.add_trace(go.Scatter(x=x_list+x_rev,y=y_upper+y_lower_rev,
                        # fill='toself',
                        # fillcolor='rgba(0,0,255,.1)',
                        # line_color='rgba(255,255,255,0)',
                        # name='Confidence Interval (95%)',))
                        
                        fig.add_trace(go.Scatter(
                        x =xs_list+xs_rev,
                        y=yp_upper+yp_lower_rev,
                        fill='toself',
                        fillcolor='rgba(0,0,255,.1)',
                        line_color='rgba(255,255,255,0)',
                        name='Confidence Interval (95%)',
                        showlegend=True))
                        fig.update_traces(mode='lines')
                        fig.update_layout(template=template)
                        # fig.update_layout(legend=dict(yanchor="top",
                        # y=0.99,
                        # xanchor="left",
                        # x=0.01))
                        fig.update_layout(legend=dict(orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",x=1))
                        fig.update_yaxes(automargin=True)

                        if return_fig:
                            return fig
                        else:
                            fig.show()

                        #fig.update_layout(autosize=False)
                        #fig.show()
                else:
                    #prediction = True
                    x_list = self._xtrain.tolist()
                    x_rev = x_list[::-1]
                    y_list = self._ytrain.tolist()
                    yf_list = self._yfit.tolist()

                    y_upper = self._yfit + 1.96*self._std_yfit

                    y_upper = y_upper.tolist()
                    #y_lower = [s - .2 for s in y_list]
                    y_lower = self._yfit - 1.96*self._std_yfit
                    y_lower = y_lower.tolist()
                    y_lower_rev = y_lower[::-1]



                    xs_list = self._xtest.tolist()
                    xs_rev = xs_list[::-1]
                    #ytest_list = ytest.tolist()
                    yp_list = self._ypred.tolist()

                    #y_n = (np.sin(np.array(x)*3) + .1*np.random.randn(1000)).tolist()

                    #y_upper = [s + .2 for s in y_list]
                    yp_upper = self._ypred + 1.96*self._std_ypred

                    yp_upper = yp_upper.tolist()
                    yp_lower = self._ypred - 1.96*self._std_ypred
                    yp_lower = yp_lower.tolist()
                    yp_lower_rev = yp_lower[::-1]

                    ytest_list = ytest.tolist()

                    fig = go.Figure()

                    
                    fig.add_trace(go.Scatter(
                        x=x_list, y=y_list,
                        line_color='rgba(0,0,0, 1)',
                        name='Training data',
                    ))


                    fig.add_trace(go.Scatter(
                        x=x_list, y=yf_list,
                        line_color='rgba(255,0,0, 1)',
                        name='Model',
                    ))



                    fig.add_trace(go.Scatter(
                        x=self._xtest, y=ytest_list,
                        line_color='rgba(128,128,128, 1)',
                        name='Testing data',
                    ))
                    fig.add_trace(go.Scatter(
                        x=self._xtest, y=yp_list,
                        line_color='rgba(0,0,255, .8)',
                        name='Prediction',
                    ))
                    # fig.add_trace(go.Scatter(
                    #     x=x_list+x_rev,
                    #     y=y_upper+y_lower_rev,
                    #     fill='toself',
                    #     fillcolor='rgba(0,0,255,.1)',
                    #     line_color='rgba(255,255,255,0)',
                    #     name='Confidence Interval (95%)',
                    # ))

                    fig.add_trace(go.Scatter(
                        x =xs_list+xs_rev,
                        y=yp_upper+yp_lower_rev,
                        fill='toself',
                        fillcolor='rgba(0,0,255,.1)',
                        line_color='rgba(255,255,255,0)',
                        name='Confidence Interval (95%)',
                        showlegend=True,

                    ))



                    fig.update_traces(mode='lines')            
                    #fig.update_layout(autosize=False)
                    fig.update_layout(template=template)
                    fig.update_layout(legend=dict(orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",x=1))
                    #fig.update_layout(height=700, width=900)
                    fig.update_yaxes(automargin=True)


                    if return_fig:
                        return fig
                    else:
                        fig.show()

            if save:
                #fig.html
                if filename is None:filename = "fig"
                fig.write_html(filename + ".html")

            
                        #fig.show()


            #return  fig






    def score(self,ytest = None):
        if self.transform is not None:
            if ytest is not None:
                ytest = self.transform(ytest)

        if ytest is None:
            L_score = [mean_absolute_error(self._ytrain,self._yfit) ,
            mean_squared_error(self._ytrain,self._yfit) ,
            r2_score(self._ytrain,self._yfit)]
            d = dict(zip(["MAE","MSE","R2"],L_score))
            result= {"train_errors":d}

            
        else:
            L_score_1 = [mean_absolute_error(self._ytrain,self._yfit) ,
            mean_squared_error(self._ytrain,self._yfit) ,
            r2_score(self._ytrain,self._yfit)]
            d1 = dict(zip(["MAE","MSE","R2"],L_score_1))

            L_score_2 = [mean_absolute_error(ytest,self._ypred) ,
            mean_squared_error(ytest,self._ypred) ,
             r2_score(ytest,self._ypred)]
            d2 = dict(zip(["MAE","MSE","R2"],L_score_2))
            result = {"train_errors":d1,"test_errors":d2 }

        self._score = result
        return result


        


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
        
        if self._modelList is None:
            model_new = GPR(self._xtrain,self._ytrain,self._kernel)
        else:
            model_new =  self._modelList
            
        print(model_new)
        if isinstance(hyp, list):
            k=0
            for h in hyp:
                if self._modelList[k]._kernel.label() == h[0]:
                    model_new[k].set_hyperparameters(h[1])
                    k+=1
                else:
                    raise ValueError("Error in kernel name choice '{} != {}'.".format(self._modelList[k]._kernel.label(),h[0]))
        else: 
            model_new.set_hyperparameters(hyp)

        self._modelList = model_new


    ###Path('path/to/file.txt').touch()



    def save(self,filename = None):
        #if "_model" in self.__dict__.keys():
        #    self.__dict__.pop("_model")

        new_directory = str(int(datetime.datetime.now().timestamp()))
        if filename is None:
            filename = "ts_model"

        parent_directory = filename

        # Setting the path for folder creation
        path = os.path.join(parent_directory, new_directory)

        # Handle the errors
        try:
        # Create the directory in the path
            os.makedirs(path, exist_ok = True)
            print("Directory %s Created Successfully" % new_directory)
        except OSError as error:
            print("Directory %s Creation Failed" % new_directory)
        

        joblib.dump(self, os.path.join(path , 'mp.joblib'))
        
    def load(self,path = None):
        if path is None:
            path = "ts_model/"
        path = os.path.join(os.getcwd(),path)
        files = os.listdir(path)
        files = [f for f in  files if not f.startswith(".")]
        files_num = list(map(int,files))
        path = os.path.join(path,str(files[np.argmax(files_num)]))
        path = os.path.join(path,os.listdir(path)[0])
        
        return joblib.load(path) 


    

    def deep_periodic_predict(self,x=None, y=None, split = None):
        if split is None:
            split = [.2,  .5, 1]

        if x is None:
            x = self._xtrain
        if y is None:
            y = self._ytrain

        #x,y = self._xtrain, self._ytrain

        n = y.size
        probs = []
        results = []

        for p in split:
            mm = GPR(x[-int(n*p):],y[-int(n*p):])
            res, prob =  mm.periodic_predict()
            probs.append(prob.tolist())
            results.append(res)
        if 'Periodic' in results:
            probs_filtered = [probs[i] for i in range(len(results)) if results[i] == 'Periodic']
            probs_max = probs_filtered[np.argmax([prob[1] for prob in probs_filtered])]
            return 'Periodic', probs_max
        else:
            probs_max = probs[np.argmax([prob[0] for prob in probs])]
            return 'NonPeriodic', probs_max

            
        
    



    def autofit(self,max_periodic = 2,  stationary_kernel = None):
        x,y = self._xtrain, self._ytrain 

        #if max_periodic is None:
         #   max_periodic = 5
        if  stationary_kernel is None:
             stationary_kernel = 'Matern32'
        
        models = []
        comp = []
        sig_list = []
        decision , Prob  = GPR(x,y).linear_predict()
        if decision == "Linear":
            model = GPR(x,y)
            model.kernel_choice = "Linear"
            model.fit()
            copy_model = deepcopy(model)
            
                
            models.append(copy_model)
            yf,sig = model.predict()
            comp.append(yf)
            sig_list.append(sig)
            y = y - yf
            
            
        periodic_number = 0
        while (periodic_number< max_periodic):
            
            model = GPR(x,y)

            model.choice('Periodic')

            dec1 , Prob1  = self.deep_periodic_predict(x,y)
            #print(dec1 , Prob1)

            period_0 = 10
            
            if dec1 == 'Periodic':
                periodic_number+=1

                model.fit(method = 'inter')
                period = model.get_hyperparameters()["period"]
                if np.abs(period - period_0)<.01:
                    model.fit()


                    
                

                
                copy_model = deepcopy(model)

                models.append(copy_model)
                yf,sig = model.predict()
                comp.append(yf)
                sig_list.append(sig)
                y = y - yf
                period_0 = period
            else:
                break
                
        model = GPR(x,y)
        model.kernel_choice =  stationary_kernel
        model.fit()
        copy_model = deepcopy(model)

        models.append(copy_model)
        yf,sig = model.predict()
        comp.append(yf)
        sig_list.append(sig)
            
        self._modelList = models
        self.components = comp
        self._yfit = sum(comp)
        self._std_yfit = sum(sig_list)
        self._kernel = sum([s._kernel for s in self._modelList])
        
        self.score()






    def fit(self, method=None, max_periodic = None,  stationary_kernel = None):
        xtrain = self._xtrain
        ytrain = self._ytrain

        dictio = {'Stationary':3 ,"Linear":1 ,"Polynomial":0, "Periodic":2,"Matern12":4}
        if max_periodic is None:
            max_periodic = 2
        if stationary_kernel is None:
            stationary_kernel = 'Matern32'

        if self._kernel is None:
            return self.autofit(max_periodic,  stationary_kernel)
            
        kernel_expr = self._kernel

        #ker_simple = self.simple_kernel_predict()
        #print("ker_simple :" ,ker_simple)
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
                #print("step --> {} ...".format(model._kernel.label()))
                #print("finished.")

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

        self.score()







    

   

    def predict(self, xt=None, yt=None, horizon=1,option=True, sparse = None, sparse_size=None, components=None,return_value=False, ProgressBar = True):

        self._xtest = xt

        if self.transform is not None:
            if yt is not None:
                yt = self.transform(yt)

        #if return_value is None:
         #   return_value = False

        if xt is None:
            ypred_,std_ = self._yfit, self._std_yfit
            cmps = self.components

        else:

            if self._modelList.__class__.__name__ == "GaussianProcessRegressor": 
                #ypred_, std_ = self._modelList.predict(xt,yt,horizon,option)
                ypred_, std_ = self._modelList.updated_predict(xt,yt,horizon, ProgressBar)

                components = False
            else:
                models = self._modelList
                if models is None:
                    raise NotFittedError("This GaussianProcessesRegressor instance is not fitted yet")
                yt_std_list = []
                yt_pred_list = []
                step = 1

                for mdl in models:
                    #print("step --> {} ...".format(mdl._kernel.label()))
                    
                    if yt is not None:
                        if mdl._kernel.label() in ['Periodic','Polynomial','Linear']:
                            ProgressBar = False
                        else:
                            ProgressBar = True
                        #if ProgressBar:
                        #    print('the {}-th step --> ...'.format(step).center(50))
                        #    step += 1

                    #yt_pred, yt_std = mdl.predict(xt,yt,horizon, option, sparse, sparse_size)
                    yt_pred, yt_std = mdl.updated_predict(xt,yt,horizon, ProgressBar)

                    if yt is not None:
                        zz = yt_pred[:-horizon]
                        yt = yt - zz[:yt.size]

                    yt_pred_list.append(yt_pred)
                    yt_std_list.append(yt_std)
                    if yt is not None:
                        if ProgressBar:
                            print("finished".center(50))

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
        if return_value:
            if components:
                return ypred_,std_, cmps
            else:
                return ypred_,std_


    def update(self, x_update = None , y_update=None, method = None):
        y_list = []
        yu = y_update
        x = x_update
        models = []
        if not isinstance(self._modelList,list):
            
            return self._modelList.update(x_update, y_update, method)
            #self._modelList.update(x_update, y_update, method)
            #self._xtrain = self._modelList._xtrain
            #self._ytrain = self._modelList._ytrain

        else:
            y = y_update
            for m in self._modelList:
                #copy_m = deepcopy(m)
                m.update(x_update,  y, method)
                #print(m._xtrain)
                yp, _ = m.predict(x_update)
                y = y - yp

            #self._xtrain = self._modelList[0]._xtrain
            #self._ytrain = self._modelList[0]._ytrain


    def deploy_(self, horizon = None,freq = None, filename = None):
        
        
        model_copy = deepcopy(self)

        f = np.diff(model_copy._xtrain).mean()
        f =  pd.to_timedelta(f)
        if freq is None:
            freq = f
        if filename is None:
            filename = 'MakePrediction.csv'
        
        fieldnames = ["future", "ypred","ypred_std"]
        os.makedirs('prediction_deployment', exist_ok = True)


        # with open(os.path.join('makeprediction_deploy',filename), 'w') as csv_file:
        #     csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        #     csv_writer.writeheader()

        if horizon is None:
            if 'Periodic' in model_copy._kernel.label().split(' + '):
                hyp_periodic = list(filter(lambda w:w[0]=='Periodic',model_copy.get_hyperparameters()))
                max_period = max([w[1]['period'] for w in hyp_periodic])
                horizon = int(max_period*model_copy._ytrain.size)
            else:
                horizon = int(.1*model_copy._ytrain.size)
            #print('Automatic horizon detected is ~= ', horizon)
        
        try:
            while True:
                #print(df_realtime.shape)

                
                d=datetime.datetime.now()
                #dround = d - datetime.timedelta(microseconds=d.microsecond)

                future = pd.date_range(start = d + freq  ,periods = horizon, freq = freq)
                #print(horizon,future.size)
                ypred, ypred_std = model_copy.predict(future , return_value = True)
                 

                    
                #prediction.to_csv(os.path.join(makeprediction_deploy,filename),  index=False)

                ##################################################
                # with open(os.path.join('makeprediction_deploy',filename), 'a') as csv_file:
                #     csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                #     info = {
                #         "future":  future[0].strftime("%m/%d/%Y, %H:%M:%S.%f"),
                #         "ypred": ypred[0],
                #         "ypred_std":ypred_std[0],
                #        }


                #     csv_writer.writerow(info)
                #     print(info)
                #############################################
                file = os.path.join('makeprediction_deploy',filename)
                prediction = pd.DataFrame({'future':future,'ypred':ypred,'ypred_std': ypred_std,})
                prediction.to_csv(file)
                print(prediction.head(1))
                self.deploy_file = file

                time.sleep(freq.total_seconds())
        except KeyboardInterrupt:
            #print('====='.center(100))
            print((100*'#').center(100))


            print('Prediction interrupted by user.')










        



        
    def deploy(self,realtime_data = None, xname = None, yname = None, horizon = None,freq = None, filename = None):
        

        train_size = self._ytrain.size
        model_copy = deepcopy(self)
        f = np.diff(model_copy._xtrain).mean()
        #val_f = f
        f =  pd.to_timedelta(f)


        dir_ = 'prediction_deployment'

        os.makedirs(dir_, exist_ok = True)

        if freq is None:
            freq = f

        if filename is None:
            filename = 'deploy.csv'
            #now = datetime.datetime.now()
            #filename = str(now.timestamp()) + '.csv'



        
        if realtime_data is None:
            return self.deploy_(horizon,freq,filename)
        else:
            df_realtime = pd.read_csv(realtime_data)
            colNames = df_realtime.columns[:2]
            if xname is None:
                xname = colNames[0]
            if yname is None:
                yname = colNames[1]




        #df_realtime = pd.read_csv(realtime_data)
        #print(df_realtime.shape)

        #df_realtime = pd.read_csv(realtime_data,names= ['date','y_pred','std_pred'])

        
        skip_rows = df_realtime.shape[0]
        
        df_realtime[xname] = pd.to_datetime(df_realtime[xname])
        
        
        
        #fieldnames = ["date", "y_pred","y_pred_std"]
        
        
        #model = qgpr()
        #model = model.load(model_path)
        
        
        df_update = df_realtime[-train_size:]
        model_copy.update(df_update[xname],df_update[yname])
        
        t0, y0 =  df_update.iloc[-1]

        
        

        #with open(filename, 'w') as csv_file:
         #   csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
         #   csv_writer.writeheader()
        
        if horizon is None:
            if 'Periodic' in model_copy._kernel.label().split(' + '):
                hyp_periodic = list(filter(lambda w:w[0]=='Periodic',model_copy.get_hyperparameters()))
                max_period = max([w[1]['period'] for w in hyp_periodic])
                horizon = int(max_period*model_copy._ytrain.size)
            else:
                mm = model_copy._ytrain.size
                if mm>10000:
                    h = .01
                elif (mm>5000):
                    h = .05
                else:
                    h = .1

                horizon = int(h*model_copy._ytrain.size)
            #print('Automatic horizon detected is ~= ', horizon)


        try:
            while True:
                #print(df_realtime.shape)

                df_realtime = pd.read_csv(realtime_data, skiprows=range(1, skip_rows))
                #print(df_realtime.shape)
                skip_rows = df_realtime.shape[0] + skip_rows - 1

                df_realtime[xname] = pd.to_datetime(df_realtime[xname])
                t1 = df_realtime[xname].iloc[-1]
                df = df_realtime[df_realtime[xname] > t0]
                #d=datetime.datetime.now()
                #dround = d - datetime.timedelta(microseconds=d.microsecond)
                if t1>t0:
                    model_copy.update(df.date,df.value.values)

                    dround = df.date.iloc[-1] + freq
                    #dround = df.date.iloc[-1].replace(microsecond=0)

                    future = pd.date_range(start = dround ,periods = horizon, freq = freq)
                    ypred, ypred_std = model_copy.predict(future , return_value = True)
                    t0 = t1


                
                    prediction = pd.DataFrame({'future':future,'ypred':ypred,'ypred_std': ypred_std,})
                
                

                    file = os.path.join(dir_,filename)

                    prediction.to_csv(file,  index=False)

                    
                
        except KeyboardInterrupt:
            print((100*'#').center(100))
            print('Prediction interrupted by user.')



    
    
    def deploy2dashbord(self,realtime_db = None, deployment_file = None, prediction_horizon = 1, max_horizon=None, freq = None, port = None, filename = None, return_value = False, refresh_time_in_milliseconds = 1000):
        
        if deployment_file is None:
            deployment_file = 'deployment_file.csv'
            
        db_deploy = 'prediction_deployment'
        db_deploy = os.path.join(db_deploy,deployment_file)
        
        kwargs = {'realtime_data':realtime_db , 'horizon' : max_horizon, 'freq' : freq, 'filename': deployment_file}
        if realtime_db is None:
            kwargs = {'horizon' : max_horizon, 'freq' : freq, 'filename': deployment_file}

            process = multiprocessing.Process(target=self.deploy, kwargs = kwargs)
            process.start()
        else:
            
            process = multiprocessing.Process(target=self.deploy,kwargs = kwargs)
            process.start()
           
        
        
        if len(self._score) == 2:
            RMSE_train = np.sqrt(self._score['train_errors']['MSE'])
            RMSE_test = np.sqrt(self._score['test_errors']['MSE'])
        else:
            RMSE_train = np.sqrt(self._score['train_errors']['MSE'])



        dir_ = 'predicted_horizon_files'
        os.makedirs(dir_, exist_ok = True)


        if filename is None:
            filename = 'prediction_with_horizon_{}.csv'.format(prediction_horizon)
            
        filename = os.path.join(dir_,filename)

        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                    meta_tags=[{'name': 'viewport',
                                'content': 'width=device-width, initial-scale=1.0'}]
                    )

        df = pd.DataFrame(columns=['date', 'value'])
        df_pred = pd.DataFrame(columns=['date', 'ypred','ypred_std'])

        df_pred_horizon = pd.DataFrame(columns=['date', 'ypred_horizon','ypred_std_horizon'])
        df_pred_horizon.to_csv(filename,  index=False)
        
        
        logo = dbc.Col(dbc.CardImg(
                            src="/assets/mp_logo_orange.png",style={"width": "6rem"},
                            top=True,))
        
        app.layout = html.Div([dbc.Row([html.Div(logo,style = style_logo_left),dbc.Col(html.Div(html.H1("MakePrediction DashBoard", style = {"color":'#d35400',}),
                                                                                        style = {'backgroundColor': '#e5ecf6',
                                                                                        'padding': '30px 0 30px 0',
                                                                                        'border': '2px solid green',
                                                                                        'fontfamily': 'sans-serif',
                                                                                        "textAlign": "center",
                                                                                       }),width={'size': 8, 'offset': 0})
                                        ,html.Div(logo,style = style_logo_right)]
                    ),
            dcc.Graph(
                id='graphid',
                figure={
                    'data': [
                        go.Scatter(x=df['date'], y=df['value'], mode = 'lines+markers')
                    ],
                    'layout': {
                        'title': 'MakePrediction',
                    },
                }
            ),
                               
                            html.Div([dbc.Row(
                                [dbc.Col(width={'size': 1, 'offset': 0}),dbc.Col(html.Div(html.H5("Realtime DataTable",style = {"color":'#d35400'}),style = style_text),
                            width={'size': 3, 'offset': 1}),
                                dbc.Col(width={'size': 1, 'offset': 0}),
                                dbc.Col(html.Div(html.H5("Predicted DataTable",style = {"color":'#d35400'}),style = style_text),
                            width={'size': 4, 'offset': 0})]
                    )]),
                               
              dbc.Row([dbc.Col(width={'size': 1, 'offset': 0}), dbc.Col(DataTable(id='table',
                                    data=[],
                                    columns=[{'id': c, 'name': c} for c in df.columns],
                                    style_cell = styleCell,
                                    
                                    style_header={'backgroundColor': '#e5ecf6','fontWeight': 'bold', 'color':'blue'},),
                                    width={'size': 3, 'offset': 1},),
                     dbc.Col(width={'size': 1, 'offset': 0}),
                     dbc.Col(DataTable(id='table_pred',
                                    data=[],
                                    columns=[{'id': c, 'name': c} for c in df_pred.columns],
                                    style_cell= styleCell ,
                                    
                                    style_header={'backgroundColor': '#e5ecf6','fontWeight': 'bold', 'color':'blue'},),
                                    width={'size': 4, 'offset': 0})]),
                               
            dcc.Interval(
                id='1-second-interval',
                interval=refresh_time_in_milliseconds, # 2000 milliseconds = 2 seconds
                n_intervals=0
            ),
            
        ])

        @app.callback(Output('graphid', 'figure'),
                      [Input('1-second-interval', 'n_intervals')])
        def update_layout(n):
            
            
            if realtime_db is None:
                pass
            else:
                path_db = realtime_db
                df = pd.read_csv(path_db)
                df.date = pd.to_datetime(df.date)
               
            path_db_pred = db_deploy
            df_pred = pd.read_csv(path_db_pred)
            df_pred.future = pd.to_datetime(df_pred.future)
            f,yp,yp_std = df_pred.iloc[prediction_horizon - 1]
            if f not in df_pred_horizon['date'].values:
                df_pred_horizon.loc[n] = [f,yp,yp_std]
                df_pred_horizon.to_csv(filename,  index=False)
            intersect_df = pd.merge(df, df_pred_horizon, how='inner', on='date')
            
            rmse = lambda predictions,targets: np.sqrt(np.mean((predictions-targets)**2))
            
            RMSE = rmse(intersect_df.value,intersect_df.ypred_horizon)  
            
            if len(self._score)==2: 
                title = f'RMSE errors (train/test/realtime_test): {round(RMSE_train,3)}/{round(RMSE_test,3)}/{round(RMSE,3)}'
            else:
                title = f'RMSE errors (train/realtime_test): {round(RMSE_train,3)}/{round(RMSE,3)}'

            if realtime_db is None:
                pass
            else:
                trace1 = go.Scatter(x=df['date'], y=df['value'], mode = 'lines+markers',
                                  line_color = 'rgb(0,0,0)',
                                    
                               name='Realtime data streaming')
            trace2 = go.Scatter(x=df_pred['future'], y=df_pred['ypred'], mode = 'lines',
                                  line_color = 'rgba(0,0,255,.8)',
                                  name='Long-term prediction',)
            trace3 = go.Scatter(x=df_pred_horizon['date'], y=df_pred_horizon['ypred_horizon'], 
                                mode = 'markers + lines',
                                name=f'Short-term prediction(prediction horizon = {prediction_horizon})',

                                  line_color = 'rgba(255,0,0,.8)')
            
            
            
            
            
            xs_list = df_pred.future.tolist()
            xs_rev = xs_list[::-1]
            yp_upper = df_pred.ypred + 1.96*df_pred.ypred_std
            yp_upper = yp_upper.tolist()
            yp_lower = df_pred.ypred - 1.96*df_pred.ypred_std
            yp_lower = yp_lower.tolist()
            yp_lower_rev = yp_lower[::-1]



            
            trace4 = go.Scatter(
                                x =xs_list+xs_rev,
                                y=yp_upper+yp_lower_rev,
                                fill='toself',
                                fillcolor='rgba(255,0,0,.1)',
                                line_color='rgba(255,255,255,0)',
                                name='Confidence Interval (95%)',
                                showlegend=True,

                            )
            linedate = df.date.iloc[-1]
            now = datetime.datetime.now()
            
            if realtime_db is None:
                figure={
                    'data': [ trace2, trace3,trace4,],
                    'layout': {
                        'title': 'MakePrediction deployement'
                    }
                }
            else:
                figure={
                    'data': [trace1, trace2, trace3,trace4,],
                    'layout': {
                        'title': f'Realtime deployement ({linedate}):' + title,
                        'font_color' : "blue",
                        'title_font_color': "green",
                        
                         #'width':900,
                        'height':600,

                    }
                }
            fig  = go.Figure(figure)
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                font = dict(size = 14),
                x=1))

            
            
            fig.update_xaxes({'title':'date'})
            fig.update_yaxes({'title':'value'})
            fig.update_layout(#title='Dates are pandas timestamps',
                              #yaxis_title='AAPL Stock',
                              shapes = [dict(x0=linedate, x1=linedate, y0=0, y1=1, xref='x', yref='paper',line_width=1,line=dict(
            color="green",
            width=.5,
            dash="dashdot",
        ))],)
            

            return fig
        

        @app.callback(
        [Output("table", "data"), Output('table', 'columns')],
        [Input('1-second-interval', 'n_intervals')])
        def updateTable(n):
            if realtime_db is None:
                pass
            else:
                path_db = realtime_db
                df = pd.read_csv(path_db)
            return df.tail().round(decimals = 3).values, [{"id": i, "name": df.columns[i].capitalize()} for i in range(df.shape[1])]
        
        
        @app.callback(
        [Output("table_pred", "data"), Output('table_pred', 'columns')],
        [Input('1-second-interval', 'n_intervals')])
        def updateTable_pred(n):
            path_db_pred = db_deploy

            df_pred = pd.read_csv(path_db_pred)
            
            return  df_pred.head().round(decimals = 3).values, [{"id": i, "name": '' + df_pred.columns[i].capitalize()} for i in range(df_pred.shape[1])]

        if port is None:
            port = 9412
        
        app.run_server(debug = False,port = port)
            
        if return_value:
            return df_pred_horizon
        
            