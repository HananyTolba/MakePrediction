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



import os
import logging

logging.basicConfig(level=logging.ERROR)

#URL = 'http://www.makeprediction.com/periodic/v1/models/periodic_1d:predict'
#URL_IID = 'http://makeprediction.com/iid/v1/models/iid_periodic_300:predict'

simple_DomainName = "https://simple.makeprediction.com"
periodic_DomainName = "https://periodic.makeprediction.com"

periodic_models = [
          #"periodic_1d",
          "iid_periodic_300",
          "tf_Periodic_ls",
          "tf_PeriodicPlusMatern",
          "tf_PeriodicPlusRBF",
          "periodic_model",
          ]


simple_models = ["rbf_1d",
          "matern12_1d",
          "matern32_1d",
          "matern52_1d",
          "linear_1d",
          "polynomial_1d",
          "model_expression_predict",
          "gp_kernel_predict_300",
          "gp_kernel_predict_simple_300",
          "periodic_kernel_predict",
          "linear_kernel_predict"
          ]

# def kernel2url(kernel_str):
#     #kernel_str = self._kernel.label().lower()
#     if kernel_str  in simple_models:
#         url_ec2 = os.path.join(simple_DomainName,kernel_str + "/v1/models/" + kernel_str + ":predict")

#     else:
#         kernel_str_1d = kernel_str + "_1d"
#         url_ec2 = os.path.join(simple_DomainName,kernel_str + "/v1/models/" + kernel_str_1d + ":predict")
    
#     if kernel_str == "periodic":
#         url_ec2_noise = os.path.join(periodic_DomainName,"iid_periodic_300" + "/v1/models/" + "iid_periodic_300" + ":predict")
#         url_ec2_ls = os.path.join(periodic_DomainName,"tf_Periodic_ls" + "/v1/models/" + "tf_Periodic_ls" + ":predict")

#         return url_ec2, url_ec2_ls, url_ec2_noise
#     else:
#         return url_ec2 



def kernel2url(kernel_str):
    #kernel_str = self._kernel.label().lower()
    sub_models = ["model_expression_predict",
          "gp_kernel_predict_300",
          "gp_kernel_predict_simple_300",
          "periodic_kernel_predict",
          "linear_kernel_predict",

          ]
    if (kernel_str  in sub_models):
        url_ec2 = os.path.join(simple_DomainName,kernel_str + "/v1/models/" + kernel_str + ":predict")

    else:
        kernel_str_1d = kernel_str + "_1d"
        url_ec2 = os.path.join(simple_DomainName,kernel_str + "/v1/models/" + kernel_str_1d + ":predict")
  
    return url_ec2 

def periodic2url(kernel_str, method = None):
    if method is None: method = "periodic"
    if kernel_str.lower() == "periodic":
        if method == 2:
            periodic_method = "tf_PeriodicPlusMatern"
        elif method == 1:
            periodic_method = "tf_PeriodicPlusRBF"
        else:
            periodic_method = "periodic_model"


        url_ec2_period = os.path.join(periodic_DomainName,periodic_method + "/v1/models/" + periodic_method + ":predict")
        url_ec2_noise = os.path.join(periodic_DomainName,"iid_periodic_300" + "/v1/models/" + "iid_periodic_300" + ":predict")
        url_ec2_ls = os.path.join(periodic_DomainName,"tf_Periodic_ls" + "/v1/models/" + "tf_Periodic_ls" + ":predict")
    return url_ec2_period, url_ec2_ls, url_ec2_noise

