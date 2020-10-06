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

logging.basicConfig(level=logging.DEBUG)

URL = 'http://www.makeprediction.com/periodic/v1/models/periodic_1d:predict'
URL_IID = 'http://makeprediction.com/iid/v1/models/iid_periodic_300:predict'

DomainName = "http://www.makeprediction.com"

models = ["rbf_1d","matern12_1d","matern32_1d","matern52_1d",
          "linear_1d",
          "polynomial_1d",
          "periodic_1d",
          "iid_periodic_300",
          "model_expression_predict",
          "gp_kernel_predict_300",
          "gp_kernel_predict_simple_300",
          ]

def kernel2url(kernel_str):
    #kernel_str = self._kernel.label().lower()
    if kernel_str  in models:
        url_ec2 = os.path.join(DomainName,kernel_str + "/v1/models/" + kernel_str + ":predict")

    else:
        kernel_str_1d = kernel_str + "_1d"
        url_ec2 = os.path.join(DomainName,kernel_str + "/v1/models/" + kernel_str_1d + ":predict")
    
    if kernel_str == "periodic":
        url_ec2_noise = os.path.join(DomainName,"iid_periodic_300" + "/v1/models/" + "iid_periodic_300" + ":predict")
        return url_ec2, url_ec2_noise
    else:
        return url_ec2 




