

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


import logging

logging.basicConfig(level=logging.ERROR)

from concurrent.futures import ThreadPoolExecutor
from makeprediction.invtools import date2num
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

#from makeprediction.gp import date2num


import matplotlib.pyplot as plt 

import json

import numpy as np
from scipy.signal import resample
from scipy import interpolate
from makeprediction.url import periodic2url
#from makeprediction.quasigp import QuasiGPR as QGPR 


#from makeprediction.log_lh import log_lh_stable

# URL = 'http://www.makeprediction.com/periodic/v1/models/periodic_1d:predict'
# URL_IID = 'http://makeprediction.com/iid/v1/models/iid_periodic_300:predict'
# URL = "localhost:8533/v1/models/PeriodicPlusRBF:predict"


from collections import Counter





def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 





SMALL_SIZE = 300








def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        method_whitelist=frozenset(['GET', 'POST']),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def fetch(session, url,data):
    with session.post(url,data=json.dumps(data)) as response:
        result = np.array(response.json()["outputs"][0])
        return result



def thread_fit(self,method = None):
    (URL,URL_LS ,URL_IID) = periodic2url("periodic",method)

    x,y = self._xtrain, self._ytrain

    x = date2num(x)

    


    std_y = y.std()
    y =  (y - y.mean())/std_y


    min_p = 50/y.size 
    #p_large = np.linspace(.2,1,100)
   # p_small = np.linspace(min_p,.2,100)
    #p = np.hstack([p_small,p_large])

    p = np.linspace(min_p,1,100)
    mm = y.size
    int_p = mm*p
    int_p = int_p.astype(int)
    int_p = np.unique(int_p)
    
    y_list = [y[-s:] for s in int_p]

    
    data_list = []
    #yr_list = []
    for _ in y_list:
        f = interpolate.interp1d(np.linspace(-1,1,len(_)), _)
        xnew = np.linspace(-1,1,SMALL_SIZE)
        ynew = f(xnew)
        ynew =  (ynew - ynew.mean())/ynew.std()
        #plt.plot(ynew)
        #plt.show()
        #yr_list.append(ynew)

        data = {"inputs":ynew.reshape(1,-1).tolist()}
        data_list.append(data)



    tt = len(data_list)

    with requests.Session() as session:
        std_noise = fetch(session, URL_IID,data_list[-1])
        #length_scale = fetch(session, URL_LS,data_list[-1])

    self._sigma_n = std_noise[0]*std_y

    result = []


    with ThreadPoolExecutor(max_workers=50) as executor:
        #with requests.Session() as session:
        result += executor.map(fetch, [requests_retry_session()] * tt, [URL] * tt,data_list)
        executor.shutdown(wait=True)



    result = np.array(result).ravel()

    result= result*np.array(int_p)/mm
    sd_result = (result-result.mean())/result.std()
    p_estimate_= sd_result[np.argmin(np.abs(np.diff(sd_result)))]
    p_estimate = result[sd_result == p_estimate_]

   
    sd_result_round = np.round(sd_result,2)
    p_freq_round = most_frequent(sd_result_round)
    p_freq_set =  result[sd_result_round == p_freq_round]
    p_freq = p_freq_set.mean()

    z_ls = y[:int(2*p_freq*y.size)]
    z_ls_resized = resample(z_ls,SMALL_SIZE)
    z_ls_resized =  (z_ls_resized - z_ls_resized.mean())/z_ls_resized.std()
    data_ls = {"inputs":z_ls_resized.reshape(1,-1).tolist()}

    with requests.Session() as session:
        length_scale = fetch(session, URL_LS,data_ls)
    length_scale = length_scale[0]
    
    hyp_dict1 = dict(zip(["length_scale","period"],[length_scale,result[-1]]))
    hyp_dict1["variance"] = std_y**2

    hyp_dict2 = dict(zip(["length_scale","period"],[length_scale,p_freq]))
    hyp_dict2["variance"] = std_y**2

    hyp_dict3 = dict(zip(["length_scale","period"],[length_scale,p_estimate[0]]))
    hyp_dict3["variance"] = std_y**2

    #self.set_hyperparameters(hyp_dict)
    #print(hyp_dict2,hyp_dict3)
    #return hyp_dict3,hyp_dict3

    return hyp_dict2,hyp_dict3







def thread_interfit(self):
    x,y = self._xtrain, self._ytrain
    x = date2num(x)
    f = interpolate.interp1d(np.linspace(-1,1,y.size), y)
    xnew = np.linspace(-1,1,int(x.size*5))
    ynew = f(xnew)
    #x_plus = np.linspace(x[0],  x[-1],int(x.size*5) )
    #x_plus = np.linspace(-1,  1,int(x.size*5) )

    #y_plus = resample(y, int(x.size*5))
    self._xtrain, self._ytrain = xnew, ynew
    L = thread_fit(self)
    #print(L)
    self._xtrain, self._ytrain = x, y

    return L

def date2num_old(dt):
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

