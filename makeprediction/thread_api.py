
from concurrent.futures import ThreadPoolExecutor
#from makeprediction.gp import date2num
import requests
#from makeprediction.gp import date2num


import matplotlib.pyplot as plt 

import json

import numpy as np
from scipy.signal import resample


URL = 'http://www.makeprediction.com/periodic/v1/models/periodic_1d:predict'
URL_IID = 'http://makeprediction.com/iid/v1/models/iid_periodic_300:predict'

from collections import Counter

def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 





SMALL_SIZE = 300





def fetch(session, url,data):
    with session.post(url,data=json.dumps(data)) as response:
        result = np.array(response.json()["outputs"][0])
        return result



def thread_fit(self):
    x,y = self._xtrain, self._ytrain

    x = date2num(x)


    std_y = y.std()

    y = (y - y.mean())/std_y

    min_p = 50/x.size
    
    p = np.linspace(min_p,1,100)
    mm = y.size
    y_list = [y[:int(s*mm)] for s in p]

    y_list = [list(x) for x in set(tuple(x) for x in y_list)]
    y_list.sort(key=len)

    # plt.plot(y_list[3],'o')
    # plt.plot(y_list[20],'x')
    # plt.plot(y_list[99])
    # plt.show()
    #y_list2 = [y[-int(s*mm):] for s in p]
    #y_list = y_list1 + y_list2
    nn = list(map(len,y_list))

    #print("nombre de listes :",len(nn))
    data_list = []
    for _ in y_list:
        z = resample(_,SMALL_SIZE)
        z =  (z - z.mean())/z.std()

        data = {"inputs":z.reshape(1,-1).tolist()}
        data_list.append(data)
    tt = len(data_list)

    with requests.Session() as session:
        std_noise = fetch(session, URL_IID,data_list[-1])
    self.std_noise = std_noise
    result = []


    with ThreadPoolExecutor(max_workers=10) as executor:
        with requests.Session() as session:
            result += executor.map(fetch, [session] * tt, [URL] * tt,data_list)
            executor.shutdown(wait=True)



    result = np.array(result)

    result[:,1] = result[:,1]*np.array(nn)/mm

    #plt.plot(result[:,1])
    #plt.show()

    #print("most frequent : ", most_frequent(np.round(result[:,1].ravel(),2)))


    hyp = result[-1,:]

    if result[-1,1]>=.99:
        hyp = result[-1,:]
    else:
        hyp = result[-1,:]
        L = result[:,-1]
        print("erreur : ",np.abs(np.diff(L)).min())
        hyp[-1]  = L[np.argmin(np.abs(np.diff(L)))]


    #L = result[:,-1]
    #hyp[-1]  = L[np.argmin(np.abs(np.diff(L)))]
    #hyp[-1] = most_frequent(np.round(result[:,1].ravel(),2))

    # if hyp[-1]<.01:
    #     hyp[-1] = round(hyp[-1] ,4)
    # elif hyp[-1]<.1:
    #     hyp[-1] = round(hyp[-1] ,3)
    # else:
    #     hyp[-1] = round(hyp[-1] ,2)



    hyp_dict = dict(zip(["length_scale","period"],hyp))
    hyp_dict["variance"] = std_y**2

    self.set_hyperparameters(hyp_dict)










def thread_interfit(self):
    x,y = self._xtrain, self._ytrain
    x = date2num(x)
    x_plus = np.linspace(x[0],  x[-1],int(x.size*5) )
    y_plus = np.interp(x_plus, x, y)
    self._xtrain, self._ytrain = x_plus, y_plus
    thread_fit(self)

    self._xtrain, self._ytrain = x, y

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











def thread_intersplitfit(self):
    x,y = self._xtrain, self._ytrain
    x = date2num(x)
    x_plus = np.linspace(x[0],  x[-1],int(x.size*5) )
    y_plus = np.interp(x_plus, x, y)
    self._xtrain, self._ytrain = x_plus, y_plus
    thread_splitfit(self)

    self._xtrain, self._ytrain = x, y







def thread_splitfit(self):
    x,y = self._xtrain, self._ytrain

    x = date2num(x)


    std_y = y.std()

    y = (y - y.mean())/std_y

    min_p = 50/x.size
    
    p = np.linspace(min_p,1,100)
    mm = y.size
    y_list = [y[:int(s*mm)] for s in p]

    y_list = [list(x) for x in set(tuple(x) for x in y_list)]
    y_list.sort(key=len)

    # plt.plot(y_list[3],'o')
    # plt.plot(y_list[20],'x')
    # plt.plot(y_list[99])
    # plt.show()
    #y_list2 = [y[-int(s*mm):] for s in p]
    #y_list = y_list1 + y_list2
    nn = list(map(len,y_list))

    #print("nombre de listes :",len(nn))
    data_list = []
    for _ in y_list:
        x_interp = np.linspace(-1, 1, SMALL_SIZE )

        x_transform, a, b = self.line_transform(np.linspace(-1, 1, len(_) ).reshape(-1, 1))

        y_interp = np.interp(x_interp, x_transform, _)
        #print("shape_periodic : ",y_interp.shape)
        #period_est_ = get_parms_from_api(y_interp,self._kernel.label())

        z = y_interp
        z =  (z - z.mean())/z.std()

        data = {"inputs":z.reshape(1,-1).tolist()}
        data_list.append(data)
    tt = len(data_list)

    with requests.Session() as session:
        std_noise = fetch(session, URL_IID,data_list[-1])
    self.std_noise = std_noise
    result = []


    with ThreadPoolExecutor(max_workers=10) as executor:
        with requests.Session() as session:
            result += executor.map(fetch, [session] * tt, [URL] * tt,data_list)
            executor.shutdown(wait=True)



    result = np.array(result)

    result[:,1] = result[:,1]*np.array(nn)/mm

    plt.plot(result[:,1])
    plt.show()

    #print("most frequent : ", most_frequent(np.round(result[:,1].ravel(),2)))


    hyp = result[-1,:]

    if result[-1,1]>=.99:
        hyp = result[-1,:]
    else:
        hyp = result[-1,:]
        L = result[:,-1]
        print("erreur : ",np.abs(np.diff(L)).min())
        hyp[-1]  = L[np.argmin(np.abs(np.diff(L)))]


    #L = result[:,-1]
    #hyp[-1]  = L[np.argmin(np.abs(np.diff(L)))]
    #hyp[-1] = most_frequent(np.round(result[:,1].ravel(),2))

    # if hyp[-1]<.01:
    #     hyp[-1] = round(hyp[-1] ,4)
    # elif hyp[-1]<.1:
    #     hyp[-1] = round(hyp[-1] ,3)
    # else:
    #     hyp[-1] = round(hyp[-1] ,2)



    hyp_dict = dict(zip(["length_scale","period"],hyp))
    hyp_dict["variance"] = std_y**2

    self.set_hyperparameters(hyp_dict)