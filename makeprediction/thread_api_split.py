
from concurrent.futures import ThreadPoolExecutor

import requests



import json

import numpy as np
from scipy.signal import resample


URL = 'http://www.makeprediction.com/periodic/v1/models/periodic_1d:predict'
URL_IID = 'http://makeprediction.com/iid/v1/models/iid_periodic_300:predict'

def fetch(session, url,data):
    with session.post(url,data=json.dumps(data)) as response:
        result = np.array(response.json()["outputs"][0])
        return result



def thread_fit_split(self):
    x,y = self._xtrain, self._ytrain

    std_y = y.std()

    y = (y - y.mean())/std_y

    min_p = 100/x.size
    
    p = np.linspace(min_p,1,200)
    mm = y.size
    y_list = [y[:int(s*mm)] for s in p]


    #y_list2 = [y[-int(s*mm):] for s in p]
    #y_list = y_list1 + y_list2
    nn = list(map(len,y_list))
    #print("sizes is :",nn)
    data_list = []
    for _ in y_list:
        data = {"inputs":resample(_,300).reshape(1,-1).tolist()}
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


    if result[-1,1]>=.9:
        hyp = result[-1,:]
    else:
        hyp = result[-1,:]
        L = result[:,-1]
        hyp[-1]  = L[np.argmin(np.abs(np.diff(L)))]

    if hyp[-1]<.01:
        hyp[-1] = round(hyp[-1] ,4)
    elif hyp[-1]<.1:
        hyp[-1] = round(hyp[-1] ,3)
    else:
        hyp[-1] = round(hyp[-1] ,2)



    hyp_dict = dict(zip(["length_scale","period"],hyp))
    hyp_dict["variance"] = std_y**2

    self.set_hyperparameters(hyp_dict)





