
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import requests

from timer import timer


URL = 'http://www.makeprediction.com:8507/v1/models/periodic_1d:predict'
#from makeprediction.gp import get_parms_from_api
URL_IID = "http://www.makeprediction.com:8508/v1/models/iid_periodic_300:predict"
import json

import numpy as np

x = np.linspace(-1,1,1000)

#y = np.sin(1*x)  + .1*np.random.randn(x.size)
from makeprediction.kernels import *  

import matplotlib.pyplot as plt 

#x = np.linspace(-1,1,1000)

kernel  = Periodic(period = .05,length_scale=.6) #+ RBF(.3)

y = kernel.simulate(x) + .2*np.random.randn(x.size)

plt.figure()
plt.plot(x,y)
plt.show()

min_p = 100/x.size

p = np.linspace(min_p,1,500)
mm = y.size
y_list = [y[:int(s*mm)] for s in p]

from scipy.signal import resample

data_list = []
for _ in y_list:
#     y = np.random.randn(1,300)
    data = {"inputs":resample(_,300).reshape(1,-1).tolist()}
    data_list.append(data)


nn = list(map(len,y_list))
#print(nn)




def fetch(session, url,data):
    with session.post(url,data=json.dumps(data)) as response:
        result = np.array(response.json()["outputs"][0])
        return result


with requests.Session() as session:
	std_noise = fetch(session, URL_IID,data_list[-1])

# @timer(1, 1)
# def main():
result = []

tt = len(data_list)

with ThreadPoolExecutor(max_workers=tt) as executor:
#with ProcessPoolExecutor(max_workers=100) as executor:
    with requests.Session() as session:
        result += executor.map(fetch, [session] * tt, [URL] * tt,data_list)
        executor.shutdown(wait=True)

result = np.array(result)
import matplotlib.pyplot as plt 
plt.figure()
plt.plot(result[:,1]*np.array(nn)/mm)
plt.show()
