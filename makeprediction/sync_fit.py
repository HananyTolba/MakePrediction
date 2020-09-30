URL = 'http://www.makeprediction.com/toto/v1/models/periodic_1d:predict'
#from makeprediction.gp import get_parms_from_api
import requests
from timer import timer
import json

import numpy as np
from makeprediction.kernels import *  

import matplotlib.pyplot as plt 

x = np.linspace(-1,1,1000)


y = Periodic(period = .1).simulate(x) + .1*np.random.randn(x.size)




def p_for_fit(x,y):
    ystd = y.std()
    y = (y - y.mean()) / y.std()

    
    n = y.size
#=================================================================
    x_interp = np.linspace(-1, 1, SMALL_SIZE )

    x_transform, a, b = gpr().line_transform(x.reshape(-1, 1))

    y_interp = np.interp(x_interp, x_transform, y)
    #print("shape_periodic : ",y_interp.shape)
    #period_est_ = get_parms_from_api(y_interp,self._kernel.label())

    return y_interp



def split_ts(y):
    m = 100

    r_list = [y[:int((i+1)/m*y.size)] for i in range(m) if int((i+1)/m*y.size)>=100]
    return r_list

from makeprediction.gp import GaussianProcessRegressor as gpr  


SMALL_SIZE = 300

def data_resize(args):
    y_interp_list = []
    for y in args:
        ystd = y.std()
        y = (y - y.mean()) / y.std()
        n = y.size
#=================================================================
        x_interp = np.linspace(-1, 1, SMALL_SIZE )

        x_transform, a, b = gpr().line_transform(y.reshape(-1, 1))
        y_interp = np.interp(x_interp, x_transform, y)
        y_interp_list.append(y_interp)
    
    return y_interp_list

all_list = split_ts(y)

data_ = data_resize(all_list)

data_list = []
for _ in data_:
#     y = np.random.randn(1,300)
    data = {"inputs":_.reshape(1,-1).tolist()}
    data_list.append(data)



print(len(data))


def fetch(session, url,data):
    with session.post(url,data=json.dumps(data)) as response:
        #print(np.array(response.json()["outputs"][0]))
        l = np.array(response.json()["outputs"][0])*np.array([1,])
    return l 



L = []
with requests.Session() as session:
    for _ in data_list:
        L.append(fetch(session, URL,_))

length = list(map(len, all_list))
parms = np.array(L)

parms[:,1] = parms[:,1]*np.array(length)/y.size



#parms[np.argmin(np.diff(parms[1])),1]
# #.mean(axis=1).tolist()
# d = dict(zip(["length_scale","period"],parms)) 
# d["variance"] = y.std()
# #d["variance"] = y.std()

# print(d)
# model = gpr(x,y)
# model.choice("periodic")

# model.set_hyperparameters(d)
# yp, _ = model.predict()


plt.figure()
plt.plot(parms[:,1],'k')
#plt.plot(x,yp,'r')
plt.show()


