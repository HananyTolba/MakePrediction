import asyncio

import aiohttp

from timer import timer

URL = 'http://www.makeprediction.com/periodic/v1/models/periodic_1d:predict'
#from makeprediction.gp import get_parms_from_api

import json

import numpy as np

x = np.linspace(0,10,1000)

y = np.sin(10*x) + np.sin(7*x) + .1*np.random.randn(x.size)
p = np.linspace(0.1,1,100)
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

async def fetch(session, url,data):
    async with session.post(url,data=json.dumps(data)) as response:
        json_response = await response.json()
        parms = np.array(json_response["outputs"][0])
        print(parms)




async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, URL,_) for _ in data_list]
        await asyncio.gather(*tasks)


@timer(1, 1)
def func():
    #asyncio.run(main())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())

#loop = asyncio.get_event_loop()

#done, pending = loop.run_until_complete(asyncio.wait(main()))
# for future in done:
#     value = future.result() #may raise an exception if corou
#     print(values)

