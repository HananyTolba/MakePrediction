import asyncio

import aiohttp

from timer import timer

URL = 'http://www.makeprediction.com:8507/v1/models/periodic_1d:predict'
#from makeprediction.gp import get_parms_from_api

import json

import numpy as np
data_list = []
for _ in range(100):
    y = np.random.randn(1,300)
    data = {"inputs":y.tolist()}
    data_list.append(data)

async def fetch(session, url,data):
    async with session.post(url,data=json.dumps(data)) as response:
        json_response = await response.json()
        print(np.array(json_response["outputs"][0]))




async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, URL,_) for _ in data_list]
        await asyncio.gather(*tasks)


@timer(1, 10)
def func():
    #asyncio.run(main())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())


