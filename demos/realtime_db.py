#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Author = "Hanany Tolba"
#01/02/2021

# __author__ = "Hanany Tolba"



from makeprediction.ts_generation import rtts
import numpy as np
def func(t):
    f_t  = 100*np.sin(2*np.pi*t/500)*np.sin(2*np.pi*t/3003)  + 500  + 7*np.random.randn(1)[0]
    return f_t

if '__main__' == __name__:
    rtts(function = func,step = 3,filename = 'live_db.csv')