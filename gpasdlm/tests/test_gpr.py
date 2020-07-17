#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gpasdlm.kernels import *
from gpasdlm.kernels import KernelSum, KernelProduct
from gpasdlm.gp import GaussianProcessRegressor as GPR

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
#from tensorflow.keras.models import load_model
#from tensorflow import keras

import tensorflow



x = np.linspace(-3,3,10)
f = lambda s: np.sin(s)
y = f(x).ravel()

#kernel = RBF()
kers =["periodic","matern12","linear","matern32","rbf","matern52","polynomial"]
@pytest.mark.parametrize('kernel', kers)
def test_gpr_kernel_choice(kernel):
	gpr = GPR(x,y)
	gpr.kernel_choice = kernel
	assert gpr._kernel.__class__.__name__.upper()== kernel.upper()

@pytest.mark.parametrize('kernel', kers)
def test_tf_model(kernel):
    gpr = GPR(x,y)
    gpr.kernel_choice = kernel
    #print(type(gpr._model))
    assert isinstance(gpr._model,tensorflow.keras.Model)


@pytest.mark.parametrize('kernel', kers)
def test_get_hyperparameters(kernel):
    gpr = GPR(x,y)
    gpr.kernel_choice = kernel
    parms = gpr.get_hyperparameters()
    if kernel == "periodic":
        assert parms["length_scale"] == 1
        assert parms["period"] == 1
        assert parms["variance"] == 1

    else:
        assert parms["length_scale"] == 1
        assert parms["variance"] == 1


@pytest.mark.parametrize('kernel', kers)
def test_set_hyperparameters(kernel):
    gpr = GPR(x,y)
    gpr.kernel_choice = kernel
    parms_per = {"length_scale":.5,"period":.5, "variance":2} 
    parms = {"length_scale":.5,"variance":2} 

    if kernel == "periodic":
        gpr.set_hyperparameters(parms_per)
        assert gpr.get_hyperparameters() == parms_per
        #assert all(v == actual_dict[k] for k,v expected_dict.items()) and len(expected_dict) == len(actual_dict)

    else:
        gpr.set_hyperparameters(parms)

        assert  gpr.get_hyperparameters() == parms



kernels =["rbf","matern12","matern32","matern52"]

@pytest.mark.parametrize('kernel', kernels)
def test_prediction(kernel):
    #kernel = "rbf"
    gpr = GPR(x,y)
    gpr.kernel_choice = kernel
    #gpr.std_noise =.001
    # Test the interpolating property for different kernels.
    gpr.fit()
    gpr.std_noise =.0001

    y_pred, y_cov = gpr.predict()
    #print(y_pred,"y : ", y)

    assert_almost_equal(y_pred, y,decimal = 5)
    assert_almost_equal(y_cov**2, 0.,decimal = 5)

# @pytest.mark.parametrize('kernel', kers)
# def test_simulate(kernel):
#     #kernel = "rbf"
#     gpr = GPR()
#     gpr.kernel_choice = kernel
#     # Test the interpolating property for different kernels.
#     assert gpr.simulate(x)


########################################################
# A = np.random.RandomState(314).normal(0, 1, (4,4))
# A = A + A.T
# m = np.random.RandomState(314).normal(0, 1, (4,1))
# r = np.array([1.])

# inv_A = np.linalg.inv(A)

# A_augmented = np.block([[A, m], [m.T, r]])
# gpr = GPR(x,y)
# inv_A_augmented = gpr._inv_add_update(inv_A, m, r)
# def test_invupdate():
#     A_augmented
#     assert_almost_equal(inv_A_augmented,np.linalg.inv(A_augmented))


