Welcome to gpasdnn's documentation!
===================================

To install gpasdnn:
-------------------

    Gaussian Process as deep Neural-Network gpasdnn can be installed by cloning the repository and running in the root folder:
    1) pip install . 
    This also installs required dependencies including TensorFlow, and sets everything up. Or with:
    2) pip install gpasdnn 






    Gaussian process regression (GPR):
    =====================================
    This package use a tensorflow pretrained model to estimate the Hyperparameters of 
    a GPR model and then fitting the data with.

    The advantages of Gaussian processes are:
        1) The prediction interpolates the observations.
        2) The prediction is probabilistic (Gaussian) so that one can compute empirical confidence intervals and decide based on those if one should refit (online fitting, adaptive fitting) the prediction in some region of interest.
        3) Versatile: different kernels can be specified. Common kernels are provided, but it is also possible to specify custom kernels.

    In addition to standard scikit-learn estimator API,
       1) The methods proposed here are much faster than standard scikit-learn estimator API.
       2) The prediction method here "predict" is very complete compared to scikit-learn estimator API with many options such as:
         "sparse" and the automatic online update of prediction.