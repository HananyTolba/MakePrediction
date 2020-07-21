
<!-- ![alt text](http://url/to/img.png)
 -->


GPasDNN is a package for building Gaussian process models in Python, using TensorFlow (http://www.tensorflow.org).
It was originally created by [Hanany Tolba].
 
 *GPasDNN is an open source project. If you have relevant skills and are interested in contributing then please do contact us (hananytolba@yahoo.com).*

 Gaussian process regression (GPR):
=====================================
This implementation use a tensorflow pretrained model to estimate the Hyperparameters of a GPR model and then fitting the data with.
The advantages of Gaussian processes are:
   * The prediction interpolates the observations.
   * The prediction is probabilistic (Gaussian) so that one can compute empirical confidence intervals and decide based on those if one should refit (online fitting, adaptive fitting) the prediction in some region of interest.
   * Versatile: different kernels can be specified. Common kernels are provided, but it is also possible to specify custom kernels.

In addition to standard scikit-learn estimator API,
   * The methods proposed here are much faster than standard scikit-learn estimator API.
   * The prediction method here "predict" is very complete compared to scikit-learn estimator API with many options such as:
    "sparse" and the automatic online update of prediction.

   


## What does GPasDNN do?




### Latest release from PyPI

pip install gpasdnn



### Latest source from GitHub

*Be aware that the `master` branch may change regularly, and new commits may break your code.*

[GPasDNN GitHub repository](https://github.com/HananyTolba/Gaussian-process-as-deep-neural-network.git), run:
pip install  .

