#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
Fast and easy gaussian process regression using deep learning with tensorflow


"""




from setuptools import setup, find_packages

# import pkg_resources  # part of setuptools
#version = pkg_resources.require("gpasdlm")[0].version
version = "0.0.1"
#import gprbytf

setup(
    name='gpasdlm',         # How you named your package folder (MyLib)
    packages=['gpasdlm'],
    # packages=find_packages(where='src'),
    # package_dir='gprbytf',   # Chose the same as "name"
    # version = gprbytf.__version__, #"0.0.1",      # Start with a small
    # number and increase it with every change you make
    version=version,      # Start with a small number and increase it with every change you make

    # Chose a license from here:
    # https://help.github.com/articles/licensing-a-repository
    license="Apache License 2.0",
    description="Fast and easy gaussian process regression using deep learning with tensorflow",
    author='TOLBA Hannay', 
    long_description = __doc__,                 # Type in your name
    author_email="hanany100@gmail.com",
    # Provide either the link to your github or to your website
    url='https://github.com/hananytolba',
    download_url='https://github.com/hananytolba',    # I explain this later on
    keywords=[
        'Gaussian Process Regression',
        'Time series prediction',
        'Tensorflow',
        'Deep Learning'],
    platforms= [],
    # Keywords that define your package best
    install_requires=[            # I get to this in a second
        'numpy',
        "colorama",
        "tqdm",
        "termcolor",
        "scipy",
        "keras",
        "scikit_learn",
        "tensorflow",
        "matplotlib",
        "pandas",
        # only for the demo
    ],
    #package_data={'gpasdlm': ['keras_600/*']},
    python_requires=">=3.6, <3.8",
    include_package_data=True,

    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as
        # the current state of your package
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License  2.0',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
