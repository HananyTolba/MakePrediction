#!/usr/bin/env python
# -*- coding: utf-8 -*-


#from __future__ import absolute_import
import numpy as np
#import re


#def func_(s): return ' '.join(re.sub(r"([A-Z])", r" \1", s).split())


# __author__ = "Hanany Tolba"
# __copyright__ = "Copyright 2020, Guassian Process by Deep Learning Project"
# __credits__ = ["Hanany Tolba"]
# __license__ = "Apache License 2.0"
# __version__ = "0.0.1"
# __maintainer__ = "Hanany Tolba"
# __email__ = "hanany100@gmail.com"
# __status__ = "Production"

__all__ = ["RBF",
           "Matern12",
           "Matern32",
           "Matern52",
           "Periodic",
           "Linear",
           "Polynomial",
          


           
                
]

class Kernel:
    '''
    La classe 'Kernel' modèlisant le kernel d'un Gaussian process.
    et qui possédes plusieurs attribues et une seule méthode  "count"
    qui permet de calculer le kernel entre  deux vecteurs de localisations x et y comme suit
    : self.count(x,y) '''

    def __init__(self, length_scale=1,variance = 1,hyperparameter_number=2):
        '''
        Le constructeur ne contient par defaut qu'un seul attribut
        le length_scale qui modèlise la correlation
        '''
        self._variance = variance
        self._length_scale = length_scale
        self._hyperparameter_number = hyperparameter_number



        

    #def __repr__(self):
    #    return "Instance of class '{}'".format(self.__class__.__name__)

    def __str__(self):
        #if hyperparameters == True:
        return "({}: length_scale = {}, variance = {})".format(self.__class__.__name__,\
         self._length_scale,self._variance)
        #else:
        #return self.__class__.__name__


    #def __str__(self):
    #    return self.__class__.__name__


    def get_length_scale(self):
        return self._length_scale

    # @get_length_scale.setter
    def set_length_scale(self, length_scale):
        self._length_scale = length_scale

    def radial_dist(self, x, y=None):
        x = x.ravel()

        if y is None:
            y = x
        y = y.ravel()

        r = np.abs(x - y.reshape(-1, 1))
        return r

    def get_hyperparameters(self):
        d = self.__dict__
        parms = dict()
        for cle,valeur in d.items():
            if cle != "_hyperparameter_number":
                #print(cle.lstrip('_') + " = ", valeur)
                parms[cle.lstrip('_')] = valeur
        return parms

        ##return getattr(self._kernel)

    def set_hyperparameters(self,dic):
        for cle in self.__dict__.keys():
            if cle != "_hyperparameter_number":
                setattr(self, cle, dic[cle.lstrip('_')])


    #def label(self):
    #    r = self.__class__.__name__
    #    return r

    def label(self):
        if self.__class__.__name__ == "KernelProduct":
            r = self.recursive_kernel1()
            r = list(map(lambda x: x.__class__.__name__,r))
            r = " x ".join(r)
        elif self.__class__.__name__ == "KernelSum":
            r = self.recursive_kernel1()
            r = list(map(lambda x: x.__class__.__name__,r))
            r = " + ".join(r)
           
        else:
            r = self.__class__.__name__
        

        return r

    


    def __repr__(self):
        return "Instance of class '{}'".format(self.label())


    def __add__(self, other):
        return KernelSum(self, other)

    def __mul__(self, other):
        return KernelProduct(self, other)
        
    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return KernelProduct(Constant(b), self)
        return KernelProduct(b, self)
    
    def __radd__(self, b):
        if not isinstance(b, Kernel):
            if b == 0:
                return self
            else:
                return KernelSum(Constant(b), self)
        return KernelSum(b, self)
        






    def recursive_kernel1(self):
        kernel_list = []
        signs = []
        for v in self.__dict__.values():
            if v.__class__.__name__ == "KernelProduct":
                kernel_list.extend(v.recursive_kernel1())

            elif v.__class__.__name__ in ["int","float"]:
                pass
            elif v.__class__.__name__ == "KernelSum":
                kernel_list.extend(v.recursive_kernel1())
            
            else:
                kernel_list.append(v)
            
        return kernel_list

    def label_(self):
        f = self.label()
        return f.split("+")

    # def label1(self):
    #     if self.__class__.__name__ == "KernelProduct":
    #         r = self.recursive_kernel1()
    #         r = list(map(lambda x: x.__class__.__name__,r))
    #         r = " x ".join(r)
    #     elif self.__class__.__name__ == "KernelSum":
    #         r = self.recursive_kernel1()
    #         r = list(map(lambda x: x.__class__.__name__,r))
    #         r = " + ".join(r)

    #     else:
    #         r = self.__class__.__name__
    #     return r

    #@staticmethod
    def square_root_matrix(self,K):

        if self.__class__.__name__ == "Constant":
            Q = np.sqrt(K)
        else:
            np.fill_diagonal(K, K.diagonal() + 1e-10)

            try:
                Q = np.linalg.cholesky(K)
            except BaseException:
                U, s, VT = np.linalg.svd(K)
                Q = U@np.diag(np.sqrt(s))
        return Q


    

    def simulate(self, x, y=None,seed=None):
        '''
         This method allows the simulation of a Gaussian process (1d) on a domain x.
        '''
        #x = self.x_type(x)
        if y is None:
            y = x
        

        x = x.ravel()
        m = x.size

        K = self.count(x)

        Q = self.square_root_matrix(K)


        if seed is None:
            iid = np.random.randn(m)
        else:
            seed = seed
            iid = np.random.randn(m)


        y = Q @ iid
        return y 



   



    def simulate_2d(self,x1,x2=None,hyperparameters=None, seed=None):
        '''
         This method allows the simulation of a Gaussian process (1d) on a domain x.
        '''
        
        x1 = x1.ravel()
        m1 = x1.size




        
        if hyperparameters is None:
            
            if x2 is None:
                K1 = self.count(x1)
                Q1 = self.square_root_matrix(K1)
                iid = np.random.randn(m1,m1)
                y = Q1 @ iid @ Q1.T
            else:
                x2 = x2.ravel()
                m2 = x2.size
                K1 = self.count(x1)
                Q1 = self.square_root_matrix(K1)
                K2 = self.count(x2)
                Q2 = self.square_root_matrix(K2)
                iid = np.random.randn(m1,m2)
                y = Q1 @ iid @ Q2.T
        # elif isinstance(hyperparameters,dict):
        #     if len(hyperparameters)==1:
        #         hyperparameters = [hyperparameters]
        #     else:
        #         raise TypeError("In the 'simulate_2d' method, the hyperparameters must be a list or dict of size one.")





        elif isinstance(hyperparameters,list):


            if len(hyperparameters)==1:
                #if isinstance(hyperparameters,dict):
                #    self.set_hyperparameters(hyperparameters)
                #else:
                #    self.set_hyperparameters(hyperparameters[0])

                self.set_hyperparameters(hyperparameters[0])

                if x2 is None:
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    iid = np.random.randn(m1,m1)
                    y = Q1 @ iid @ Q1.T
                else:
                    x2 = x2.ravel()
                    m2 = x2.size
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    K2 = self.count(x2)
                    Q2 = self.square_root_matrix(K2)
                    iid = np.random.randn(m1,m2)
                    y = Q1 @ iid @ Q2.T
            elif len(hyperparameters)==2:
                self.set_hyperparameters(hyperparameters[0])
                if x2 is None:
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    self.set_hyperparameters(hyperparameters[1])
                    K2 = self.count(x1)
                    Q2 = self.square_root_matrix(K2)
                    iid = np.random.randn(m1,m1)
                    y = Q1 @ iid @ Q2.T
                else:
                    
                    x2 = x2.ravel()
                    m2 = x2.size
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    self.set_hyperparameters(hyperparameters[1])
                    K2 = self.count(x2)
                    Q2 = self.square_root_matrix(K2)
                    iid = np.random.randn(m1,m2)
                    y = Q1 @ iid @ Q2.T
            else:
                raise ValueError("In the 'simulate_2d' method, the hyperparameters must be a list of maximum size of 2.")


        else:
            raise TypeError("In the 'simulate_2d' method, the hyperparameters must be a list.")
            
        return y 



    def simulate_3d(self,x1,x2=None,x3=None,hyperparameters=None):
        '''
         This method allows the simulation of a Gaussian process (1d) on a domain x.
        '''
        
        x1 = x1.ravel()
        m1 = x1.size


        
        if hyperparameters is None:
            
            if ((x2 is None)&(x3 is None)):
                K1 = self.count(x1)
                Q1 = self.square_root_matrix(K1)
                iid = np.random.randn(m1**2,m1)
                y = np.kron(Q1,Q1) @ iid @ Q1.T
                m2=m3=m1
            elif x3 is None:
                x2 = x2.ravel()
                m2 = x2.size
                K1 = self.count(x1)
                Q1 = self.square_root_matrix(K1)
                K2 = self.count(x2)
                Q2 = self.square_root_matrix(K2)
                iid = np.random.randn(m1*m2,m1)
                y = np.kron(Q1,Q2) @ iid @ Q1.T
                m3=m1

            elif x2 is None:
                x3 = x3.ravel()
                m3 = x3.size
                K1 = self.count(x1)
                Q1 = self.square_root_matrix(K1)
                K3 = self.count(x3)
                Q3 = self.square_root_matrix(K3)
                iid = np.random.randn(m1**2,m3)
                y = np.kron(Q1,Q1) @ iid @ Q3.T
                m2=m1

            else:
                x2 = x2.ravel()
                m2 = x2.size
                x3 = x3.ravel()
                m3 = x3.size
                K1 = self.count(x1)
                Q1 = self.square_root_matrix(K1)
                K2 = self.count(x2)
                Q2 = self.square_root_matrix(K2)
                K3 = self.count(x3)
                Q3 = self.square_root_matrix(K3)
                iid = np.random.randn(m1*m2,m3)
                y = np.kron(Q1,Q2) @ iid @ Q3.T






        elif isinstance(hyperparameters,list):


            if len(hyperparameters)==1:
               
                self.set_hyperparameters(hyperparameters[0])
                if ((x2 is None)&(x3 is None)):
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    iid = np.random.randn(m1**2,m1)
                    y = np.kron(Q1,Q1) @ iid @ Q1.T
                    m2=m3=m1
                elif x3 is None:
                    x2 = x2.ravel()
                    m2 = x2.size
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    K2 = self.count(x2)
                    Q2 = self.square_root_matrix(K2)
                    iid = np.random.randn(m1*m2,m1)
                    y = np.kron(Q1,Q2) @ iid @ Q1.T
                    m3=m1
                elif x2 is None:
                    x3 = x3.ravel()
                    m3 = x3.size
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    K3 = self.count(x3)
                    Q3 = self.square_root_matrix(K3)
                    iid = np.random.randn(m1**2,m3)
                    y = np.kron(Q1,Q1) @ iid @ Q3.T
                    m2=m1
                else:
                    x2 = x2.ravel()
                    m2 = x2.size
                    x3 = x3.ravel()
                    m3 = x3.size
                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    K2 = self.count(x2)
                    Q2 = self.square_root_matrix(K2)
                    K3 = self.count(x3)
                    Q3 = self.square_root_matrix(K3)
                    iid = np.random.randn(m1*m2,m3)
                    y = np.kron(Q1,Q2) @ iid @ Q3.T


            elif len(hyperparameters)==3:
                if ((x2 is None)&(x3 is None)):
                    self.set_hyperparameters(hyperparameters[0])

                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    self.set_hyperparameters(hyperparameters[1])

                    K2 = self.count(x1)
                    Q2 = self.square_root_matrix(K2)
                    self.set_hyperparameters(hyperparameters[2])

                    K3 = self.count(x1)
                    Q3 = self.square_root_matrix(K3)

                    iid = np.random.randn(m1**2,m1)
                    y = np.kron(Q1,Q2) @ iid @ Q3.T
                    m2=m3=m1
                elif x3 is None:
                    self.set_hyperparameters(hyperparameters[0])

                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    self.set_hyperparameters(hyperparameters[1])
                    x2 = x2.ravel()
                    m2 = x2.size
                    K2 = self.count(x2)
                    Q2 = self.square_root_matrix(K2)
                    self.set_hyperparameters(hyperparameters[2])

                    K3 = self.count(x1)
                    Q3 = self.square_root_matrix(K3)

                    iid = np.random.randn(m1*m2,m1)
                    y = np.kron(Q1,Q2) @ iid @ Q3.T
                    m3=m1
                elif x2 is None:
                    self.set_hyperparameters(hyperparameters[0])

                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    self.set_hyperparameters(hyperparameters[1])
                    
                    K2 = self.count(x1)
                    Q2 = self.square_root_matrix(K2)
                    self.set_hyperparameters(hyperparameters[2])
                    x3 = x3.ravel()
                    m3 = x3.size

                    K3 = self.count(x3)
                    Q3 = self.square_root_matrix(K3)

                    iid = np.random.randn(m1**2,m3)
                    y = np.kron(Q1,Q2) @ iid @ Q3.T
                    m2=m1
                else:
                    self.set_hyperparameters(hyperparameters[0])

                    K1 = self.count(x1)
                    Q1 = self.square_root_matrix(K1)
                    self.set_hyperparameters(hyperparameters[1])
                    x2 = x2.ravel()
                    m2 = x2.size
                    K2 = self.count(x2)
                    Q2 = self.square_root_matrix(K2)
                    self.set_hyperparameters(hyperparameters[2])
                    x3 = x3.ravel()
                    m3 = x3.size

                    K3 = self.count(x3)
                    Q3 = self.square_root_matrix(K3)

                    iid = np.random.randn(m1*m2,m3)
                    y = np.kron(Q1,Q2) @ iid @ Q3.T
            else:
                raise ValueError("In the 'simulate_2d' method, the hyperparameters must be a list of  size 1 or 3.")


        else:
            raise TypeError("In the 'simulate_2d' method, the hyperparameters must be a list.")
            
        return y.reshape(m1,m2,m3)



    @staticmethod
    def sigmoid_(x,s,x0):
        return .5*(1+np.tanh((x-x0)/s))
class Constant(Kernel):

    # def __init__(self, length_scale=1.0,hyperparameter_number=1,variance=1):
    #     self._length_scale = length_scale
    #     self._hyperparameter_number = hyperparameter_number
    #     self._variance = variance



    def count(self, x, y=None):
        """Squared Exponential covariance function or RBF with isotropic distance measure."""
        x = x.ravel()
        if y is None:
            y = x

        kernel = np.full((x.size,y.size), self._length_scale)
        return self._variance*kernel



class RBF(Kernel):

    def __init__(self, length_scale=1.0,hyperparameter_number=2,variance=1.):
        self._length_scale = length_scale
        self._hyperparameter_number = hyperparameter_number
        self._variance = variance


    def count(self, x, y=None):
        """Squared Exponential covariance function or RBF with isotropic distance measure."""
        kernel = np.exp(-self.radial_dist(x, y)**2 /
                        (2 * self._length_scale**2))
        return self._variance*kernel


class Linear(Kernel):

    def __init__(self,hyperparameter_number=1,variance=1.):
        self._hyperparameter_number = hyperparameter_number
        self._variance = variance

    def __str__(self):
        #parent = super().__str__()

        #if hyperparameters == True:
        return "{}:(variance = {})".format(
            self.__class__.__name__,self._variance)
        #else:
        #    return self.__class__.__name__



    def count(self, x, y=None):
        if y is None:
            y = x
        x = x.reshape(x.size, 1)
        y = y.reshape(y.size, 1)
        #r = np.dot(x - self._length_scale, y.T - self._length_scale).T
        #kernel = (self._variance*x@y.T + self._length_scale**2).T
        kernel = (x@y.T).T


        return self._variance*kernel

class Polynomial(Kernel):


    def __init__(self,length_scale=0,hyperparameter_number=2,variance=1.):
        self._hyperparameter_number = hyperparameter_number
        self._variance = variance
        self._length_scale = length_scale

    def __str__(self):
        #parent = super().__str__()

        #if hyperparameters == True:
        return "{}:(variance = {})".format(
            self.__class__.__name__,self._variance)
        #else:
        #    return self.__class__.__name__

    def count(self, x, y=None):
        if y is None:
            y = x
        x = x.reshape(x.size, 1)
        y = y.reshape(y.size, 1)
        kernel = ((x@y.T + self._length_scale**2)**3).T
        return self._variance*kernel


# class Exponential(Kernel):

#     def count(self, x, y=None):
#         """Exponential covariance function. """
#         kernel = np.exp(-0.5 * self.radial_dist(x, y) / self._length_scale)
#         return kernel


class Matern12(Kernel):

    def count(self, x, y=None):
        r = self.radial_dist(x, y)
        """Matern covariance function with nu = 1/2 and isotropic distance measure. """
        kernel = np.exp(-r / self._length_scale)
        return self._variance*kernel


class Matern32(Kernel):

    def count(self, x, y=None):
        if y is None:
            y = np.copy(x)
            
        r = self.radial_dist(x, y)
        """Matern covariance function with nu = 3/2 and isotropic distance measure. """
        kernel = (1 + np.sqrt(3) * r) * \
            np.exp(-np.sqrt(3) * r / self._length_scale)
        return self._variance*kernel


class Matern52(Kernel):

    def count(self, x, y=None):
        """Matern covariance function with nu = 5/2 and isotropic distance measure."""
        

        r = self.radial_dist(x, y)

        kernel = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * \
            np.exp(-np.sqrt(5) * r / self._length_scale)

        return self._variance*kernel


# class Cosine(Kernel):

#     def count(self, x, y=None):
#         """Stationary covariance function for a sinusoid."""
#         r = self.radial_dist(x, y)

#         kernel = np.cos(np.pi * r / self._length_scale)
#         return kernel


class Periodic(Kernel):

    def __init__(self, length_scale=1,period=1,hyperparameter_number=3,variance=1):

        # Kernel.__init__(self)
        super().__init__(length_scale)
        self._period = period
        self._hyperparameter_number = hyperparameter_number
        self._variance =  variance


    def get_period(self):
        return self._period

    def set_period(self, period):
        self._period = period

    def __str__(self):
        #parent = super().__str__()

        #if hyperparameters == True:
        return "({}: length_scale = {}, period = {}, variance = {})".format(
            self.__class__.__name__,self._length_scale,self._period,self._variance)
        #else:
        #    return self.__class__.__name__



    def count(self, x, y=None):
        """Stationary covariance function for a sinusoid."""
        r = self.radial_dist(x, y)
        kernel = np.exp(-2 * np.sin(np.pi * r / self._period)
                        ** 2 / (self._length_scale**2))
        return self._variance*kernel



# class ChangePointLinear(Linear):

#     def __init__(self, length_scale=1,length_scale1=.5,location=.5, steepness=.00001,hyperparameter_number=4):
#         self._length_scale = length_scale
#         self._length_scale1 = length_scale1
#         self._steepness = steepness
#         self._location = location
#         self._hyperparameter_number = hyperparameter_number

#     def count(self, x, y=None):
#         if y is None:
#             y = x
#         sig_x = self.sigmoid_(x,self._steepness,self._location).reshape(-1,1)
#         sig_y = self.sigmoid_(y,self._steepness,self._location).reshape(-1,1)
#         M0=(1 - sig_x)@(1 - sig_y).T
#         M1=sig_x@sig_y.T
#         C = Linear(self._length_scale).count(x,y)*M0.T +  Linear(self._length_scale1).count(x,y)*M1.T
#         return C

# class ChangePointRBF(RBF):

#     def __init__(self, length_scale=1,length_scale1=.01,location=.5, steepness=.0001,hyperparameter_number=4):
#         self._length_scale = length_scale
#         self._length_scale1 = length_scale1
#         self._steepness = steepness
#         self._location = location
#         self._hyperparameter_number = hyperparameter_number

#     def count(self, x, y=None):
#         if y is None:
#             y = x
#         sig_x = self.sigmoid_(x,self._steepness,self._location).reshape(-1,1)

#         sig_y = self.sigmoid_(y,self._steepness,self._location).reshape(-1,1)
#         M0=(1 - sig_x)@(1 - sig_y).T
#         M1=sig_x@sig_y.T
#         C = RBF(self._length_scale).count(x,y)*M0.T +  RBF(self._length_scale1).count(x,y)*M1.T
#         return C




class KernelSum(Kernel):
    """
    Represents sum of a pair of kernels
    """

    def __init__(self, kernel_1= RBF(), kernel_2=None,hyperparameter_number=4):
        self._kernel_1 = kernel_1
        if kernel_2 is None:
            kernel_2 = kernel_1
        self._kernel_2 = kernel_2
        if ((kernel_1 == RBF())&(kernel_2 == RBF())):
            self._hyperparameter_number = hyperparameter_number
        else :
            self._hyperparameter_number = self._kernel_1._hyperparameter_number + self._kernel_2._hyperparameter_number

    def count(self, data_1, data_2=None):
        return self._kernel_1.count(data_1, data_2) + \
            self._kernel_2.count(data_1, data_2)

    def label(self):
        if self.__class__.__name__ == "KernelSum":
            r = self.recursive_kernel()
            r = list(map(lambda x: x.__class__.__name__,r))
            r = " + ".join(r)
        else:
            r = self.__class__.__name__
        return r


    def __str__(self):
        return str(self._kernel_1) + ' + ' + str(self._kernel_2)

    def recursive_kernel(self):
        kernel_list = []
        for v in self.__dict__.values():
            if v.__class__.__name__ == "KernelSum":
                kernel_list.extend(v.recursive_kernel())
            elif v.__class__.__name__ in ["int","float"]:
                pass
            else:
                kernel_list.append(v)
            
        return kernel_list
    def get_hyperparameters(self):
        return list(map(lambda x:x.get_hyperparameters(),self.recursive_kernel()))

    def set_hyperparameters(self,D):
        l = self.recursive_kernel()
        for i in range(len(D)):
            l[i].set_hyperparameters(D[i])
        #return self.get_hyperparameters()




class KernelProduct(Kernel):
    """
    Represents product of a pair of kernels
    """

    def __init__(self, kernel_1=RBF(), kernel_2=None,hyperparameter_number=4):
        self._kernel_1 = kernel_1
        if kernel_2 is None:
            kernel_2 = kernel_1
        self._kernel_2 = kernel_2
        if ((kernel_1 == RBF())&(kernel_2 == RBF())):
            self._hyperparameter_number = hyperparameter_number
        else :
            self._hyperparameter_number = self._kernel_1._hyperparameter_number + self._kernel_2._hyperparameter_number


    def count(self, data_1, data_2=None):
        return self._kernel_1.count(data_1, data_2) * \
            self._kernel_2.count(data_1, data_2)

    def __str__(self):
        return str(self._kernel_1) + ' x ' + str(self._kernel_2)

    def recursive_kernel(self):
        kernel_list = []
        for v in self.__dict__.values():
            if v.__class__.__name__ == "KernelProduct":
                kernel_list.extend(v.recursive_kernel())
            elif v.__class__.__name__ in ["int","float"]:
                pass
            
            else:
                kernel_list.append(v)
            
        return kernel_list


    def label(self):
        if self.__class__.__name__ == "KernelProduct":
            r = self.recursive_kernel()
            r = list(map(lambda x: x.__class__.__name__,r))
            r = " x ".join(r)
        else:
            r = self.__class__.__name__
        return r


    def get_hyperparameters(self):
        return list(map(lambda x:x.get_hyperparameters(),self.recursive_kernel()))

    def set_hyperparameters(self,D):
        l = self.recursive_kernel()
        for i in range(len(D)):
            l[i].set_hyperparameters(D[i])
        #return self.get_hyperparameters()


if __name__ == "__main__":
    from gpbytf.kernels import *
    from gpbytf.gaussianprocess import GaussianProcessRegressor as GP
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    x = np.linspace(0,10,3000)

    # x1 = np.linspace(0,1,300)

    # for i in tqdm(range(1000)):
    #     y = Polynomial().simulate(x)
    # z = Polynomial().simulate(x)
    # #mdl = ChangePointLinear()
    # #mdl.count(x,x1)

    # plt.plot(x,y)
    # plt.plot(x,z)
    # plt.show()


    # x1 = np.linspace(-3,3,100)
    # x2 = np.linspace(0,6,110)

    # x3 = np.linspace(-1,1,40)

    # #y =RBF().simulate_2d(x1=x1,hyperparameters=[{"length_scale":.4},{"length_scale":.9}],x2=x2)
    # h=[{"length_scale":.4},{"length_scale":.7}]
    # h = [{"length_scale":.9}]

    # y =RBF().simulate_2d(x1=x1,hyperparameters=h)

    # ll=[{"length_scale":1},{"length_scale":.4},{"length_scale":.5}]
    # #ll = [{"length_scale":.9}]
    # y =RBF().simulate_3d(x1=x1,x2=x2,x3=x3,hyperparameters=ll)
    # print(y.shape)
    # #plt.pcolor(x2,x1,y,cmap='jet')
    # plt.imshow(y[:,:,20],cmap='jet')
    # #plt.imshow(y,cmap='jet')

    # plt.show()


    #rbf = RBF(length_scale=.123456789)
    #per = Periodic(length_scale=.123456789,period=.123456789)
    #print(rbf.label())
    #print(per)
    #np.random.seed(1)
    ks =    RBF(2.5) + Polynomial(variance=.1)
    d = [{"length_scale":np.random.rand(1),"variance" : np.random.rand(1)},{"length_scale":np.random.rand(1),"variance":np.random.rand(1)}]

    kernel =  RBF(.3)

    np.random.seed(0)

    #kernel.set_hyperparameters(d)



    #ks = RBF(1)
    #ks =   Linear(0) 
    #print(ks.__dict__ )
    #ks =  Constant(5) 
    y0 = kernel.simulate(x)

    y = y0 + .2*np.random.randn(x.size)
    train_size = int(x.size*.85)
    (xtrain,ytrain) = x[:train_size],y[:train_size]
    (xtest,ytest) = x[train_size:],y[train_size:]

    mdl = GP()
    #mdl.kernel_choice ="Matern52"
    mdl.fit_bootstrap(xtrain,ytrain)

    #yf,_ = mdl.predict(xtrain,ytrain)
    yp,std_pred= mdl.predict(xtrain,ytrain,x)

    #mdl.kernel_choice = "Polynomial"
    #mdl.fit_bootstrap(xtrain,ytrain)
    #print(mdl)
    # list_prob = []
    # list_prob = []
    # hyparms_list = []
    # std_list = []
    # for i in range(100):
    #     prob,df = mdl.expression(y,prob="df")
    #     #list_prob.append((df.index[0],df.Probability[0]))
    #     list_prob.append(df.index[0])
    #     mdl._fit(x,y)
    #     hyparms = mdl.get_hyperparameters()
    #     hyparms_list.append(hyparms)
    #     std_list.append(mdl.std_noise)

    # from collections import Counter
    # freq = Counter(list_prob)
    # import pandas as pd
    # print(freq)
    # hyparms_df = pd.DataFrame(hyparms_list)
    # plt.plot(hyparms_df.length_scale,label="ell")
    # plt.plot(hyparms_df.variance,label="sig")

    # plt.plot(std_list,label="std_noise")
    # plt.legend()
    # plt.show()

    # mdl.std_noise = sum(std_list)/100
    # #mdl.set_length_scale(hyparms_df.length_scale.values)
    # dd = {"length_scale":hyparms_df.length_scale.values.mean(),"variance":hyparms_df.variance.values.mean()}
    
    # mdl.set_hyperparameters(dd)
    # print(mdl)
    # print(dd)
    #yp,_=mdl.predict(xtrain,ytrain,x)

    # #print(parms_mean)

    # max_key = max(freq, key=freq.get)
    # print("Le modèle est : ", max_key)

    print(mdl.expression_bootstrap_predict(ytrain,probability=True))

    

    plt.plot(x,y,'k')
    plt.plot(x,y0,'b')
    plt.plot(x,yp,'r')
    plt.fill_between(x,yp - 1.96 * std_pred, yp + 1.96 * std_pred,facecolor='red',edgecolor='blue',alpha=0.3, label='95% confidence interval')

    plt.show()

    #print(ks.label())
    #print(ks.label_())

    #print(ks)
    #print(ks.get_hyperparameters())
    #print(ks.recursive_kernel1())
    #print(ks.label())

    #
    #isinstance(ks,Periodic)

    #y = ks.simulate(x)
    #plt.plot(x,y,'b')
    #plt.show()
    #print(ks.__dict__.values())
    #print(ks.__dict__.values())

    #print(list(map(lambda x: isinstance(x,KernelProduct),ks.__dict__.values())))

    # ss = list(filter(lambda x: not isinstance(x,int),ks.__dict__.values()))
    # ss1 = list(map(lambda x: isinstance(x,KernelSum),ss))

    # print(ss1)


    # ll = [{'length_scale': 0.6},{'length_scale': 1.3, 'period': 2},{'length_scale': 0.18}]
    #print(ks.get_hyperparameters())
    # ks.set_hyperparameters(ll)
    # print(ks.get_hyperparameters())
    #print(ks.label())

    #print(ks.__dict__.values())
    # print(ks._hyperparameter_number)

    # c = RBF(length_scale=1) + Periodic() + Linear()
    # print(c)

    