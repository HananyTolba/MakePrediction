from gpasdnn.kernels import *
from gpasdnn.gp import GaussianProcessRegressor as GPR

class QuasiGPR():
    '''
    
    '''

    def __init__(self,xtrain,ytrain,kernel = RBF(),modelList = None):
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._kernel = kernel
        self._modelList = modelList 
        
    def fit(self):
        xtrain = self._xtrain
        ytrain = self._ytrain
        kernel_expr = self._kernel
        kernel_names = kernel_expr.recursive_str_list()
        #print(kernel_names)
        #yfit = 0
        #ypred = 0
        models = []
        for ker in kernel_names:
            model = GPR(xtrain,ytrain)
            model.kernel_choice = ker
            model.fit()
            #print(model)
            models.append(model)
            yf,_ = model.predict()
            ytrain = ytrain - yf
            #plt.plot(ytrain)
            #model._ytrain = ytrain
                #yfit = yfit + yf
            self._modelList = models
        #return models
    def predict(self,xs=None,yt=None,components=False):
        models = self._modelList
        PRED = list(map(lambda mdl : mdl.predict(xs,yt),models))
        PRED_mat = np.array(PRED)
        if components:
            X = PRED_mat[:,0,:].reshape(PRED_mat.shape[0],PRED_mat.shape[2]).T

        ypred_, std_ = np.sum(PRED_mat,axis=0).tolist()
        ypred_, std_ = np.array(ypred_),np.array(std_)
        if components:
            return ypred_, std_, X
        else:
            return ypred_, std_
