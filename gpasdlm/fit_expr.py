import importlib
import copy


def fit_expr(self,model_expr):
    #model_expr = "Linear  +  Periodic +    RBF"
    x,yn = self._xtrain,self._ytrain
    model_expr = model_expr.replace(" ", "")
    cmp = model_expr.split("+")
    print(cmp)
    module_ = importlib.import_module("gpasdlm.kernels")
    gprs = []
    data = yn
    for ker_name in cmp:
        class_ = getattr(module_, ker_name)
        instance = class_()
        #print(instance)
        gpr = GPR(x,data)
        gpr._kernel = instance
        gpr.fit()
        #print(gpr)
        newobj = copy.copy(gpr)
        gprs.append(newobj)

        yfit,_ = gpr.predict()
        data =  data - yfit
        gpr._ytrain = data 

    return gprs

def predict_expr(self,xs,gprs):
    list_pred_pstd =[mdl.predict(xs) for mdl in  gprs]
    yp,std = np.array(list_pred_pstd).sum(axis=0)
    return yp,std


