from makeprediction.gp import date2num
import numpy as np 
def FitSplit(self):
    pp = []
    x,y = self._xtrain, self._ytrain

    x = date2num(x)

    hyp = self.p_fit(x,y)

    L = list()
    m = 100
    iteration = 0

    r_list = [int((i+1)/m*x.size) for i in range(m) if int((i+1)/m*x.size)>=100]
    #r_list = [r for r in r_list if r>=100 ]
    L = list(map(lambda s:self.p_fit(np.linspace(-1,1,s),y[:s])[-1]*s/x.size,r_list))


    # for i in range(m):
    #     r = int((i+1)/m*x.size)
    #     if r>=100:
    #         L.append(self.p_fit(np.linspace(-1,1,r),y[:r])[-1]*r/x.size)
    #         iteration= iteration + 1

    periodEst = L[np.argmin(np.abs(np.diff(L)))]
    print("number of iteration is : ",iteration)



    hyp[-1] = round(periodEst,3)

    

    hyp_dict = dict(zip(["length_scale","period"],hyp))
    hyp_dict["variance"] = y.std()**2
    self.set_hyperparameters(hyp_dict)
    