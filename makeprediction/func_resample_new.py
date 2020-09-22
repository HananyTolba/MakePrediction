def periodicFitByResampling(self):
        y = self._ytrain
        ystd = y.std()
        y = (y - y.mean()) / y.std()



        n = y.size
        p=1
        parms = []
        parms_end = []
        if (n<350):
            print("The 'Split' method was automatically chosen because the data size is very small for the 'Resampling' method.")

            return self.periodicFitBySplit()
            
            
        while int(n*p)>=SMALL_SIZE:
            m = n*p
            yre12 = scipy.signal.resample(y[:int(m)],SMALL_SIZE )
            yre12_end = scipy.signal.resample(y[-int(m):],SMALL_SIZE )

            p_est_12 = get_parms_from_api(yre12,"periodic")
            p_est_12_end = get_parms_from_api(yre12_end,"periodic")


            #p_est_12 = newModel.predict(yre12.reshape(1,-1)).ravel()
            p_est_12[-1] = p_est_12[-1]*int(m)/y.size
            p_est_12_end[-1] =   p_est_12_end[-1]*int(m)/y.size
            #print("parms resample: ",p_est_12[-1])
            #print("parms resample: ",p_est_12_end[-1])


            #noise_std = model_periodic_noise.predict(yre12.reshape(1,yre12.size))
            noise_std = get_parms_from_api(yre12,"iid_periodic_300")

            noise_std = noise_std.ravel()
            self._sigma_n = noise_std[0]*ystd


            if ((p==1)&(p_est_12[-1]>.95)):
                hyp = p_est_12
                hyp_dict = dict(zip(["length_scale","period"],hyp))
                hyp_dict["variance"] = ystd**2
                self.set_hyperparameters(hyp_dict)
                break


                
            else:
                p = p - .01
                if (p<=0):
                    break
                parms.append(p_est_12)
                parms_end.append(p_est_12_end)

        if len(parms)>1:
            parms = np.array(parms)
            parms_end = np.array(parms_end)

            plt.plot(parms[:,1],'b')
            plt.plot(parms_end[:,1],'r')

            plt.show()
            L = parms[:,1]
            periodEst = L[np.argmin(np.abs(np.diff(L)))]


            
            List = np.round(parms[:,1],3).tolist()
            period_est___ = most_frequent(List)
            #print("Estimated period is: ",period_est___)
            
            Est = parms[0,:]
            #Est[-1] = periodEst
            Est[-1] = period_est___
            hyp = Est
            hyp_dict = dict(zip(["length_scale","period"],hyp))
            hyp_dict["variance"] = ystd**2
            self.set_hyperparameters(hyp_dict)

            
            
            


    
