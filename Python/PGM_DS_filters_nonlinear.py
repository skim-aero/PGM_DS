import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import statistics
import copy
import math

from sklearn.cluster import DBSCAN
from scipy.stats import chi2

class PGM_DS(object):
    def __init__(self):
        self.load_parameters()

    def load_parameters(self):
        # Set the parameters

        # Scenario
        self.test_scenario = "Nonlinear"

        self.n = 1 # dimention of the state
        self.nm = 1 # dimention of the measurement
        self.dt = 1
        self.tf = 52
        self.t = np.arange(0,self.tf+self.dt,self.dt)
        self.numStep = len(self.t)
        self.trueinitial = 0.0
        self.P0 = 2.0

        self.initial = self.trueinitial+np.sqrt(self.P0)*np.random.randn(self.n,1)      

        # PGM
        self.numParticles = 50

        self.Q = 10.0
        self.R = 1.0
        self.Rmat = self.R*np.eye(self.nm)

        self.particle = np.empty((self.n,self.numParticles,self.tf))
        self.particle_meas = np.empty((self.nm,self.numParticles))

        self.dbscan_minpts = self.n+1
        self.dbscan_epsilon = 10
        self.merging_thres = chi2.ppf(0.995, self.n)
        self.epsilon = 1e-8 # For covariance correction trick

        self.numMixture = 0
        self.particle_mean = []
        self.particle_var = []
        self.cweight = []

        self.estmean = np.zeros((self.n,self.tf))
        self.error = np.zeros((self.n,self.tf))
        self.rmse = 0

        self.with_ut = False # Boolean

        if self.with_ut:
            print("Using PGM-DU")
            self.ut_numsigma = 2*self.n+1
            self.ut_alpha = 1.3
            self.ut_beta = 1.5
            self.ut_kappa = 0
            self.ut_lambda = 0.2
            # self.ut_lambda = self.ut_alpha**2*(self.n+self.ut_kappa)-self.n

            self.wm = np.empty((self.ut_numsigma,1))
            self.wc = np.empty((self.ut_numsigma,1))

            self.wm[0,:] = self.ut_lambda/(self.n+self.ut_lambda)
            self.wc[0,:] = self.ut_lambda/(self.n+self.ut_lambda)+(1-self.ut_alpha**2+self.ut_beta)

            for j in range(1, self.ut_numsigma):
                self.wm[j,:] = 1/(2*(self.n+self.ut_lambda))
                self.wc[j,:] = self.wm[j,:]
        else:
            print("Using PGM-DS")

    def funsys(self, x, w, k):
        propagated = 0.5*x+25*x/(1+x**2)+8*math.cos(1.2*k)+w

        return propagated
    
    def funmeas(self, x, v):
        transformed = (np.diag(np.outer(x,x)).T)/20+v

        return transformed

    def PGM_DS_clustering(self, label, t):
        clster = DBSCAN(eps=self.dbscan_epsilon, min_samples=self.dbscan_minpts).fit(self.particle[:,:,t].T)
        label = clster.labels_

        temp_dbscan_epsilon = self.dbscan_epsilon + 0.5
        while all(x == -1 for x in label):
            clster = DBSCAN(eps=temp_dbscan_epsilon , min_samples=self.dbscan_minpts).fit(self.particle[:,:,t].T)
            label = clster.labels_

            temp_dbscan_epsilon += 0.5

        leng_old = 0

        idx = np.argsort(label)
        self.particle[:,:,t] = self.particle[:,idx,t]

        self.numMixture = max(label)+1

        self.particle_mean = np.zeros((self.n,self.numMixture))
        self.particle_var = np.zeros((self.n,self.n,self.numMixture))
        self.cweight = np.zeros((self.numMixture,1))

        for clst in range(self.numMixture):
            leng = np.sum(label == clst)
            self.cweight[clst] = leng/self.numParticles

            for i in range(self.n):
                self.particle_mean[i,clst] = statistics.fmean(self.particle[i,leng_old:leng_old+leng,t])
            
            self.particle_var[:,:,clst] = np.var(self.particle[i,leng_old:leng_old+leng,t])

            leng_old += leng

        self.cweight /= sum(self.cweight)

        return label

    def PGM_DS_merging(self, label, t):
        notdone = True

        while notdone:
            idx_mer = []
            for clst1 in range(self.numMixture-1):
                if clst1 not in idx_mer:
                    for clst2 in range(clst1+1,self.numMixture):
                        temp1 = self.particle_mean[:,clst1]-self.particle_mean[:,clst2]
                        temp2 = np.linalg.inv((self.particle_var[:,:,clst1]+self.particle_var[:,:,clst2,])/2+self.epsilon*np.eye(self.n)) # Add small jitter

                        distance = np.sqrt(np.dot(np.dot(temp1, temp2), temp1))

                        if distance < self.merging_thres:
                            idx_mer.append(clst2)
                            label[label == clst2] = clst1

            if not idx_mer:
                notdone = False
            else:
                unique_values = np.unique(label)
                new_label = np.zeros(len(label))

                cnt = 0
                for i in range(len(unique_values)):
                    if unique_values[i] != -1:
                        indices = label == unique_values[i]
                        if sum(indices) == 1:
                            new_label[indices] = -1
                        else:
                            new_label[indices] = cnt
                            cnt += 1

                new_label[label == -1] = -1     
                label = new_label

                leng_old = 0

                idx = np.argsort(label)
                self.particle[:,:,t] = self.particle[:,idx,t]

                self.numMixture = np.sum(unique_values != -1)
                self.cweight = np.zeros((self.numMixture,1))

                for clst in range(self.numMixture):
                    leng = np.sum(label == clst)
                    self.cweight[clst] = leng/self.numParticles

                    for i in range(self.n):
                        self.particle_mean[i,clst] = statistics.fmean(self.particle[i,leng_old:leng_old+leng,t])
                    
                    self.particle_var[:,:,clst] = np.var(self.particle[i,leng_old:leng_old+leng,t])

                    leng_old += leng

                distance = 0

        self.particle_mean = self.particle_mean[0:self.numMixture,:]
        self.particle_var = self.particle_var[0:self.numMixture,:,:]
        self.cweight /= sum(self.cweight)

        # Covariance correction trick to avoid numerical instability
        for clst in range(self.numMixture):
            self.particle_var[:,:,clst] = (self.particle_var[:,:,clst]+self.particle_var[:,:,clst].T)/2

            if np.any(np.isnan(self.particle_var[:,:,clst])) or np.any(np.isinf(self.particle_var[:,:,clst])):
                self.particle_var[:,:,clst] = self.epsilon*np.eye(self.n)
            elif np.any(np.linalg.eigvals(self.particle_var[:,:,clst]) <= self.epsilon):
                while np.any(np.linalg.eigvals(self.particle_var[:,:,clst]) <= self.epsilon):
                    self.particle_var[:,:,clst] += self.epsilon*np.eye(self.n)

        return label

    def PGM_DS_update(self, label, measurement, t):
        leng_old = 0

        likelih = np.zeros(self.numMixture)

        if self.with_ut:
            xs = np.zeros((self.n,self.ut_numsigma))
            zs = np.zeros((self.nm,self.ut_numsigma))

            for clst in range(self.numMixture):
                xn = np.zeros((self.n,1))
                zn = np.zeros((self.nm,1))
                Pn = np.zeros((self.n,self.n))
                Pxz = np.zeros((self.n,self.nm))
                Pzz = np.zeros((self.nm,self.nm))

                sqrt_P = np.linalg.cholesky((self.n+self.ut_lambda)*self.particle_var[:,:,clst]) # Square root of scaled covariance
                xs[:,0] = self.particle_mean[:,clst]

                for j in range(self.n):
                    xs[:,j+1] = self.particle_mean[:,clst] + sqrt_P[:,j]
                    xs[:,j+self.n+1] = self.particle_mean[:,clst] - sqrt_P[:,j]

                for j in range(self.ut_numsigma):
                    xn[:,:] += self.wm[j]*xs[:,j] 
        
                for j in range(self.ut_numsigma):
                    Pn[:,:] += self.wc[j]*np.outer((xs[:,j]-xn[:,:]),(xs[:,j]-xn[:,:]))
                
                # Measurements update
                for j in range(self.ut_numsigma):
                    zs[:,j] = self.funmeas(xs[:,j],0)
                    zn[:,:] += self.wm[j]*zs[:,j]

                for j in range(self.ut_numsigma):
                    Pxz[:,:] += self.wc[j]*np.outer((xs[:,j]-xn[:,:]),(zs[:,j]-zn[:,:]))
                    Pzz[:,:] += self.wc[j]*np.outer((zs[:,j]-zn[:,:]),(zs[:,j]-zn[:,:]))
                Pzz += self.R + 1e-6*np.eye(self.nm)
        
                K_temp = Pxz/Pzz
                K = K_temp.flatten()
                
                residual = measurement-zn

                self.particle_mean[:,clst] = xn + K*residual
                self.particle_var[:,:,clst]= Pn - K*Pzz*K[:,None]

                likelih[clst] = self.cweight[clst]*(1/np.sqrt(2*self.R))*np.exp(-0.5*((residual)/self.Rmat)**2)
        else:
            for p in range(self.numParticles):
                self.particle_meas[:,p] = self.funmeas(self.particle[:,p,t],np.sqrt(self.R)*np.random.randn(self.nm,1))

            for clst in range(self.numMixture):
                leng = np.sum(label == clst)

                # Temporary test for weighted expected measurement mean
                temp_w = np.array((1,leng))
                temp_w = scipy.stats.norm(self.particle_meas[:,leng_old:leng_old+leng],self.R).pdf([measurement])

                # avoid round off to zero error
                temp_w += 1e-300
                temp_w /= np.sum(temp_w)  

                particle_exp = np.sum(self.particle_meas[:,leng_old:leng_old+leng]*temp_w)
                # particle_exp = np.mean(self.particle_meas[:,leng_old:leng_old+leng])

                temp_meas = np.zeros(leng)
                temp_mean = np.zeros((3,leng))

                temp_meas = self.particle_meas[:,leng_old:leng_old+leng] - particle_exp
                temp_mean = (self.particle[:,leng_old:leng_old+leng,t] - self.particle_mean[:,clst])

                Pxz = np.dot(temp_mean,temp_meas.T)/(leng-1)
                Pzz = np.dot(temp_meas,temp_meas.T) + self.R + 1e-6*np.eye(self.nm)
                
                K_temp = Pxz/Pzz
                K = K_temp.flatten()

                residual= measurement-particle_exp

                self.particle_mean[:,clst] += K*residual
                self.particle_var[:,:,clst] -= K*Pzz*K[:,None]

                likelih[clst] = self.cweight[clst]*(1/np.sqrt(2*self.R))*np.exp(-0.5*((residual)/self.Rmat)**2)

                leng_old += leng
            
        for clst in range(self.numMixture):
            self.cweight[clst] = likelih[clst]/sum(likelih)

        # avoid round off to zero error
        self.cweight += 1e-300
        self.cweight /= sum(self.cweight)
        self.cweight = self.cweight.flatten()

        # Covariance correction trick to avoid numerical instability
        for clst in range(self.numMixture):
            self.particle_var[:,:,clst] = (self.particle_var[:,:,clst]+self.particle_var[:,:,clst].T)/2
            
            if np.any(np.linalg.eigvals(self.particle_var[:,:,clst]) <= self.epsilon):
                while np.any(np.linalg.eigvals(self.particle_var[:,:,clst]) <= self.epsilon):
                    self.particle_var[:,:,clst] += self.epsilon*np.eye(self.nm)

        for clst in range(self.numMixture):
            self.estmean[:,t] += self.cweight[clst]*self.particle_mean[:,clst]

    def PGM_DS_resampling(self, t):
        mixture_idx = np.random.choice(len(self.cweight), size=self.numMixture, replace=True, p=self.cweight)

        new_particle = np.zeros((self.n,self.numParticles))
        for i, idx in enumerate(mixture_idx):
            new_particle[:,i] = scipy.stats.multivariate_normal.rvs(self.particle_mean[:,idx],self.particle_var[:,:,idx])

        self.particle[:,:,t] = copy.deepcopy(new_particle)

    def plot_graph(self, true):
        # Plot
        plt.subplot(2,1,1)
        plt.plot(self.t[0:52],self.estmean[0,:],label="Estimated State")
        plt.plot(self.t[0:52],true[0,0:52],label="True State")
        plt.title("Estimated vs True State")
        plt.xlabel("Time Step")
        plt.ylabel("State Value")
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(self.t[0:52],self.error[0,:],label="Error")
        plt.title(f"Error Over Time (RMSE: {self.rmse:.2f})")
        plt.xlabel("Time Step")
        plt.ylabel("Error")
        plt.legend()  # Add the legend

        # Show the plot
        plt.tight_layout()  # Adjust the spacing between subplots
        plt.show()

    def main(self):
        # Generate true states
        trueState = np.empty((self.n,self.numStep))
        procNoise = np.sqrt(self.Q)*np.random.randn(self.n,self.numStep)
        trueState[:,0] = self.trueinitial+np.sqrt(self.P0)*np.random.randn(self.n,1)

        for i in range(1,self.numStep):
            trueState[:,i] = self.funsys(trueState[:,i-1],procNoise[:,i-1],i-1)
        
        # Generate measurements
        measNoise = np.sqrt(self.R)*np.random.randn(self.nm,self.numStep)
        meas = self.funmeas(trueState,measNoise)

        # PGM filter
        self.particle[:,:,0] = self.initial+np.sqrt(self.P0)*np.random.randn(self.n,self.numParticles)
        self.estmean[:,0] = np.mean(self.particle[:,:,0])
        self.error[:,0] = abs(self.estmean[:,0]-trueState[:,0])

        label = np.zeros((self.n,self.numParticles))

        # Clustering
        label = self.PGM_DS_clustering(label,0)

        for t in range(1,self.tf):
            # Prediction
            for p in range(self.numParticles):
                self.particle[:,p,t] = self.funsys(self.particle[:,p,t-1],np.sqrt(self.Q)*np.random.randn(1),t-1)

            # Clustering
            label = self.PGM_DS_clustering(label,t)

            # Merging
            label = self.PGM_DS_merging(label,t)

            # Update
            self.PGM_DS_update(label,meas[:,t],t)

            self.error[:,t] = abs(self.estmean[:,t]-trueState[:,t])

            # Resampling from GM
            self.PGM_DS_resampling(t)

        self.rmse = np.sqrt(sum([x**2 for x in self.error[0,:]])/len(self.error[0,:]))

        # Plot
        self.plot_graph(trueState)

def main():
    pgm_ds_test = PGM_DS()
    pgm_ds_test.main()

if __name__ == "__main__":
    main()
