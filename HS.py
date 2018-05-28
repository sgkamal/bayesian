import numpy as np
import scipy as sp
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 200
from numpy.linalg import inv
from scipy.stats import norm as normal
from scipy.stats import gamma
from scipy.stats import multivariate_normal as mvnorm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
import seaborn as sns
import pymc
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


X = []
p = 20
n = 100
for i in range(n):
    X.append(normal.rvs(0,10,p))
X = np.array(X)


sigma = 1
sigma_2_I = np.diag([sigma]*100)
eta_model = gamma(1/2,1/2)
eta = eta_model.rvs(1)[0]
phi_eta_model = gamma(1/2, eta/2)
phi_eta = phi_eta_model.rvs(1)[0]
lmbda_model = gamma(1/2, phi_eta)
LMBDA = lmbda_model.rvs(p).reshape(1,p)


class T_model:
    def __init__(self, LMBDA):
        self.LMBDA = LMBDA
        self.dim = self.LMBDA.shape[1]

    def rvs(self, nums):
        ans = np.empty((nums, self.dim))
        for num in range(nums):
            for lmbda in range(self.dim):
                temp_model = gamma(1/2, self.LMBDA[0,lmbda])
                ans[num, lmbda] = temp_model.rvs(1)[0]
        return(ans)

T_m = T_model(LMBDA)
T = T_m.rvs(1) #(1,20)


class Beta_model:
    def __init__(self, T, sigma):
        self.T = T
        self.sigma_2 = sigma**2
        self.dim = self.T.shape[1]

    def rvs(self, nums):
        ans = np.empty((nums,self.dim))
        #print(ans)
        for num in range(nums):
            for t in range(self.dim):
                temp_model = normal(0, self.sigma_2/self.T[0,t])
                ans[num, t] = temp_model.rvs(1)[0]
        return(ans)

Beta_m = Beta_model(T, sigma)
Beta = Beta_m.rvs(1)



class Y_model:
    def __init__(self, X, Beta, sigma):
        self.Beta = Beta
        self.X = X
        self.sigma = sigma
        self.sigma_2 = sigma**2
        self.sigma_2_I = np.diag([self.sigma_2] * self.X.shape[0])

    def rvs(self, nums):
        ans = np.empty((nums, self.X.shape[0]))
        for num in range(nums):
            temp_model = mvnorm(np.dot(self.X, self.Beta.T).T[0,:], self.sigma_2_I)
            ans[num,:] = (temp_model.rvs(1))
        return(ans.T)

Y_m = Y_model(X, Beta, sigma)
Y = Y_m.rvs(1)



def l2norm(Y, X, Beta):
    return(np.dot(Y.T, Y) - np.dot(np.dot(Y.T, X), np.dot(inv(np.dot(X.T, X) + T),np.dot(X.T, Y))))

squarer = np.vectorize(lambda i:i**2)


class Beta_full:
    def __init__(self, X, T, Y, sigma_2):
        self.X = X
        self.T = T
        self.dim = self.T.shape[1]
        self.Y = Y
        self.sigma_2 = sigma_2

    def rvs(self, nums):
        ans = np.empty((nums, self.dim))
        cov = inv(np.dot(self.X.T, self.X) + self.T)*self.sigma_2
        min_eig = np.min(np.real(np.linalg.eigvals(cov)))
        if min_eig < 0:
            cov -= 10*min_eig * np.eye(*cov.shape)
        temp_model = mvnorm(np.dot(inv(np.dot(self.X.T, self.X) + self.T),np.dot(self.X.T, self.Y)).T[0,:], cov)
        for num in range(nums):
            ans[num,:] = temp_model.rvs(1)
        return(ans)


np.min(np.real(np.linalg.eigvals(inv(np.dot(X.T, X) + T)*sigma**2)))


class Sigma_2_full:
    def __init__(self, Y, X, Beta, T):
        self.Y = Y
        self.X = X
        self.Beta = Beta
        self.T = T
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

    def rvs(self, nums):
        #print(((l2norm(self.Y, self.X, self.Beta) + np.dot(squarer(self.Beta), self.T) + 1)/2).T.shape)
        temp_model = gamma((self.n + self.p + 1)/2, ((l2norm(self.Y, self.X, self.Beta)[0,0] + np.dot(squarer(self.Beta), self.T.T)[0,0] + 1)/2))
        return(1/(temp_model.rvs(1)[0]))


class T_full:
    def __init__(self, Beta, LMBDA, sigma_2):
        self.Beta = Beta
        self.LMBDA = LMBDA
        self.sigma_2 = sigma_2
        self.dim = self.Beta.shape[1]

    def rvs(self, nums):
        ans = np.empty((nums,self.dim))
        for num in range(nums):
            for i in range(self.dim):
                temp_model = gamma(1, self.Beta[0,i]**2/(2*self.sigma_2) + self.LMBDA[0,i]/2)
                ans[num, i] = temp_model.rvs(1)[0]
        return(ans)


class LMBDA_full:
    def __init__(self, T, phi):
        self.T = T
        self.dim = self.T.shape[1]
        self.phi = phi

    def rvs(self, nums):
        ans = np.empty((nums, self.dim))
        for num in range(nums):
            for i in range(self.dim):
                temp_model = gamma(1, (self.T[num,i] + self.phi)/2)
                ans[num, i] = temp_model.rvs(1)[0]
        return(ans)


class phi_full:
    def __init__(self,eta, p):
        self.eta = eta
        self.p = p
    def rvs(self, nums):
        ans = np.empty((nums, 1))
        for num in range(nums):
            temp_model = gamma((self.p + 1)/2 , (self.eta + 1)/2)
            ans[num, 0] = temp_model.rvs(1)[0]
        return(ans)


class eta_full:
    def __init__(self, phi):
        self.phi = phi

    def rvs(self, nums):
        ans = np.empty((nums,1))
        for num in range(nums):
            temp_model = gamma(1, (self.phi+1)/2)
            ans[num,0] = temp_model.rvs(1)[0]
        return(ans)


nums = 3000
sigma_2_hat = np.random.uniform(0,5)**2
T_hat = normal.rvs(0, np.random.uniform(0,5)**2, size=p).reshape(1,p)
LMBDA_hat = gamma.rvs(np.random.uniform(0,5), np.random.uniform(0,5), size=p).reshape(1,p)
phi_hat = gamma.rvs(np.random.uniform(0,5), np.random.uniform(0,5))
eta_hat = gamma.rvs(np.random.uniform(0,5), np.random.uniform(0,5))


BigT = np.empty((3000, 20))
for num in range(nums):
    Beta_hat = Beta_full(X=X, T=T_hat, Y=Y, sigma_2=sigma_2_hat).rvs(1) #1xp
    sigma_2_hat = Sigma_2_full(Y=Y, X=X, Beta=Beta_hat, T=T_hat).rvs(1) #scalar
    T_hat = T_full(Beta=Beta_hat, LMBDA=LMBDA_hat, sigma_2=sigma_2_hat).rvs(1) #1xp
    BigT[num,:] = T_hat
    LMBDA_hat = LMBDA_full(T=T_hat, phi=phi_hat).rvs(1) #1xp
    phi_hat = phi_full(eta=eta_hat, p=p).rvs(1)[0,0] #scalar
    eta_hat = eta_full(phi=phi_hat).rvs(1)[0,0] #scalar
print(Beta)
print(Beta_hat)


from matplotlib import pyplot
for i in range(20):
    plot_acf(BigT[:,i])
    pyplot.show()


for num in range(nums):
    Beta_hat = Beta_full(X=X, T=T_hat, Y=Y, sigma_2=sigma_2_hat).rvs(1) #1xp
    sigma_2_hat = Sigma_2_full(Y=Y, X=X, Beta=Beta_hat, T=T_hat).rvs(1) #scalar
    T_hat = T_full(Beta=Beta_hat, LMBDA=LMBDA_hat, sigma_2=sigma_2_hat).rvs(1) #1xp
    LMBDA_hat = LMBDA_full(T=T_hat, phi=phi_hat).rvs(1) #1xp
    phi_hat = phi_full(eta=eta_hat, p=p).rvs(1)[0,0] #scalar
    eta_hat = eta_full(phi=phi_hat).rvs(1)[0,0] #scalar
print(Beta)
print(Beta_hat)


#BigT = np.empty((1000, 20))
for num in range(nums):
    Beta_hat = Beta_full(X=X, T=T_hat, Y=Y, sigma_2=sigma_2_hat).rvs(1) #1xp
    sigma_2_hat = Sigma_2_full(Y=Y, X=X, Beta=Beta_hat, T=T_hat).rvs(1) #scalar
    T_hat = T_full(Beta=Beta_hat, LMBDA=LMBDA_hat, sigma_2=sigma_2_hat).rvs(1) #1xp
    #BigT[num,:] = T_hat
    LMBDA_hat = LMBDA_full(T=T_hat, phi=phi_hat).rvs(1) #1xp
    phi_hat = phi_full(eta=eta_hat, p=p).rvs(1)[0,0] #scalar
    eta_hat = eta_full(phi=phi_hat).rvs(1)[0,0] #scalar
print(Beta)
print(Beta_hat)
