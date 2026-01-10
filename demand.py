import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import gym
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from gym import spaces
from numpy.random import default_rng


class Demands:

    def __init__(self,max_price=500,min_price=100,exp_decay_constant=3,
                 gaussian_mean=300,gaussian_std=8000,sigmoid_bias=6,sigmoid_weight=-0.02,a1=20.446,a2=4.44,type='Linear'):
        self.min_price=min_price
        self.max_price=max_price
        self.type=type
        self.k=exp_decay_constant
        self.b=sigmoid_bias
        self.w=sigmoid_weight
        self.mean=gaussian_mean
        self.std=gaussian_std
        self.a1=a1 
        self.a2=a2

    def  discrete_demand(self,price=250):
        self.price=price
        if self.type=='Linear':
            prob=(self.max_price-self.price)/(self.max_price-self.min_price)
            return np.random.binomial(n=1,p=prob)
        if self.type=='Exponential':
            prob=math.exp(-self.k*(self.price-self.min_price)/(self.max_price-self.min_price))
            return np.random.binomial(n=1,p=prob)
        if self.type=='Logit':
            exp_prob=math.exp(-self.k*(self.price-self.min_price)/(self.max_price-self.min_price))
            exp_value=np.random.binomial(n=1,p=exp_prob)
            prob=(exp_value/(1+exp_value))
            return np.random.binomial(n=1,p=prob)
        if self.type =='Sigmoidal':
            prob=1/(1+np.exp(-self.b-self.w*price))
            return np.random.binomial(n=1,p=prob)
        if self.type =='Inverse-Sigmoidal':
            prob=1-(1/(1+np.exp(-self.b-self.w*price)))
            return np.random.binomial(n=1,p=prob)
        if self.type =='Gaussian':
            prob=np.exp(-((price-self.mean)**2)/self.std)
            return np.random.binomial(n=1,p=prob)
        if self.type =='Log-Linear':
            prob= np.exp(self.a1) * price**(-self.a2)
            return np.random.binomial(n=1,p=prob)
        if self.type =='Random':
            prob=0.5
            return np.random.binomial(n=1,p=prob)
        if self.type =='Price-Inelastic':
            return 1
    
    def interpolate_demand(self,num_ranges=100,prices_explored_per_range=2500):
        price_ranges=np.linspace(self.min_price,self.max_price, num_ranges + 1)
        new_price_ranges=[]
        prob_counts=[]
        for i in range(len(price_ranges)-1):
            new_price_ranges.append((price_ranges[i]+price_ranges[i+1])/2)
            
        for i in range(num_ranges):
            start_price = price_ranges[i]
            end_price = price_ranges[i+1]
            prices = np.random.uniform(start_price, end_price, prices_explored_per_range)
            prob_counts.append(sum(self.discrete_demand(price)  for price in prices)/prices_explored_per_range)
            
        interpolated_demand = CubicSpline(new_price_ranges, prob_counts)
        return interpolated_demand
        
    
    def demand(self,price=250):
        prob=self.interpolate_demand()
        return prob(price)
    
    def plot_demand(self):
        prices_test=np.linspace(self.min_price,self.max_price, 100)
        test_counts=[]
        for test  in prices_test:
            test_counts=self.interpolate_demand()
            counts=test_counts(prices_test)
        if self.type=='Sigmoidal':    
            plt.plot(prices_test,counts,label={self.type},color='blue')
        if self.type=='Inverse-Sigmoidal':    
            plt.plot(prices_test,counts,label={self.type},color='red')
        if self.type=='Gaussian':    
            plt.plot(prices_test,counts,label={self.type},color='navy')    
        if self.type=='Log-Linear':    
            plt.plot(prices_test,counts,label={self.type},color='orange')    
        if self.type=='Random':
            plt.plot(prices_test,counts,label={self.type},color='orange')
        if self.type=='Price-Inelastic':
            plt.plot(prices_test,counts,label={self.type},color='green')           
        plt.ylabel('Demand Probability')
        plt.xlabel('Price')
        plt.title('Demand Functions of Groups')

'''

dem1=Demands(type='Sigmoidal')
dem1.plot_demand()

dem2=Demands(type='Inverse-Sigmoidal')
dem2.plot_demand()


dem4=Demands(type='Log-Linear')
dem4.plot_demand()

dem5=Demands(type='Price-Inelastic')
dem5.plot_demand()
plt.legend()
plt.savefig('Demand Probabilities of Groups')
'''
