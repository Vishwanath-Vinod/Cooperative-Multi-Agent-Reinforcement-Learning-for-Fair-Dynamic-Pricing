import numpy as np
import pandas as pd
import torch as T

class Fairness:

    def __init__(self,max_price=500,min_price=100):
        self.max_price=max_price
        self.min_price=min_price
        
    def jain_index(self,list_prices,target_fairness=0.9, alpha1=1.0, alpha2=1.0):
        sum_prices=sum(price for price in list_prices)
        sum_square_prices=sum(price*price for price in list_prices)
        n=len(list_prices)
        if n==0:
            return 0.0
        jainindex=(sum_prices*sum_prices)/(n*sum_square_prices)
        
        return jainindex
        
    def gini_index(self,list_prices):
        sum_diff=0
        for price in list_prices:
            for price_ in list_prices:
                sum_diff+=abs(price-price_)
        sum=np.sum(price for price in list_prices)  
        n=len(list_prices)  
        if n==0:
            return 1
        giniindex=sum_diff/(2*n*sum)  
        
        return giniindex
        
    def rotated_jain_index(self,list_prices):  
        sum=np.sum(self.max_price-price for price in list_prices)
        sum_square=np.sum((self.max_price-price)*(self.max_price-price) for price in list_prices)
        n=len(list_prices)
        rot_jainindex=(sum*sum)/(n*sum_square)
        
        return rot_jainindex   
    
    def QoE(self,list_prices):
        mean = np.mean(list_prices)
        std = np.sqrt(np.sum((price-mean)**2 for price in list_prices)/len(list_prices))
        QoE = 1-(2*std/(self.max_price-self.min_price)) 
        
        return QoE
