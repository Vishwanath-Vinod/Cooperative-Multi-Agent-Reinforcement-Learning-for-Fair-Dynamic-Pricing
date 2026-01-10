import math
import numpy as np
import matplotlib.pyplot as plt


class Customer:

  def __init__(self,num_days=30):
    self.num_days = num_days
    self.num_customers = 600
    
  def customer_influx_constant(self):
    customers_per_day = np.round(np.full(self.num_days, self.num_customers/self.num_days))
    '''
    plt.title('Constant customer influx')
    plt.figtext(x=0.4,y=0.85,s=f'Customers per day={int(customers_per_day[0])}')
    plt.plot(customers_per_day)
    plt.show()
    '''
    return customers_per_day

  def customer_influx_poisson(self,lamb=3):
    self.lamb=lamb
    customers_per_day=np.random.poisson(self.lamb,self.num_days)
    customers_sum = np.sum(customers_per_day)
    customers_per_day= np.round((self.num_customers / customers_sum)*customers_per_day) #NORMALIZING THE CUSTOMER DISTRIBUTION
    ''''
    plt.title('Poisson customer influx')
    plt.figtext(x=0.4,y=0.85,s=f'lambda={self.lamb}')
    plt.plot(customers_per_day)
    plt.show()
    '''
    return customers_per_day
  
  def customer_influx_gaussian(self,mean=3,std=1):
    self.mean=mean
    self.std=std
    customers_per_day = np.round(np.random.normal(loc=self.mean,scale=self.std,size=self.num_days))
    print(sum(customers_per_day))
    '''
    plt.plot(customers_per_day)
    plt.title('Gaussian  customer influx')
    plt.figtext(x=0.4,y=0.85,s=f'mean={self.mean} std={self.std}')
    plt.show()
    '''
    return customers_per_day
  
  def customer_influx_exp_decaying(self,decay_constant=1.5):
    self.k=decay_constant
    customers_per_day=[np.exp(-self.k*(i)/(self.num_days))for i in range(self.num_days+1)]
    customers_sum=np.sum(customers_per_day)
    customers_per_day=np.array(customers_per_day)
    customers_per_day= np.round((self.num_customers/ customers_sum)*customers_per_day)
    '''
    plt.plot(customers_per_day)
    plt.title('Decaying customer influx')
    plt.figtext(x=0.4,y=0.85,s=f'Decay constant = {self.k}')
    plt.show()
    '''
    return customers_per_day
    
    
  def customer_influx_exp_growing(self,growth_constant=1.5):
    self.k=growth_constant
    customers_per_day=[np.exp(self.k*(i)/(self.num_days))for i in range(self.num_days+1)]
    customers_sum=np.sum(customers_per_day)
    customers_per_day=np.array(customers_per_day)
    customers_per_day= np.round((self.num_customers/ customers_sum)*customers_per_day)
    '''
    plt.plot(customers_per_day)
    plt.title('Growing customer influx')
    plt.figtext(x=0.4,y=0.85,s=f'Decay constant = {self.k}')
    plt.show()
    '''
    return customers_per_day
    
      
    
    
