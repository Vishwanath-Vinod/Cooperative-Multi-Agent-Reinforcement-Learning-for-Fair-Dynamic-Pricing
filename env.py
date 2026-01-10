import numpy as np
import gym
from gym import spaces
from numpy.random import default_rng
from customer import Customer
from demand import Demands
np.bool8 = np.bool_

class DynamicPricingMultiAgent(gym.Env):

    def __init__(self, min_price=100, max_price=500, num_inventory=300, num_days=29, num_agents=4):
        super(DynamicPricingMultiAgent, self).__init__()
        self.min_price = min_price
        self.max_price = max_price
        self.num_inventory = num_inventory
        self.num_days = num_days
        self.num_agents = num_agents
        self.types = ['Inverse-Sigmoidal','Sigmoidal','Price-Inelastic','Log-Linear','Random','Gaussian']
        self.customers = [Customer().customer_influx_constant() for _ in range(num_agents)]
        self.demands=[]
        for i in range(num_agents):
            self.demands.append(Demands(max_price=self.max_price, min_price=self.min_price, type=self.types[i]).interpolate_demand())
        
        self.action_space = spaces.Box(low=self.min_price, high=self.max_price, shape=(self.num_agents,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0] * self.num_agents),
                                            high=np.array([self.num_inventory, self.num_days] * self.num_agents),
                                            dtype=np.float32)
        
        
    def reset(self):
        self.days_left = np.full((self.num_agents,),self.num_days)
        self.inventory_left = np.full((self.num_agents,),self.num_inventory)
        self.rewards = np.zeros((self.num_agents))
        self.customer_numbers = [0] * self.num_agents
        self.done = np.full((self.num_agents,),False)
        self.current_obs = np.concatenate([(self.inventory_left[i], self.days_left[i]) for i in range(self.num_agents)])
        self.rng = default_rng()
        return self.current_obs
    
    def step(self, actions):
        self.prices=actions
        remaining_inventory_penalty = 0.5
        sale = {'SOLD': np.zeros(self.num_agents, dtype=bool)}
        for i in range(self.num_agents): 
            if self.done[i]:
                sale['SOLD'][i]=False
                self.rewards[i]=0.0
                
            else:
                self.customer_numbers[i] += 1
                if self.customer_numbers[i] == self.customers[i][self.num_days - self.days_left[i]] and self.days_left[i] != 0:
                    self.customer_numbers[i] = 0
                    self.days_left[i] = self.days_left[i] - 1
                    
                if self.customer_numbers[i] == self.customers[i][self.num_days - self.days_left[i]] and self.days_left[i] == 0:
                    self.done[i] = np.bool_(True)
                    self.customer_numbers[i] = 0
                    
                
                prob = self.demands[i](actions[i])
                #print(f'Action : {actions[i]} probab: {prob}')
                
            
                if self.rng.uniform(0, 1) < prob:
                    sale['SOLD'][i] = True
                    self.inventory_left[i] -= 1
                    self.rewards[i] = ((actions[i]-self.min_price) /(self.max_price-self.min_price))
                else:
                    self.rewards[i]= - remaining_inventory_penalty
                
                if self.inventory_left[i] == 0:
                    self.done[i] = np.bool_(True)
                    
            
        self.current_obs = np.concatenate([(self.inventory_left[i], self.days_left[i]) for i in range(self.num_agents)])
        return self.current_obs, self.rewards, self.done, sale

    def render(self, mode="human"):
        print(f'Next State:{self.current_obs}')
        print(f'Rewards:{self.rewards}')
        print(f'Dones:{self.done}')
        for i in range(self.num_agents):
          print(f'Agent:{i+1} Day: {self.num_days - self.days_left[i]+1}, Inventory Left: {self.inventory_left[i]},Action:{self.prices[i]},Rewards: {self.rewards[i]}')
        
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.rng = default_rng(seed)
        return

##############  TRIAL RUN ################
# env = DynamicPricingMultiAgent()
# env.reset()

# done = [False] * env.num_agents
# scores =[0.0]*env.num_agents
# while not all(done):
#     actions = env.action_space.sample()
#     next_obs, rewards, done, sale = env.step(actions)
#     scores = scores+rewards
#     env.render()
# print(scores)
# env.close()
##########################################