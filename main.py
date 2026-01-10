from dec_maddpg_dual import MADDPG
import numpy as np
import torch as T
import gym
import matplotlib.pyplot as plt
import csv
import datetime
from collections import deque, namedtuple
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler
from fairness_metric import Fairness
scale_reward = 1.0

gym.register('Fair_Dynamic_Pricing_MADDPG',entry_point='env:DynamicPricingMultiAgent')
env=gym.make('Fair_Dynamic_Pricing_MADDPG') 

np.random.seed(41)
T.manual_seed(41)
env.seed(41)

n_agents = env.num_agents
n_states = env.observation_space.shape[0] // n_agents
n_actions = env.action_space.shape[0] // n_agents
capacity = 1e6
batch_size = 256

n_episode = 2000
episodes_before_train = 200
fair = Fairness()
finite_inventory_penalization = 0.5
maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,episodes_before_train)
##########  LOGGING THE DATA  ############
csv_file = 'Dec_MADDPG_Results.csv'
logger = logging.getLogger('Algorithm-2')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(csv_file, mode='w', delay=False)
formatter = logging.Formatter('%(message)s')  # No log format, just CSV data
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Episode,Profit,Fairness")
###########################################
FloatTensor = T.cuda.FloatTensor if maddpg.use_cuda else T.FloatTensor

def train(n_episode = n_episode):
    episodic_profits = []
    episodic_fairness = []
    
    for i_episode in tqdm(range(0,n_episode)):
        obs = env.reset()
        obs = np.stack(obs)
        if isinstance(obs, np.ndarray):
            obs = T.from_numpy(obs).float()
        '''   
        rr = np.zeros((n_agents,))
        '''
        inventory_sold =[0 for _ in range(n_agents)]
        done = [np.bool_(False) for _ in range(n_agents)]
        profits = [0.0 for _ in range(n_agents)] #score
        rewards = [0.0 for _ in range(n_agents)] #combined_score
        total_profit = 0.0
        total_reward = 0.0
        stepwise_fairness = []
        price = [0 for _ in range(n_agents)]
        
        
        while not np.all(done):
            # render every 100 episodes to speed up training
            '''
            if i_episode % 100 == 0 :
                env.render()
            '''    
            obs = obs.type(FloatTensor)
            action = maddpg.select_action(obs).data.cpu()
            obs_, reward, done, sale = env.step(action.numpy())
            
            for i in range(n_agents):
                if sale['SOLD'][i]:
                    inventory_sold[i]+=1
                price[i] = action[i]
                total_profit= total_profit + reward[i]
            stepwise_fairness.append(
                    fair.jain_index([price[0],price[1],price[2],price[3]])
                                    )       
            reward = T.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = T.from_numpy(obs_).float()
            
            if not(np.all(done)):
                next_obs = obs_
            else:
                next_obs = None
            '''
            total_reward += reward.sum()
            rr += reward.cpu().numpy()
            '''
            maddpg.memory.push(obs.data, action, next_obs, reward)
            obs = next_obs

            maddpg.update_policy()
        
         
        maddpg.episode_done += 1
        episodic_profits.append(total_profit)   
        fm = np.mean(stepwise_fairness)
        episodic_fairness.append(fm)
        log_msg = f"{i_episode+1},{episodic_profits[i_episode]},{episodic_fairness[i_episode]}"
        logger.info(log_msg)
        
        if maddpg.episode_done == maddpg.episodes_before_train:
            print('TRAINING BEGINS')
            
    return [np.array(episodic_profits),np.array(episodic_fairness)]        

        
##############  TRIAL RUN ################
profits,fairness=train()
##########################################
