from networks import LocalCritic,GlobalCritic,Actor
import torch as T
from copy import deepcopy
from replay_buffer import ReplayMemory, Experience
from torch.optim import Adam
from noise import OrnsteinUhlenbeckProcess
from fairness_metric import Fairness
import torch.nn as nn
import numpy as np

scale_reward = 10.0
fair = Fairness()
target_fairness = 0.90
MAX_PRICE = 500.0
MIN_PRICE = 100.0
param1 = 1.0
param2 = 1.0

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.localcritics = [LocalCritic(dim_obs,dim_act) for i in range(n_agents)]
        self.globalcritic = GlobalCritic(n_agents, dim_obs, dim_act)
        
        self.actors_target = deepcopy(self.actors)
        self.localcritics_target = deepcopy(self.localcritics)
        self.globalcritic_target = deepcopy(self.globalcritic)
        
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        
        self.memory = ReplayMemory(capacity)
        
        self.batch_size = batch_size
        
        self.use_cuda = T.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01 
        self.var = [1.0 for i in range(n_agents)]
        
        self.ACTOR_LR = 1e-4
        self.LOCAL_CRITIC_LR = 1e-4
        self.GLOBAL_CRITIC_LR = 3e-4
        
        self.localcritic_optimizer = [Adam(x.parameters(),lr=self.LOCAL_CRITIC_LR) for x in self.localcritics]
        self.actor_optimizer = [Adam(x.parameters(),lr=self.ACTOR_LR) for x in self.actors]
        self.globalcritic_optimizer = Adam(self.globalcritic.parameters(),lr = self.GLOBAL_CRITIC_LR)
        
        
        self.ou_processes = [OrnsteinUhlenbeckProcess(theta=0.15, mu=0.0, sigma=160.0, size=dim_act,
                                                      sigma_min=5.0, n_steps_annealing=200000
                                                     ) for _ in range(n_agents)]#CHANGE-2 sigma =160.0 sigma_min =2.0 n_steps =200000
        '''
        Noise Tuning
        #2
        self.ou_processes = [OrnsteinUhlenbeckProcess(theta=0.15, mu=0.0, sigma=120.0, size=dim_act,
                                                      sigma_min=5.0, n_steps_annealing=200000
                                                     ) for _ in range(n_agents)]
        #3
        self.ou_processes = [OrnsteinUhlenbeckProcess(theta=0.10, mu=0.0, sigma=100.0, size=dim_act,
                                                      sigma_min=5.0, n_steps_annealing=500000
                                                     ) for _ in range(n_agents)]
        '''
        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.localcritics:
                x.cuda()
            self.globalcritic.cuda()
            self.globalcritic_target.cuda()        
            for x in self.actors_target:
                x.cuda()
            for x in self.localcritics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None, None

        ByteTensor = T.cuda.ByteTensor if self.use_cuda else T.ByteTensor
        FloatTensor = T.cuda.FloatTensor if self.use_cuda else T.FloatTensor

        lc_loss = []
        gc_loss = []
        a_loss = []
        
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))
        
        non_final_mask = T.tensor([s is not None for s in batch.next_states], dtype=T.bool)
        state_batch = T.stack(batch.states).type(FloatTensor)
        action_batch = T.stack(batch.actions).type(FloatTensor)
        reward_batch = T.stack(batch.rewards).type(FloatTensor) #NOT REQUIRED HERE
        
        non_final_next_states = T.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)
        
        whole_state = state_batch.view(self.batch_size, -1)
        whole_action = action_batch.view(self.batch_size, -1)
        
        self.globalcritic_optimizer.zero_grad()
        
        current_Q = self.globalcritic(whole_state, whole_action)
        non_final_next_states = non_final_next_states.view(-1,self.n_agents,self.n_states)
        non_final_next_actions = [self.actors_target[i](non_final_next_states[:,i,:]) for i in range(self.n_agents)]
        non_final_next_actions = T.stack(non_final_next_actions)
        non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())

        target_Q = T.zeros(self.batch_size).type(FloatTensor)

        target_Q[non_final_mask] = self.globalcritic_target(
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions
        fairness = []    
        for i in range(whole_action.size(0)):
            fairness.append(fair.jain_index(whole_action[i]))
        fairness = T.stack(fairness)    
        
        target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (fairness.unsqueeze(1) * scale_reward)
        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        loss_Q.backward()
        T.nn.utils.clip_grad_norm_(self.globalcritic.parameters(), max_norm=10)
              
        self.globalcritic_optimizer.step()
        gc_loss.append(loss_Q)
        
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))
            non_final_mask = T.tensor([s is not None for s in batch.next_states], dtype=T.bool)
            state_batch = T.stack(batch.states).type(FloatTensor)
            action_batch = T.stack(batch.actions).type(FloatTensor)
            reward_batch = T.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = T.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            state_batch = state_batch.view(-1,self.n_agents,self.n_states)
            action_batch = action_batch.view(-1,self.n_agents,self.n_actions)
            non_final_next_states = non_final_next_states.view(-1,self.n_agents,self.n_states)
            
            self.localcritic_optimizer[agent].zero_grad()
            
            current_Q_lc = self.localcritics[agent](state_batch[:,agent,:],action_batch[:,agent,:])
            non_final_next_actions = [self.actors_target[agent](non_final_next_states[:,agent,:]) for agent in range(self.n_agents)]
            non_final_next_actions = T.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())

            target_Q_lc = T.zeros(self.batch_size).type(FloatTensor)

            target_Q_lc[non_final_mask] = self.localcritics_target[agent](
                non_final_next_states[:,agent,:],
                non_final_next_actions[:,agent,:]
            ).squeeze()

            target_Q_lc = (target_Q_lc.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))
            loss_Q = nn.MSELoss()(current_Q_lc, target_Q_lc.detach())
            loss_Q.backward()
            T.nn.utils.clip_grad_norm_(self.localcritics[agent].parameters(), max_norm=10)
            
            self.localcritic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            local_actor_loss = -self.localcritics[agent](state_batch[:,agent,:],ac[:,agent,:]).mean()
            global_actor_loss = -self.globalcritic(whole_state, ac.view(self.batch_size, -1)).mean()
            actor_loss = param1*local_actor_loss+param2*global_actor_loss
            
            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), max_norm=10)  
            self.actor_optimizer[agent].step()
            lc_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.localcritics_target[i], self.localcritics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
            soft_update(self.globalcritic_target, self.globalcritic, self.tau)
        return lc_loss, a_loss, gc_loss

    def select_action(self, state_batch):
        actions = T.zeros(self.n_agents,self.n_actions)
        FloatTensor = T.cuda.FloatTensor if self.use_cuda else T.FloatTensor
        
        dim_obs = self.n_states
        for i in range(self.n_agents):
            sb = state_batch[i*dim_obs:(i+1)*dim_obs].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()
            ou_noise = T.from_numpy(self.ou_processes[i].sample()).type(FloatTensor)
            act += ou_noise.squeeze()
            
            act = T.clamp(act,MIN_PRICE,MAX_PRICE)
            actions[i, :] = act
        self.steps_done += 1

        return actions