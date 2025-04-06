import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from model import VPG

class Agent():
    def __init__(self):
        # Improved hyperparameters
        self.gamma = 0.99
        self.lr = 1e-3               # Higher learning rate
        self.entropy_coef = 0.01     
        self.grad_clip = 0.5         
        self.value_coef = 0.5        
        self.init_std = 0.3          # Lower initial exploration
        self.gae_lambda = 0.95       
        self.batch_size = 128        
        self.n_epochs = 4            
        self.clip_ratio = 0.2        
        self.target_kl = 0.01        
        self.lr_decay = 0.999        
        self.max_grad_norm = 0.5     
        self.replay_size = 100000   
        self.warmup_steps = 1000     
        
        self.model = VPG(with_baseline=True, init_std=self.init_std, hidden_sizes=[256, 128])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        
        self.replay_buffer = ReplayBuffer(self.replay_size)
        self.total_steps = 0
        
    def decide_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            policy_output, value = self.model(state_tensor, return_baseline=True)
            action_mean, action_log_std = policy_output.chunk(2, dim=-1)
            action_std = torch.exp(action_log_std)
        
        exploration_factor = max(0.1, 1.0 - (self.total_steps / 100000))
        dist = torch.distributions.Normal(action_mean, action_std * exploration_factor)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        return action.squeeze(0).numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        
    def update_model(self):
        if len(self.replay_buffer) < self.warmup_steps:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        with torch.no_grad():
            _, values = self.model(states, return_baseline=True)
            _, next_values = self.model(next_states, return_baseline=True)
            
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values
        
        policy_outputs, values = self.model(states, return_baseline=True)
        action_mean, action_log_std = policy_outputs.chunk(2, dim=-1)
        action_std = torch.exp(action_log_std)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(-1)
        
        ratio = torch.exp(log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        entropy = dist.entropy().mean()
        
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)
        
