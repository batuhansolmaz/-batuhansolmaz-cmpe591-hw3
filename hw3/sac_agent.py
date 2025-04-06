import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

BUFFER_SIZE = 100000    
BATCH_SIZE = 256         # Minibatch size
GAMMA = 0.99             # Discount factor
TAU = 0.005              # For soft update of target parameters
LR_ACTOR = 3e-4          
LR_CRITIC = 3e-4         
ALPHA = 0.2              # Entropy regularization coefficient
HIDDEN_SIZE = 256        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = tuple

    def add(self, state, action, reward, next_state, done):
        e = (state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=min(self.batch_size, len(self.memory)))
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=HIDDEN_SIZE, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mu, std
    
    def sample(self, state):
        mu, std = self.forward(state)
        
        normal = torch.distributions.Normal(mu, std)
        
        x_t = normal.rsample()
        
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=HIDDEN_SIZE):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1 = nn.Linear(hidden_size, 1)
        
        self.fc3 = nn.Linear(state_size + action_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.q2 = nn.Linear(hidden_size, 1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state, action):
        xs = torch.cat([state, action], dim=1)
        
        x1 = F.relu(self.fc1(xs))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        
        x2 = F.relu(self.fc3(xs))
        x2 = F.relu(self.fc4(x2))
        q2 = self.q2(x2)
        
        return q1, q2


class SACAgent:
    
    def __init__(self, state_size=6, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.alpha = ALPHA
        
        self.rewards = []
        
    def decide_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if eval_mode:
                mu, _ = self.actor(state)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(state)
        
        return action.cpu().numpy()[0]
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.rewards.append(reward)
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            q_targets = rewards + GAMMA * (1 - dones) * (q_target - self.alpha * next_log_probs)
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_targets) + F.mse_loss(q2, q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred, log_probs = self.actor.sample(states)
        q1_pred, q2_pred = self.critic(states, actions_pred)
        q_pred = torch.min(q1_pred, q2_pred)
        
        actor_loss = (self.alpha * log_probs - q_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_target)
                     
    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
            
    def add_reward(self, reward):
        """Add reward to the list (for compatibility with other agents)."""
        self.rewards.append(reward)
        
    def update_model(self):
        """Update model (for compatibility with other agents)."""
        self.rewards = [] 