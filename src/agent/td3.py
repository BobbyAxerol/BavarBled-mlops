import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple

class TD3Actor(nn.Module):
    def __init__(self, num_assets=29):
        super(TD3Actor, self).__init__()
        # Input is the theoretical weight w_star (29 assets)
        self.fc1 = nn.Linear(num_assets, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_assets + 1) # 29 Assets + 1 Cash Position

    def forward(self, w_star: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(w_star))
        x = torch.relu(self.fc2(x))
        raw_actions = torch.tanh(self.fc3(x)) # Bound output locally in [-1, 1]
        
        # Absolute-value normalization mapping to meet hard portfolio constraints
        abs_sum = torch.sum(torch.abs(raw_actions), dim=-1, keepdim=True) + 1e-8
        refined_weights = raw_actions / abs_sum
        return refined_weights

class TD3Critic(nn.Module):
    def __init__(self, num_assets=29, state_dim=15*12):
        super(TD3Critic, self).__init__()
        # Flatten state tensor for basic representation alignment within Critic Q-networks
        total_state_dim = num_assets * state_dim
        action_dim = num_assets + 1
        
        # Q1 Architecture
        self.l1 = nn.Linear(total_state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

        # Q2 Architecture
        self.l4 = nn.Linear(total_state_dim + action_dim, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 1)

    def forward(self, state_tensor: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_s = state_tensor.view(state_tensor.size(0), -1)
        sa = torch.cat([flat_s, action], dim=-1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state_tensor: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        flat_s = state_tensor.view(state_tensor.size(0), -1)
        sa = torch.cat([flat_s, action], dim=-1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3Agent:
    def __init__(self, num_assets=29, actor_lr=2.58e-4, critic_lr=6.21e-4, gamma=0.991, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 transformer=None, cnn=None, bled_solver=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = TD3Actor(num_assets).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.transformer = transformer.to(self.device) if transformer is not None else None
        self.transformer_target = copy.deepcopy(self.transformer) if transformer is not None else None

        self.cnn = cnn.to(self.device) if cnn is not None else None
        self.cnn_target = copy.deepcopy(self.cnn) if cnn is not None else None

        self.bled_solver = bled_solver.to(self.device) if bled_solver is not None else None

        actor_params = list(self.actor.parameters())
        if self.transformer is not None:
            actor_params += list(self.transformer.parameters())
        if self.cnn is not None:
            actor_params += list(self.cnn.parameters())

        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr)

        self.critic = TD3Critic(num_assets).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, w_star: np.ndarray) -> np.ndarray:
        w_tensor = torch.FloatTensor(w_star).unsqueeze(0).to(self.device)
        return self.actor(w_tensor).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=1024):
        self.total_it += 1

        # Sample mini-batch from memory
        s, a, r, ns, d, w_star, mu_p, D_p, next_mu_p, next_D_p = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)
        next_state = torch.FloatTensor(ns).to(self.device)
        done = torch.FloatTensor(d).to(self.device)
        w_star_tensor = torch.FloatTensor(w_star).to(self.device)
        mu_p_tensor = torch.FloatTensor(mu_p).to(self.device)
        D_p_tensor = torch.FloatTensor(D_p).to(self.device)
        next_mu_p_tensor = torch.FloatTensor(next_mu_p).to(self.device)
        next_D_p_tensor = torch.FloatTensor(next_D_p).to(self.device)

        with torch.no_grad():
            if self.transformer_target is not None and self.cnn_target is not None and self.bled_solver is not None:
                next_views = self.transformer_target(next_state)
                next_delta = self.cnn_target(next_state)
                next_w_star, _, _ = self.bled_solver(next_mu_p_tensor, next_D_p_tensor, next_views, next_delta)
            else:
                next_w_star = w_star_tensor

            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_w_star) + noise
            # Absolute-value normalization mapping to force target bounds conformity
            next_action = next_action / (torch.sum(torch.abs(next_action), dim=-1, keepdim=True) + 1e-8)

            # Compute target Q-values via Twin Critics
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed updates optimization block
        if self.total_it % self.policy_freq == 0:
            if self.transformer is not None and self.cnn is not None and self.bled_solver is not None:
                views = self.transformer(state)
                delta = self.cnn(state)
                w_star_dyn, _, _ = self.bled_solver(mu_p_tensor, D_p_tensor, views, delta)
                actor_loss = -self.critic.Q1(state, self.actor(w_star_dyn)).mean()
            else:
                actor_loss = -self.critic.Q1(state, self.actor(w_star_tensor)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Synchronize Target Networks via Polyak Averaging
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.transformer is not None:
                for param, target_param in zip(self.transformer.parameters(), self.transformer_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            if self.cnn is not None:
                for param, target_param in zip(self.cnn.parameters(), self.cnn_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)