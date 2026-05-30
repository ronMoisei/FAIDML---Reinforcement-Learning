import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        # Lightweight network as requested
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # State-independent log standard deviation (standard practice for stability)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.net(state)
        # Exponential ensures std is strictly positive
        std = self.log_std.exp()
        return mean, std


class ReinforceAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, baseline=0.0):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.baseline = baseline

        # Memory buffers for the current trajectory
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state_tensor)

        # Create Normal distribution and sample action
        dist = Normal(mean, std)
        action = dist.sample()

        # Save the log probability for the loss calculation (Graph stays attached)
        # We sum the log_probs because the actions are independent dimensions
        self.saved_log_probs.append(dist.log_prob(action).sum())

        # Detach and convert to numpy for the Gym environment
        return action.detach().numpy()[0]

    def compute_returns(self):
        R = 0
        returns = []
        # Calculate discounted returns backwards
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self):
        returns = self.compute_returns()

        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            # Apply the constant baseline
            advantage = R - self.baseline
            # Negative sign because we want gradient ASCENT (PyTorch minimizes loss)
            policy_loss.append(-log_prob * advantage)

        self.optimizer.zero_grad()
        # Sum the losses over the trajectory
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        # CLEAR MEMORY buffers to prevent memory leaks in the next episode
        self.saved_log_probs[:] = []
        self.rewards[:] = []

        return loss.item()