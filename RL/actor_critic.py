import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


# =====================================================================
# 1. THE ACTOR: π_θ(a|s) -> "The Policy"
# =====================================================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        # THEORY: The Actor decides WHAT to do. It looks at the state
        # and outputs the mean (μ) of a continuous action distribution.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # THEORY: The log standard deviation controls exploration.
        # Kept independent of the state to ensure exploration doesn't collapse.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.net(state)
        std = self.log_std.exp()
        return mean, std


# =====================================================================
# 2. THE CRITIC: V_w(s) -> "The Value Function"
# =====================================================================
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        # THEORY: The Critic evaluates HOW GOOD the state is.
        # It takes the state as input and outputs a single scalar value V(s).
        # This value represents the expected discounted sum of future rewards
        # starting from this state. It acts as the "dynamic baseline".
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)


# =====================================================================
# 3. THE ACTOR-CRITIC ALGORITHM
# =====================================================================
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        # ENGINEERING BEST PRACTICE: The Critic learns faster than the Actor
        # (lr_critic > lr_actor). The Critic needs to quickly learn accurate
        # values so it can properly guide the Actor. If the Actor changes too
        # fast based on a "dumb" Critic, the whole system collapses.
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.saved_log_probs = []
        self.values = []
        self.rewards = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Forward pass through both networks
        mean, std = self.actor(state_tensor)
        value = self.critic(state_tensor)

        dist = Normal(mean, std)
        action = dist.sample()

        # Save data for the update step
        self.saved_log_probs.append(dist.log_prob(action).sum())
        self.values.append(value)  # Store the Critic's V(s) prediction

        return action.detach().numpy()[0]

    def compute_returns(self):
        # NOTE: This is the old Monte Carlo return calculator from REINFORCE.
        # It calculates the true sum of rewards. It is left here for reference,
        # but the update() function below overrides this by using TD Learning!
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self):
        # 1. Convert memory lists to PyTorch Tensors
        rewards_tensor = torch.FloatTensor(self.rewards)
        values_tensor = torch.cat(self.values)
        log_probs_tensor = torch.stack(self.saved_log_probs)

        # =================================================================
        # THEORY: BOOTSTRAPPING & TD ERROR
        # Instead of waiting for the real final return (Monte Carlo), we use
        # Temporal Difference (TD) learning. The agent guesses the future.
        # TD Target = Reward_Now + (Gamma * Critic's Guess of Next State)
        # Advantage (TD Error) = TD Target - Critic's Guess of Current State
        # Math: δ_t = R_t + γ * V(s_{t+1}) - V(s_t)
        # =================================================================
        advantages = []
        returns = []

        # We assume the value of the state AFTER the episode ends is 0.
        next_value = 0

        for t in reversed(range(len(self.rewards))):
            # Calculate the 1-step TD Target (Bootstrapping)
            td_target = rewards_tensor[t] + self.gamma * next_value
            returns.insert(0, td_target)

            # Calculate Advantage (How much better was the action than expected?)
            delta = td_target - values_tensor[t].item()
            advantages.insert(0, delta)

            # Shift the next_value for the previous step in the backwards loop
            next_value = values_tensor[t].item()

        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)

        # ENGINEERING NOTE: Advantage Normalization (Commented out, but useful)
        # Normalizing advantages to have mean=0 and std=1 keeps the gradient
        # variance perfectly bounded, drastically improving PyTorch stability.
        # advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # =================================================================
        # LOSS CALCULATIONS
        # =================================================================

        # 3. Actor Loss (Policy Gradient)
        # Objective: Maximize expected advantage.
        # Math: L_actor = - (log π(a|s) * Advantage)
        # We MUST use .detach() on advantages_tensor. If we don't, PyTorch
        # will try to backpropagate the Actor's loss through the Critic's network,
        # destroying the Critic's weights.
        actor_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()

        # 4. Critic Loss (Value Function Regression)
        # Objective: Make the Critic's guess V(s) closer to the TD Target.
        # Math: L_critic = Mean Squared Error( V(s_t), R_t + γ V(s_{t+1}) )
        critic_loss = nn.MSELoss()(values_tensor, returns_tensor)

        # Apply gradients to Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Apply gradients to Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Clear memory buffers for the next episode
        self.saved_log_probs[:] = []
        self.values[:] = []
        self.rewards[:] = []

        return actor_loss.item(), critic_loss.item()