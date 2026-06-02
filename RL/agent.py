import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


# =====================================================================
# 1. THE POLICY NETWORK: π_θ(a|s)
# =====================================================================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()

        # THEORY: In continuous control (like the Hopper), the network
        # doesn't output discrete actions (like "go left" or "go right").
        # Instead, it outputs the parameters of a probability distribution.
        # This lightweight Multilayer Perceptron (MLP) outputs the MEAN (μ)
        # of the action distribution.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        # THEORY: We also need a Standard Deviation (σ) to control exploration.
        # We define the log standard deviation as a standalone, trainable parameter
        # independent of the state. This is an industry best-practice for MuJoCo tasks
        # because it prevents the standard deviation from collapsing to zero too quickly,
        # which would halt exploration.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.net(state)
        # THEORY: We use the exponential function on log_std to guarantee that
        # the standard deviation (σ) is strictly positive (σ > 0).
        std = self.log_std.exp()
        return mean, std


# =====================================================================
# 2. THE REINFORCE ALGORITHM (Vanilla Policy Gradient)
# =====================================================================
class ReinforceAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, baseline=0.0):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Gamma (γ) determines how much we care about future rewards.
        # γ = 0.99 means we care heavily about long-term survival.
        self.gamma = gamma
        self.baseline = baseline

        # THEORY: REINFORCE is a Monte Carlo algorithm. This means it CANNOT
        # update mid-episode. We must store the trajectory data in these memory
        # buffers and calculate the loss only after the Hopper falls over.
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        # Convert the raw numpy state array from Gym into a PyTorch Tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state_tensor)

        # Create a Normal (Gaussian) distribution using our μ and σ
        dist = Normal(mean, std)

        # THEORY: Sample an action a ~ π_θ(·|s). Because this involves randomness,
        # it drives the agent's exploration of the environment.
        action = dist.sample()

        # THEORY (Policy Gradient Theorem): To update the network, we need the
        # gradient of the log probability of the action: ∇_θ log π_θ(a|s).
        # We calculate the log probability here and store it in the computation
        # graph so PyTorch can backpropagate through it later.
        # We use .sum() because the 3 joint actions in Hopper are assumed independent.
        self.saved_log_probs.append(dist.log_prob(action).sum())

        # Detach the tensor from the graph and convert to numpy for the Gym step() function
        return action.detach().numpy()[0]

    def compute_returns(self):
        R = 0
        returns = []
        # THEORY: We calculate the actual Discounted Return (G_t) for each time step.
        # We iterate backwards through the rewards list.
        # Equation: G_t = R_{t+1} + γ * G_{t+1}
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self):
        returns = self.compute_returns()
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # =================================================================
        # THEORY: THE CONSTANT BASELINE
        # We subtract a static, hardcoded scalar (like 30.0).
        # If the discounted return G_t is greater than 30, the advantage
        # is positive and the action is reinforced. If it is less than 30,
        # the advantage is negative and the action is penalized.
        # =================================================================

        policy_loss = []
        for t, (log_prob, R) in enumerate(zip(self.saved_log_probs, returns_tensor)):
            # 2. Apply the baseline to calculate Advantage
            advantage = R - self.baseline

            # 3. Apply the gamma^t scaling as defined in Sutton & Barto Section 13.3
            # THEORY: Actions taken early in the episode have a larger impact on the
            # objective function than actions taken right before falling.
            gamma_t = self.gamma ** t

            # THEORY: The Objective Function J(θ) for REINFORCE is:
            # J(θ) = E[ γ^t * (G_t - b) * ∇_θ log π_θ(a|s) ]
            # We want to MAXIMIZE this. However, PyTorch's Adam optimizer is designed
            # to MINIMIZE loss. Therefore, we multiply the entire term by negative one (-).
            policy_loss.append(-log_prob * advantage * gamma_t)

        # Clear existing gradients
        self.optimizer.zero_grad()

        # Sum the losses over the trajectory to get the total episode loss
        loss = torch.stack(policy_loss).sum()

        # Backpropagate to calculate the gradients (∇_θ)
        loss.backward()

        # Take a gradient ascent step to update the network weights (θ)
        self.optimizer.step()

        # ENGINEERING REQUIREMENT: Clear the memory buffers!
        # If we don't clear these, the lists will grow indefinitely, mixing data
        # from old episodes and causing massive memory leaks.
        self.saved_log_probs[:] = []
        self.rewards[:] = []

        return loss.item()