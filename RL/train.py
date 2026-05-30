import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent import ReinforceAgent


def train_reinforce(env_name='Hopper-v4', episodes=1000, baseline=0.0):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = ReinforceAgent(state_dim, action_dim, lr=1e-3, gamma=0.99, baseline=baseline)
    metrics_rewards = []

    print(f"--- Training REINFORCE (Baseline = {baseline}) ---")
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.rewards.append(reward)
            ep_reward += reward
            state = next_state

            if terminated or truncated:
                break

        # Update network at the end of the episode
        loss = agent.update()
        metrics_rewards.append(ep_reward)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(metrics_rewards[-100:])
            print(f"Episode {ep + 1:4d} | Avg Reward: {avg_reward:.2f} | Last Loss: {loss:.2f}")

    env.close()
    return metrics_rewards


if __name__ == "__main__":
    # Experiment 1: No baseline
    rewards_no_baseline = train_reinforce(baseline=0.0)

    # Experiment 2: Constant baseline
    # (Hint: 20-50 is a reasonable constant to test for early Hopper episodes)
    rewards_const_baseline = train_reinforce(baseline=30.0)

    # Basic plotting block for your report
    plt.plot(rewards_no_baseline, alpha=0.6, label='No Baseline')
    plt.plot(rewards_const_baseline, alpha=0.6, label='Constant Baseline = 30')
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.legend()
    plt.show()