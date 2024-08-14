import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# Ensure the correct import of your environment
from env import GroundRobotsEnv

# Helper function to initialize weights of neural networks
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# Actor Network for SAC
class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(SACActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * action_dim)  # Output both mean and log_std
        )
        self.action_bound = action_bound
        self.net.apply(init_weights)

    def forward(self, state):
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std

# Critic Network for SAC
class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SACCritic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q1.apply(init_weights)
        self.q2.apply(init_weights)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Soft Actor-Critic Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, alpha=0.2):
        self.actor = SACActor(state_dim, action_dim, action_bound)
        self.critic = SACCritic(state_dim, action_dim)
        self.critic_target = SACCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        self.memory = ReplayBuffer()

        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = alpha

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        action = torch.tanh(action) * self.actor.action_bound
        return action.detach().numpy()[0], log_prob.detach().numpy()

    def update_parameters(self, batch_size):
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(done).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor(next_state)
            next_state_action = torch.tanh(next_state_action) * self.actor.action_bound
            q1_next, q2_next = self.critic_target(next_state, next_state_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_state_log_pi
            next_q_value = reward + self.gamma * (1 - done) * min_q_next

        q1, q2 = self.critic(state, action)
        q1_loss = torch.nn.functional.mse_loss(q1, next_q_value)
        q2_loss = torch.nn.functional.mse_loss(q2, next_q_value)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_pi = self.actor(state)
        new_actions = torch.tanh(new_actions) * self.actor.action_bound
        q1_new, q2_new = self.critic(state, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, env, episodes=100, steps_per_episode=200, batch_size=64):
        total_rewards = []
        best_reward = float('-inf')
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            for step in range(steps_per_episode):
                action, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                episode_reward += reward

                if len(self.memory) > batch_size:
                    self.update_parameters(batch_size)

                state = next_state
                if done:
                    break

            total_rewards.append(episode_reward)
            best_reward = max(best_reward, episode_reward)
            print(f"Episode: {episode + 1}, Total Reward: {episode_reward}")

        # Compute Average Rewards and Regrets
        avg_rewards = [np.mean(total_rewards[max(0, i-50):i+1]) for i in range(episodes)]
        ins_regret = [best_reward - r for r in total_rewards]
        cum_regret = np.cumsum(ins_regret)

        return total_rewards, avg_rewards, ins_regret, cum_regret

    def inference(self, env, max_steps=200):
        state = env.reset()
        for step in range(max_steps):
            action, _ = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            env.render(mode='rgb_array', frame_index=step)  # Save each frame
            state = next_state
            if done:
                break

        env.make_video_from_frames("simulation.mp4")

        
        
    

if __name__ == '__main__':
    # Initialize the environment
    env = GroundRobotsEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    # Initialize the agent
    agent = SACAgent(state_dim, action_dim, action_bound)

    # Train the agent
    total_rewards, avg_rewards, ins_regret, cum_regret = agent.train(env)
    agent.inference(env, max_steps=200)

    # Plot the results
    def plot_results(avg_rewards, ins_regret, cum_regret):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(avg_rewards, label='Average Reward')
        plt.title('Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(ins_regret, label='Instantaneous Regret')
        plt.title('Instantaneous Regret per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Regret')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(cum_regret, label='Cumulative Regret')
        plt.title('Cumulative Regret per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Regret')
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        plt.savefig("results.png")

    plot_results(avg_rewards, ins_regret, cum_regret)