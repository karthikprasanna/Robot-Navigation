import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import *
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=0.1)
        m.bias.data.fill_(0.01)

# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, std=0.1):
        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.ReLU6()
        )
        
        self.mu_head = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        self.sigma_head = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softplus()
        )
        
        self.action_bound = action_bound
        self.apply(init_weights)

    def forward(self, state):
        x = self.layers(state)
        mu = self.mu_head(x) * self.action_bound
        sigma = self.sigma_head(x)
        return mu, sigma

# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.apply(init_weights)

    def forward(self, state):
        value = self.layers(state)
        return value

# Define the PPO agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, action_bound, actor=None, critic=None):
        if actor is None:
            self.actor = Actor(state_dim, action_dim, action_bound)
        else:
            self.actor = actor
            
        if critic is None:
            self.critic = Critic(state_dim)
        else:
            self.critic = critic
            
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.002)
        self.gamma = 0.99
        self.epsilon_clip = 0.2
        self.critic_discount = 0.5
        self.entropy_beta = 0.001

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        mu, sigma = self.actor(state)

        dist = MultivariateNormal(mu, torch.diag_embed(sigma))
        action = dist.sample()
        action = action.clamp(-self.actor.action_bound, self.actor.action_bound)
        
        return action.detach().numpy()[0], dist.log_prob(action).detach().numpy()

    def update(self, states, actions, log_probs, returns, advantages):
        # Actor's probability ratio
        mu, sigma = self.actor(states)
        dist = MultivariateNormal(mu, torch.diag_embed(sigma))
        new_log_probs = dist.log_prob(actions)

        # Clipped Probability ratio
        ratio = torch.exp(new_log_probs - torch.FloatTensor(log_probs))
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
        actor_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Critic's loss
        values = self.critic(states).reshape(-1)
        
        # Change the dtype of returns and values to torch.DoubleTensor
        values, returns = values.type(torch.DoubleTensor), returns.type(torch.DoubleTensor)
        critic_loss = F.mse_loss(values, returns)

        # Entropy regularization
        entropy = dist.entropy().mean()

        loss = actor_loss - self.critic_discount * critic_loss + self.entropy_beta * entropy

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
    
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        
    def train(self, env, episodes=1000, steps_per_episode=200, save_every=50):
        self.actor.train()
        self.critic.train()
        
        total_rewards, avg_rewards = [], []
        
        for episode in range(episodes):
            self.state = env.reset()
            rewards = []
            states = []
            actions = []
            log_probs = []
            
            for step in range(steps_per_episode):
                action, log_prob = self.select_action(self.state)
                next_state, reward, done, _ = env.step(action)
                
                states.append(self.state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                
                self.state = next_state
                
                if done:
                    break
                
            states = np.array(states)
            actions = np.array(actions)
            log_probs = np.array(log_probs)
                
            returns = []
            advantages = []
            R = 0
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
                
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            for t in range(len(rewards)):
                value = self.critic(torch.FloatTensor(states[t].reshape(1, -1)))
                advantage = returns[t] - value.detach().numpy()[0, 0]
                advantages.append(advantage)
                
            advantages = torch.tensor(advantages)
                      
            self.update(torch.FloatTensor(states), torch.FloatTensor(actions), torch.FloatTensor(log_probs), returns, advantages)
            
            tot_reward = sum(rewards)
            avg_reward = tot_reward / len(rewards)
            
            print(f"Episode: {episode + 1}, Total Reward: {tot_reward}")
            total_rewards.append(tot_reward)
            avg_rewards.append(avg_reward)
            
            if (episode + 1) % save_every == 0:
                checkpoint = {
                    "epoch": episode + 1,
                    "reward": tot_reward,
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "optimizer_actor": self.optimizer_actor.state_dict(),
                    "optimizer_critic": self.optimizer_critic.state_dict()
                }
                torch.save(checkpoint, f"Models/checkpoints/ppo_{episode + 1}.pt")
                
            # Save the best model
            if tot_reward == max(total_rewards):
                checkpoint = {
                    "epoch": episode + 1,
                    "reward": tot_reward,
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "optimizer_actor": self.optimizer_actor.state_dict(),
                    "optimizer_critic": self.optimizer_critic.state_dict()
                }
                
                torch.save(checkpoint, "Models/best/ppo_best.pt")
            
        return total_rewards, avg_rewards
    
    # Function to plot the control input and state trajectories
    def inference(self, env, max_steps=200):
        self.state = env.reset()
        
        reward = 0
        for step in range(max_steps):
            action, _ = self.select_action(self.state)
            next_state, r, done, _ = env.step(action)
            
            self.state = next_state
            reward = r + self.gamma * reward
            
            env.render(frame_index=step)
            
            if done:
                break
            
        env.make_video_from_frames("simulation.mp4")
        return reward

# Create the environment and agent
if __name__ == '__main__':
    env = GroundRobotsEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    print("State Dimensions: ", state_dim)
    print("Action Dimensions: ", action_dim)
    print("Action Bound: ", action_bound)

    num_episodes = 1000
    steps_per_episode = 200

    agent = PPOAgent(state_dim, action_dim, action_bound)

    # Train the agent
    total_rewards, avg_rewards = agent.train(env, episodes=num_episodes, steps_per_episode=steps_per_episode)

    best_reward  = max(total_rewards)
    best_episode = total_rewards.index(best_reward) + 1
    x = np.linspace(1, best_episode + 1, best_episode)

    # Plot average reward per episode
    plt.figure(1)
    plt.plot(x, avg_rewards[:best_episode])
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Average Reward per Episode")
    plt.show()

    # Plot instantaneous regret
    ins_regret = [best_reward - r for r in total_rewards[:best_episode]]
    plt.figure(2)
    plt.plot(x, ins_regret)
    plt.xlabel("Episodes")
    plt.ylabel("Instantaneous Regret")
    plt.title("Instantaneous Regret per Episode")
    plt.show()

    # Plot cumulative regret
    cum_regret = np.cumsum(ins_regret)
    plt.figure(3)
    plt.plot(x, cum_regret)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret per Episode")
    plt.show()
