from ppo import *
from env import *

env = GroundRobotsEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# Load the best model
actor = Actor(state_dim, action_dim, action_bound)
actor.load_state_dict(torch.load('Models/best/ppo_best.pt')['actor'])

critic = Critic(state_dim)
critic.load_state_dict(torch.load('Models/best/ppo_best.pt')['critic'])

agent = PPOAgent(state_dim, action_dim, action_bound, actor, critic)
agent.inference(env, max_steps=200)
