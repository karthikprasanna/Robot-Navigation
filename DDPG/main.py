import numpy as np
import DDPG
import environment
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DDPG as baseline_DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure


np.random.seed(0)
num_episodes = 100
num_iterations_ = 100
instantaneous_regret = []

env = environment.RobotEnv()
baseline_env = make_vec_env(environment.RobotEnv, n_envs=1)
robots = DDPG.Agent(alpha=0.1, beta=0.2, input_dims=[8], tau=0.1, env=env, max_size=3000,
              batch_size=32,  layer1_size=32, layer2_size=32, n_actions=4, gamma=1, sigma = 0.3)

model = baseline_DDPG('MlpPolicy', baseline_env, verbose=1,  learning_rate=0.1, gamma=1)
# model.set_logger(logger)

def train_baseline(num_episodes=num_episodes, num_iterations_=num_iterations_):
    # Train the model
    for i in range(num_episodes):
        model.learn(total_timesteps=num_iterations_)
        print('episode ', i, 'ended', end='\r')

    # Evaluate the model
    frame_index = 0
    for i in range(5):
        states = env.reset()
        done = False
        discounted_rewards= 0
        num_iterations = num_iterations_
        while not done and num_iterations > 0:
            action, _states = model.predict(states, deterministic=False)
            new_states, reward, done, info = env.step(action)
            states = new_states
            discounted_rewards+= reward  # gamma = 1
            num_iterations -= 1
            frame_index += 1
            env.render(frame_index=frame_index)

        instantaneous_regret.append(discounted_rewards)
        print('Evaluation: episode ', i, 'loss %.2f' % discounted_rewards, end='\r')


def train_DDPG(num_episodes=num_episodes, num_iterations=num_iterations_):
    frame_index = 0
    for i in range(num_episodes):
        states = env.reset()
        done = False
        discounted_rewards= 0
        num_iterations = num_iterations_
        while not done and num_iterations > 0:
            action = robots.choose_action(states)
            new_states, reward, done, info = env.step(action)
            robots.remember(states, action, reward, new_states, int(done))
            robots.learn()
            states = new_states
            discounted_rewards+= reward  # gamma = 1
            num_iterations -= 1
            frame_index += 1
            if num_episodes - i <= 2:
                env.render(frame_index=frame_index)

        instantaneous_regret.append(discounted_rewards)    
        print('episode ', i, 'loss %.2f' % discounted_rewards, end='\r')


if __name__ == '__main__':
    train_DDPG()
    # train_baseline()

    # make a video out of the frames using openCV
    env.make_video_from_frames("simulation.mp4", fps=30)
    
    # Plotting Instantaneous Episodic Regret
    plt.figure()  # Explicitly create a new figure
    plt.plot(-np.array(instantaneous_regret))
    plt.ylabel('Instantaneous Episodic Regret')
    plt.xlabel('Episode')
    plt.title('Instantaneous Episodic Regret vs Episode')
    plt.savefig('Instantaneous Episodic Regret vs Episode.png')
    plt.show()
    plt.pause(5)

    # Plotting Cumulative Regret
    cumulative_regret = np.cumsum(-np.array(instantaneous_regret))
    plt.figure()  # Explicitly create a new figure
    plt.plot(cumulative_regret)
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Episode')
    plt.title('Cumulative Regret vs Episode')
    plt.savefig('Cumulative Regret vs Episode.png')
    plt.show()
    plt.pause(5)

    # Plotting Average Reward
    average_reward = np.array(instantaneous_regret) / num_iterations_
    plt.figure()  # Explicitly create a new figure
    plt.plot(average_reward)
    plt.ylabel('Average Reward')
    plt.xlabel('Episode')
    plt.title('Average Reward vs Episode')
    plt.savefig('Average Reward vs Episode.png')
    plt.show()
    plt.pause(5)

