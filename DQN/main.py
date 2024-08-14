import numpy as np
import DQN
import env
import matplotlib.pyplot as plt


env = env.RobotEnv()
#self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=4,
#                 max_size=1000000, layer1_size=64, layer2_size=256, layer3_size=512, layer4_size=1024, batch_size=64
robots = DQN.Agent(alpha=0.02, beta=0.0005, input_dims=[8], tau=100, env=env, gamma=0.99, n_actions=625, max_size=1000000, layer1_size=64, layer2_size=256, layer3_size=512, layer4_size=1024, batch_size=64)


np.random.seed(0)

instantaneous_regret = []
num_episodes = 150
num_iterations = 500
# num_episodes = 10
# num_iterations = 10

if __name__ == '__main__':
    frame_index = 0
    for i in range(num_episodes):
        states = env.reset()
        done = False
        discounted_rewards= 0
        num_iterations = 500
        print("episode no. ", i)
        print("------------------")
        while not done and num_iterations > 0:
            action = robots.choose_action(states)
            new_states, reward, done, info = env.step(action)
            robots.remember(states, action, reward, new_states, int(done))
            robots.learn()
            states = new_states
            discounted_rewards+= reward  # gamma = 1
            num_iterations -= 1
            # if i > 3 * num_iterations / 4:
            #     env.render()
            frame_index += 1
            if num_episodes - i < 2:
                env.render(frame_index=frame_index)
            

        instantaneous_regret.append(discounted_rewards)

        print('episode ', i, 'loss %.2f' % discounted_rewards, end='\r')

    env.make_video_from_frames("simulation.mp4", fps=30)
    # Plotting Instantaneous Episodic Regret
    plt.figure()  # Explicitly create a new figure
    plt.plot(-np.array(instantaneous_regret))
    plt.ylabel('Instantaneous Episodic Regret')
    plt.xlabel('Episode')
    plt.title('Instantaneous Episodic Regret vs Episode for DQN Algorithm')
    plt.savefig('Instantaneous Episodic Regret vs Episode.png')
    plt.show()
    plt.pause(5)

    # Plotting Cumulative Regret
    cumulative_regret = np.cumsum(-np.array(instantaneous_regret))
    plt.figure()  # Explicitly create a new figure
    plt.plot(cumulative_regret)
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Episode')
    plt.title('Cumulative Regret vs Episode for DQN Algorithm')
    plt.savefig('Cumulative Regret vs Episode.png')
    plt.show()
    plt.pause(5)