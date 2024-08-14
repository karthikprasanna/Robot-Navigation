import os
import cv2
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym import spaces
from matplotlib.patches import Circle

class GroundRobotsEnv(gym.Env):
    def __init__(self, grid=10, robot_distance=0.3, obstacle_radius=0.1):
        super(GroundRobotsEnv, self).__init__()
        
        self.grid = grid
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)        # Acceleration inputs: (ax1, ay1, ax2, ay2)
        self.observation_space = spaces.Box(low=0, high=grid, shape=(8,), dtype=np.float32)     # State: (x1, y1, vx1, vy1, x2, y2, vx2, vy2)
        
        self.dt = 0.1                                                                           # Time step
        self.mu = 0.2                                                                           # Friction coefficient
        self.obstacles = [(3,7), (5,7), (2,5), (5,5), (8,5), (5,3), (7,3)]                      # Coordinates of static obstacles
        
        self.state = None
        self.goal_positions = [(8,1), (3,10)]                                                   # Destinations of G1 and G2
        
        self.robot_distance = robot_distance
        self.obstacle_distance = obstacle_radius
        
        self.fig, self.ax = None, None                                                          # Initialize here for clarity

    def reset(self):
        self.state = np.array([3, 9, 0, 0, 8, 2, 0, 0], dtype=np.float32)                       # Initial state: (x1, y1, vx1, vy1, x2, y2, vx2, vy2)
        return self.state.copy()

    def step(self, action):
        # Apply acceleration inputs to update velocities
        self.state[2:4] += action[0:2] - self.mu * self.state[2:4]
        self.state[6:8] += action[2:4] - self.mu * self.state[6:8]

        # Update positions using velocities to nearest integer
        self.state[0:2] = self.state[0:2] + self.state[2:4] * self.dt
        self.state[4:6] = self.state[4:6] + self.state[6:8] * self.dt

        # Ensure the robots stay within the boundaries of the arena
        self.state[0:2] = np.clip(self.state[0:2], 0, self.grid)
        self.state[4:6] = np.clip(self.state[4:6], 0, self.grid)
        
        reward = 0

        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            dist1 = np.linalg.norm(self.state[0:2] - obstacle)
            dist2 = np.linalg.norm(self.state[4:6] - obstacle)
            if dist1 <= self.obstacle_distance or dist2 <= self.obstacle_distance:
                reward += -1000
                done = False

        # Check for collision between robots
        robot_dist = np.linalg.norm(self.state[0:2] - self.state[4:6])
        if robot_dist <= self.robot_distance:
            reward += -1000
            done = False

        # Check if robots reached their destinations
        reached1 = np.allclose(self.state[0:2], self.goal_positions[0], atol=0.1) and np.allclose(self.state[4:6], self.goal_positions[1], atol=0.1)
        reached2 = np.allclose(self.state[0:2], self.goal_positions[1], atol=0.1) and np.allclose(self.state[4:6], self.goal_positions[0], atol=0.1)
        if reached1 or reached2:
            reward += 0
            done = True
        
        # Compute Euclidean distance between robots and their destinations
        else:
            dist1, dist2 = 0, 0
            if np.linalg.norm(self.state[0:2] - self.goal_positions[0]) <= np.linalg.norm(self.state[4:6] - self.goal_positions[0]):
                dist1 = np.linalg.norm(self.state[0:2] - self.goal_positions[0])
                dist2 = np.linalg.norm(self.state[4:6] - self.goal_positions[1])
                
            else:
                dist1 = np.linalg.norm(self.state[0:2] - self.goal_positions[1])
                dist2 = np.linalg.norm(self.state[4:6] - self.goal_positions[0])

            reward += (-dist1 + -dist2) * 100
            done = False

        return self.state.copy(), reward, done, {}

    # def render(self):
    #     fig, ax = plt.subplots()
    #     ax.set_xlim([0, 10])
    #     ax.set_ylim([0, 10])
    #     ax.set_aspect('equal', adjustable='box')

    #     # Plot obstacles
    #     for obstacle in self.obstacles:
    #         ax.add_patch(Circle(obstacle, radius=self.obstacle_distance/2, color='gray'))

    #     # Plot robots
    #     ax.add_patch(Circle(self.state[0:2], radius=self.robot_distance/2, color='red'))
    #     ax.add_patch(Circle(self.state[4:6], radius=self.robot_distance/2, color='blue'))

    #     ax.plot(*self.goal_positions[0], marker='x', color='green', markersize=10)
    #     ax.plot(*self.goal_positions[1], marker='x', color='green', markersize=10)

    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.title('Ground Robots Environment')
    #     plt.xticks(np.arange(0, 11, 1))
    #     plt.yticks(np.arange(0, 11, 1))
    #     plt.grid(True)
    #     plt.show()
        
    def render(self, mode='human', frame_index=0):
        if self.fig is None:  # Initialize the plot if it doesn't exist
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim([0, 10])
            self.ax.set_ylim([0, 10])
            self.ax.set_aspect('equal', adjustable='box')
            plt.ion()
            plt.show()

        # Clear the current content of the plot (if any)
        self.ax.cla()
        self.ax.set_xlim([0, 10])
        self.ax.set_ylim([0, 10])
        self.ax.grid(True)

        # Plot obstacles, robots, and targets
        for obstacle in self.obstacles:
            self.ax.add_patch(Circle(obstacle, self.obstacle_distance/2, color='gray'))

        self.ax.add_patch(Circle((self.state[0], self.state[1]), self.robot_distance/2, color='blue'))
        self.ax.add_patch(Circle((self.state[4], self.state[5]), self.robot_distance/2, color='red'))
        
        self.ax.plot(*self.goal_positions[0], marker='x', color='green', markersize=10)
        self.ax.plot(*self.goal_positions[1], marker='x', color='green', markersize=10)
        
        self.fig.canvas.draw()

        if mode == 'rgb_array':
            # Ensure the folder exists
            if not os.path.exists("temp_folder"):
                os.makedirs("temp_folder")
            # Save the current frame as an image file
            frame_path = os.path.join("temp_folder", f"frame_{frame_index:04d}.png")
            self.fig.savefig(frame_path)

        plt.pause(0.001)  # Pause to update the plot if in interactive mode

    def make_video_from_frames(self, output_video_path, fps=10):
        folder_path = "temp_folder"
        images = [img for img in sorted(os.listdir(folder_path)) if img.endswith(".png")]
        if not images:
            raise ValueError("No images found in the folder")

        frame = cv2.imread(os.path.join(folder_path, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for image in images:
            frame = cv2.imread(os.path.join(folder_path, image))
            video.write(frame)

        video.release()  # Finalize the video file
        print(f"Video saved to {output_video_path}")

        # Optionally clean up the folder
        for file in images:
            os.remove(os.path.join(folder_path, file))
