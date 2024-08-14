import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import threading
import cv2
import os

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # ax1, ay1, ax2, ay2
        # Only possible actions are -1, -0.5, 0, 0.5, 1 for each robot.
        self.action_space = spaces.Discrete(625)
        self.observation_space = spaces.Box(low=0, high=10, shape=(8,), dtype=np.float32)  # x1, y1, vx1, vy1, x2, y2, vx2, vy2
        self.mu = 0.2
        self.dt = 1  # Time step
        self.robot_radius = 0.15
        self.obstacle_radius = 0.05
        self.min_distance_between_robots = 0.5  # 2 * radius of a robot + safety margin
        self.min_distance_from_obstacle = 0.2  # robot_radius + obstacle_radius 

        self.init_positions = [(3, 9), (8, 2)]
        self.target_positions = [(8, 1), (3, 10)]
        self.obstacles = [(3, 7), (5, 7), (2, 5), (5, 5), (8, 5), (5, 3), (7, 3)]
        self.fig, self.ax = None, None  # Initialize here for clarity

    def step(self, action):
        ax1 = [-1, -0.5, 0, 0.5, 1][action // 125]
        ay1 = [-1, -0.5, 0, 0.5, 1][(action // 25) % 5]
        ax2 = [-1, -0.5, 0, 0.5, 1][(action // 5) % 5]
        ay2 = [-1, -0.5, 0, 0.5, 1][action % 5]
        # ax1, ay1, ax2, ay2 = action
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = self.state
        
        # Update velocities and positions
        new_vx1, new_vy1 = vx1 + self.dt * (ax1 - self.mu * vx1), vy1 + self.dt * (ay1 - self.mu * vy1)
        new_vx2, new_vy2 = vx2 + self.dt * (ax2 - self.mu * vx2), vy2 + self.dt * (ay2 - self.mu * vy2)
        new_x1, new_y1 = x1 + new_vx1 * self.dt, y1 + new_vy1 * self.dt
        new_x2, new_y2 = x2 + new_vx2 * self.dt, y2 + new_vy2 * self.dt

        # Collision and boundary checks
        if new_x1 <= 0 or new_x1 >= 10 or new_y1 <= 0 or new_y1 >= 10 or \
           new_x2 <= 0 or new_x2 >= 10 or new_y2 <= 0 or new_y2 >= 10:
            done = False  # Out of bounds
            if(new_x1 <= 0 or new_x1 >= 10):
                new_vx1 = -1*new_vx1
            if(new_y1 <= 0 or new_y1 >= 10):
                new_vy1 = -1*new_vy1
            if(new_x2 <= 0 or new_x2 >= 10):
                new_vx2 = -1*new_vx2
            if(new_y2 <= 0 or new_y2 >= 10):
                new_vy2 = -1*new_vy2
            
            
            new_x1 = np.clip(new_x1, 0, 10)
            new_y1 = np.clip(new_y1, 0, 10)
            new_x2 = np.clip(new_x2, 0, 10)
            new_y2 = np.clip(new_y2, 0, 10)
            reward = -np.linalg.norm([new_x1 - self.target_positions[0][0], new_y1 - self.target_positions[0][1]]) \
                     -np.linalg.norm([new_x2 - self.target_positions[1][0], new_y2 - self.target_positions[1][1]])  # Penalty for going out of bounds
            #clip the position to be within the boundary
            
        elif np.linalg.norm([new_x1 - new_x2, new_y1 - new_y2]) < self.min_distance_between_robots:
            done = False
            reward = -1000  # Collision between robots
        elif any(np.linalg.norm([new_x1 - ox, new_y1 - oy]) < self.min_distance_from_obstacle or
                 np.linalg.norm([new_x2 - ox, new_y2 - oy]) < self.min_distance_from_obstacle for ox, oy in self.obstacles):
            done = False
            reward = -1000  # Collision with obstacle
        # elif it is close to one of the target, give a reward of 10000
        elif np.isclose(new_x1, self.target_positions[0][0]) and np.isclose(new_y1, self.target_positions[0][1]) \
                or np.isclose(new_x2, self.target_positions[1][0]) and np.isclose(new_y2, self.target_positions[1][1]):
            done = False
            reward = 10000 
        else:
            reward = -np.linalg.norm([new_x1 - self.target_positions[0][0], new_y1 - self.target_positions[0][1]]) \
                     -np.linalg.norm([new_x2 - self.target_positions[1][0], new_y2 - self.target_positions[1][1]])
            done = np.isclose(new_x1, self.target_positions[0][0]) and np.isclose(new_y1, self.target_positions[0][1]) \
                   and np.isclose(new_x2, self.target_positions[1][0]) and np.isclose(new_y2, self.target_positions[1][1]) \
                    and np.isclose(new_vx1, 0) and np.isclose(new_vx2, 0) and np.isclose(new_vy1, 0) and np.isclose(new_vy2, 0)
            if(done):
                reward = 20000

        self.state = (new_x1, new_y1, new_vx1, new_vy1, new_x2, new_y2, new_vx2, new_vy2)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([*self.init_positions[0], 0, 0, *self.init_positions[1], 0, 0], dtype=np.float32)
        return np.array(self.state)

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
            self.ax.add_patch(Circle(obstacle, self.obstacle_radius, color='gray'))

        self.ax.add_patch(Circle((self.state[0], self.state[1]), self.robot_radius, color='blue'))
        self.ax.add_patch(Circle((self.state[4], self.state[5]), self.robot_radius, color='red'))
        
        self.ax.plot(*self.target_positions[0], 'b*', markersize=10)
        self.ax.plot(*self.target_positions[1], 'r*', markersize=10)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if not os.path.exists("temp_folder"):
            os.makedirs("temp_folder")

        plt.savefig(f"temp_folder/frame_{frame_index}.png")

    def make_video_from_frames(self, output_video_path, fps=30):
        folder_path = "temp_folder"
        # Get all image files from the folder, assuming they are named sequentially
        images = [img for img in sorted(os.listdir(folder_path)) if img.endswith(".png")]
        if not images:
            raise ValueError("No images found in the folder")

        # Determine the width and height from the first image
        frame = cv2.imread(os.path.join(folder_path, images[0]))
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Write each frame to the video
        for image in images:
            frame = cv2.imread(os.path.join(folder_path, image))
            video.write(frame)

        # Release the VideoWriter
        video.release()

        # delete the frames after creating the video
        for file in os.listdir("temp_folder"):
            os.remove(os.path.join("temp_folder", file))
