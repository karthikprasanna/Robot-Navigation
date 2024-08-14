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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # ax1, ay1, ax2, ay2
        self.observation_space = spaces.Box(low=0, high=10, shape=(8,), dtype=np.float32)  # x1, y1, vx1, vy1, x2, y2, vx2, vy2
        self.damage = -1000
        self.penalty = -1000
        self.bonus = 1000
        self.mu = 0.2
        self.dt = 0.1  # Time step
        self.robot_radius = 0.15
        self.obstacle_radius = 0.05
        self.min_distance_between_robots = 0.5  # 2 * radius of a robot + safety margin
        self.min_distance_from_obstacle = 0.2  # robot_radius + obstacle_radius 

        self.init_positions = [(3, 9), (8, 2)]
        self.target_positions = [(8, 1), (3, 10)]
        self.obstacles = [(3, 7), (5, 7), (2, 5), (5, 5), (8, 5), (5, 3), (7, 3)]
        self.fig, self.ax = None, None  # Initialize here for clarity

    def seed(self, seed=None):
        np.random.seed(seed)

    def clip(self, x):
        if x > 10:
            return 10
        elif x < 0:
            return 0
        return x

    def robots_impact(self, vx1, vy1, vx2, vy2, x1, y1, x2, y2):
        return vx1, vy1, vx2, vy2

    def obstacle_impact(self, vx, vy, x, y):
        x = self.clip(x)
        y = self.clip(y)
        return vx, vy, x, y

    def wall_impact(self, vx, vy, x, y):
        if x < 0 or x > 10:
            vx = -vx
            x = self.clip(x)
        if y < 0 or y > 10:
            vy = -vy
            y = self.clip(y)
        return vx, vy, x, y            

    def step(self, action):
        ax1, ay1, ax2, ay2 = action
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = self.state
        
        # Update velocities and positions
        new_vx1, new_vy1 = vx1 + self.dt * (ax1 - self.mu * vx1), vy1 + self.dt * (ay1 - self.mu * vy1)
        new_vx2, new_vy2 = vx2 + self.dt * (ax2 - self.mu * vx2), vy2 + self.dt * (ay2 - self.mu * vy2)
        new_x1, new_y1 = x1 + new_vx1 * self.dt, y1 + new_vy1 * self.dt
        new_x2, new_y2 = x2 + new_vx2 * self.dt, y2 + new_vy2 * self.dt

        reward = 0
        crashed = False

        # Collision between robots
        if np.linalg.norm([new_x1 - new_x2, new_y1 - new_y2]) < self.min_distance_between_robots:
            crashed = True
            new_vx1, new_vy1, new_vx2, new_vy2 = self.robots_impact(new_vx1, new_vy1, new_vx2, new_vy2, new_x1, new_y1, new_x2, new_y2)
            reward += 2 * self.damage  

        else:
            # Collision with wall
            if new_x1 < 0 or new_x1 > 10 or new_y1 < 0 or new_y1 > 10:
                # crashed = True
                # reward += self.damage  
                new_vx1, new_vy1, new_x1, new_y1 = self.wall_impact(new_vx1, new_vy1, new_x1, new_y1)

            # Collision with obstacle
            elif any(np.linalg.norm([new_x1 - ox, new_y1 - oy]) < self.min_distance_from_obstacle for ox, oy in self.obstacles):
                crashed = True
                new_vx1, new_vy1, new_x1, new_y1 = self.obstacle_impact(new_vx1, new_vy1, new_x1, new_y1)
                reward += self.damage  

            # Collision with wall
            if new_x2 < 0 or new_x2 > 10 or new_y2 < 0 or new_y2 > 10:
                # crashed = True
                # reward += self.damage
                new_vx2, new_vy2, new_x2, new_y2 = self.wall_impact(new_vx2, new_vy2, new_x2, new_y2)

            # Collision with obstacle
            elif any(np.linalg.norm([new_x2 - ox, new_y2 - oy]) < self.min_distance_from_obstacle for ox, oy in self.obstacles):
                crashed = True
                new_vx2, new_vy2, new_x2, new_y2 = self.obstacle_impact(new_vx2, new_vy2, new_x2, new_y2)
                reward += self.damage
                        
        divergence = -np.linalg.norm([new_x1 - self.target_positions[0][0], new_y1 - self.target_positions[0][1]]) \
                     -np.linalg.norm([new_x2 - self.target_positions[1][0], new_y2 - self.target_positions[1][1]])

        reached1 = np.isclose(new_x1, self.target_positions[0][0]) and np.isclose(new_y1, self.target_positions[0][1]) 
        reached2 = np.isclose(new_x2, self.target_positions[1][0]) and np.isclose(new_y2, self.target_positions[1][1])

        stopped1 = False
        stopped2 = False

        if reached1:
            stopped1 = np.isclose(new_vx1, 0) and np.isclose(new_vy1, 0)
            # reward += self.bonus
            if not stopped1:
                divergence += -np.linalg.norm([new_vx1, new_vy1]) * self.penalty
                # reward += 3 * self.bonus

        if reached2:
            stopped2 = np.isclose(new_vx2, 0) and np.isclose(new_vy2, 0)
            # reward += self.bonus
            if not stopped2:
                divergence += -np.linalg.norm([new_vx2, new_vy2]) * self.penalty
                # reward += 3 * self.bonus

        reward += divergence

        # crashed = False
        terminated = (reached1 and reached2 and stopped1 and stopped2) or crashed
        self.state = (new_x1, new_y1, new_vx1, new_vy1, new_x2, new_y2, new_vx2, new_vy2)
        return np.array(self.state), reward, terminated, {}

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

         # Add labels for ax, ay, vx, vy of robot 1
        self.ax.text(self.state[0] + 0.3, self.state[1], f'ax1: {self.state[2]:.2f}\nay1: {self.state[3]:.2f}\n'
                                                      f'vx1: {self.state[2]:.2f}\nvy1: {self.state[3]:.2f}', color='blue')

         # Add labels for ax, ay, vx, vy of robot 2
        self.ax.text(self.state[4] + 0.3, self.state[5], f'ax2: {self.state[6]:.2f}\nay2: {self.state[7]:.2f}\n'
                                                      f'vx2: {self.state[6]:.2f}\nvy2: {self.state[7]:.2f}', color='red')

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
            return

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

        
