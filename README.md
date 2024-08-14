# README for Ground Robot Path Planning using Reinforcement Learning

## Project Overview

This project involves designing and implementing an optimal control strategy for two identical ground robots (G1 and G2) within a 10m x 10m arena, with static circular obstacles. The goal is to find the optimal control that drives the ground robots to their respective destinations in minimum time while maintaining a minimum distance of 0.2m from the static obstacles and 0.5m from each other. The environment is modeled with a known coefficient of friction and maximum acceleration input.

## Problem Statement

### Arena Configuration
- **Arena Size:** 10m x 10m
- **Initial Positions of Robots:**
  - G1: (3, 9)
  - G2: (8, 2)
- **Destination Coordinates:**
  - G1: (8, 1)
  - G2: (3, 10)
- **Static Obstacles:**
  - O1: (3, 7)
  - O2: (5, 7)
  - O3: (2, 5)
  - O4: (5, 5)
  - O5: (8, 5)
  - O6: (5, 3)
  - O7: (7, 3)
  - **Diameter of Obstacles:** 1m

### Robot Dynamics
- **State-Space Model:**
  - `dx/dt = vx`
  - `dvx/dt = ax - μ vx`
  - `dy/dt = vy`
  - `dvy/dt = ay - μ vy`
- **Coefficient of Friction:** `μ = 0.2`
- **Maximum Acceleration Input:** `1 m/s^2`


### Task
- **Objective:** Implement a reinforcement learning algorithm to control the ground robots' movement in the given environment.
- **Constraints:** 
  - Maintain a minimum distance of 0.2m from the obstacles.
  - Maintain a minimum distance of 0.5m from the other robot.
- **Algorithms to Consider:**
  - Deep Q-Learning (Discrete Action Space)
  - Deep Deterministic Policy Gradient (DDPG)
  - Proximal Policy Optimization (PPO)
  - Soft Actor-Critic (SAC)

## Setup and Requirements

### Prerequisites
To run the project, you will need to install the following libraries:

- Python 3.x
- NumPy
- OpenAI Gym
- TensorFlow or PyTorch
- Stable-Baselines3 (for pre-implemented RL algorithms)

You can install these libraries using pip:

```bash
pip install numpy gym tensorflow stable-baselines3
