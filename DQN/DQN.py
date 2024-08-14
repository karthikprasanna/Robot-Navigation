# Methods in DQN class
# Agent, learn, choose_action, remember, learn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

# use a target network to stabilize learning
class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=4,
                 max_size=1000000, layer1_size=64, layer2_size=256, layer3_size=512, layer4_size=1024, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.env = env
        self.memory = ReplayBuffer(max_size, input_dims)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.learn_step_counter = 0
        self.time_step = 0
        # use epsilon decay
        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_eval = DeepQNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, n_actions).to(self.device)
        self.q_target = DeepQNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(625)
        self.time_step += 1
        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.optimizer.zero_grad()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        action_save = action

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        done = torch.tensor(done, dtype=bool).to(self.device)

        # print the shapes of the tensors
        # print("state.shape, action.shape, reward.shape, new_state.shape, done.shape", state.shape, action.shape, reward.shape, new_state.shape, done.shape)

        # q_pred = self.q_eval.forward(state)
        # print("q_pred.shape", q_pred.shape)
        # q_pred = q_pred[action_save]
        q_values = self.q_eval.forward(state)
        # convert action into int
        q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        q_next = self.q_target.forward(new_state)

        q_next[done] = 0.0

        q_target = reward + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.loss(q_target, q_values).to(self.device)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.tau == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        

        
class DeepQNetwork(nn.Module):
    def __init__(self, alpha, input_dims, layer1_size, layer2_size, layer3_size, layer4_size, n_actions):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, layer1_size)
        self.fc2 = nn.Linear(layer1_size, layer2_size)
        self.fc3 = nn.Linear(layer2_size, layer3_size)
        self.fc4 = nn.Linear(layer3_size, layer4_size)
        self.fc5 = nn.Linear(layer4_size, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        q_value = self.fc1(state.float())
        q_value = F.leaky_relu(q_value)
        q_value = self.fc2(q_value.float())
        q_value = F.leaky_relu(q_value)
        q_value = self.fc3(q_value.float())
        q_value = F.leaky_relu(q_value)
        q_value = self.fc4(q_value.float())
        q_value = F.leaky_relu(q_value)
        q_value = self.fc5(q_value.float())
        
        return q_value.float()
    
class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size)
        # self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        # self.terminal_memory = np.zeros(self.mem_size)
        # Take terminal memory as a boolean array
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
    

        

