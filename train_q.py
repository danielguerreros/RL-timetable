from environment import BusLine
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x

state_size = 5
action_size = 2  # 0 for no departure, 1 for departure
agent = QNetwork(state_size, action_size)
optimizer = optim.Adam(agent.parameters(), lr=0.001)

# hyperparameters
gamma = 0.99  # discount factor
epsilon = 1.0  #  for exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01  # minimum exploration rate
num_episodes = 8

env = BusLine()


# training
for episode in range(num_episodes):

    state = env.reset()

    done = False
    total_reward = 0

    while not done:
        # epsilon-greedy
        if np.random.rand() <= epsilon:
            action = np.random.randint(action_size)
        else:
            with torch.no_grad():
                q_values = agent(torch.tensor(state, dtype=torch.float32))
                action = torch.argmax(q_values).item()

        # step function
        next_state, reward, done = env.step(action)

        # updating  q-values
        q_values = agent(torch.tensor(state, dtype=torch.float32))
        next_q_values = agent(torch.tensor(next_state, dtype=torch.float32))
        max_next_q_value = torch.max(next_q_values).item()
        target = reward + gamma * max_next_q_value
        target = torch.tensor(target, dtype=torch.float32)

        loss = nn.MSELoss()(q_values[action], target)

        # backtracking
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        state = next_state
        #print(f"reward: { reward}")

    # exploration rate decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

torch.save(agent.state_dict(), 'q_network.pth')

