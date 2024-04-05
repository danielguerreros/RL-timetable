import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
from environment import BusLine

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def train():
    state_size = 5
    action_size = 2  # 0 for no departure, 1 for departure
    agent = DQN(state_size, action_size)
    target_agent = DQN(state_size, action_size)

    # optimizer
    optimizer = optim.Adam(agent.parameters(), lr=0.001)

    #  hyperparameters
    gamma = 0.4  # discount factor
    epsilon = 1.0  # exploration rate
    epsilon_decay = 0.995  # decay rate
    epsilon_min = 0.01  # min exploration rate
    num_episodes = 10
    batch_size = 64
    capacity = 10000
    replay_buffer = ReplayBuffer(capacity)
    target_update_frequency = 1
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
            total_reward += reward

            # store the experience in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # sampling a random minibatch of transitions from the replay buffer
            if len(replay_buffer) > batch_size:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)

                # to tensors
                batch_state = torch.tensor(batch_state, dtype=torch.float32)
                batch_action = torch.tensor(batch_action, dtype=torch.long)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
                batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32)
                batch_done = torch.tensor(batch_done, dtype=torch.float32)

                # q-values
                q_values = agent(batch_state)
                q_values_next = target_agent(batch_next_state).detach()

                # target q-values
                target_q_values = batch_reward + gamma * (1 - batch_done) * torch.max(q_values_next, dim=1)[0]

                # loss
                loss = nn.MSELoss()(q_values.gather(1, batch_action.unsqueeze(1)), target_q_values.unsqueeze(1))

                # backtracking
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        # exploration rate decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # target network update periodically
        if episode % target_update_frequency == 0:
            target_agent.load_state_dict(agent.state_dict())

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    torch.save(agent.state_dict(), 'deep_q_network.pth')