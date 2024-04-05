
import torch 
import torch.nn as nn
from environment import BusLine
import matplotlib.pyplot as plt
import pandas as pd 


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

state = [1, 2, 3, 4, 5]
agent.load_state_dict(torch.load('q_network.pth'))


Line = BusLine("data/line1/passenger_dataframe_direction0.csv","data/line1/traffic-0.csv")
current_minute = Line.current_minute
history = pd.DataFrame(columns=["Time","Action","Reward"])
while current_minute < Line.last_minute:
    # action from agent( network)

    state_tensor = torch.tensor(state, dtype=torch.float32)
    q_values = agent(state_tensor)
    action = torch.argmax(q_values).item()

    #  environment update and reward
    reward, new_state = Line.update_environment(action)
    history = history._append({"Time": current_minute,"Action":action,"Reward":reward},ignore_index=True)

    current_minute += 1


# Filter DataFrame by Action
action_A = history[history["Action"] == 0]
action_B = history[history["Action"] == 1]




plt.plot(history["Time"], history["Reward"])
#plt.plot(action_B["Time"], action_B["Reward"], label='Action 1')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Reward over Time (Q Algo)')
plt.legend()
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.savefig("New_Scenario.png",dpi=100)
plt.show()