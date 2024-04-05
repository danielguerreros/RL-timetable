
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

passenger_path = "data/line1/passenger_dataframe_direction1.csv"
traffic_path = "data/line1/traffic-1.csv"

Line = BusLine(passenger_path,traffic_path)
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


fig,axs = plt.subplots()


axs.plot(history["Time"], history["Reward"],label="Agent")
#plt.plot(action_B["Time"], action_B["Reward"], label='Action 1')
axs.set_xlabel('Time')
axs.set_ylabel('Reward')
axs.legend()

#axs[1].tight_layout() 




rewards = []
Line = BusLine(passenger_path,traffic_path)
current_minute = Line.current_minute
history = pd.DataFrame(columns=["Time","Action","Reward"])
while Line.current_minute<Line.last_minute:
    #action = get_action(state)
    if current_minute%10==0:
        action = 1
    else:
        action = 0 
    
    reward,new_state = Line.update_environment(action)
    current_minute+=1
    history = history._append({"Time": current_minute,"Action":action,"Reward":reward},ignore_index=True)
    #fig,ax = Line.plot()
    #fig.savefig(f"plots/{current_minute}.png")




axs.plot(history["Time"], history["Reward"],label="No Agent")
#plt.plot(action_B["Time"], action_B["Reward"], label='Action 1')
axs.set_xlabel('Time')
axs.set_ylabel('Reward')
axs.set_title('Reward Over Time New Scenario')
axs.legend()
axs.set_xticks(axs.get_xticks(), axs.get_xticklabels(), rotation=45, ha='right') 
#axs[0].tight_layout() 
# Plotting

#plt.plot(action_B["Time"], action_B["Reward"], label='Action 1')


plt.savefig("joint_plots_test.png",dpi=100)
plt.show()