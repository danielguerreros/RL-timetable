import pandas as pd
import numpy as  np
import torch
import warnings
import csv
import matplotlib.pyplot as plt 
import logging 
import re 
import matplotlib.image as mpimg

warnings.filterwarnings("ignore")

logging.basicConfig(filename='app.log',
                    filemode='w',
                    format='%(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Urban Planning")

logger = logging.getLogger('urbanGUI')

class Station:

    def __init__(self, station_number,all_passenger_info_path, first_minute):

        self.station_number = station_number  #Total number of stations
        self.init_first_minute_passengers = [] #Passengers waiting at the station when the train departs
        self.current_minute_passengers = [] #Passengers waiting at all  stations at the current minute
        self.next_minute_passengers = [] #Passengers at all stations in the next minute
        self.all_passenger_info_path= all_passenger_info_path
        self.all_passenger_info_dataframe =  pd.read_csv(all_passenger_info_path)
        self.all_station_all_minute_passenger = []  # All stations, passengers boarding every minute ([[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==1]],[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==2]],[],[],])
        self.first_minute  = first_minute
        self.current_minute = self.first_minute

        for i in range(self.station_number): 
            # Array containing all passenger information per station
            self.all_station_all_minute_passenger.append(self.all_passenger_info_dataframe[self.all_passenger_info_dataframe["Boarding station"]==i])#Every station, all minutes of

            # Passengers in the first minute at each station
            self.init_first_minute_passengers.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"]<=self.first_minute])
            
            # Passengers in the next minute at each station
            self.next_minute_passengers.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"] == (self.first_minute+1)])
        
        self.current_minute_passengers =  self.init_first_minute_passengers
        

    def forward_one_step(self):
        """Update current and next minute passengers at each station
        """

        # We join current passengers with next ones and update current time
        for i in range(self.station_number):
            self.current_minute_passengers[i]=pd.concat([self.current_minute_passengers[i],self.next_minute_passengers[i]])
        self.current_minute = self.current_minute + 1
        
        # We calculate again the passengers for next minute
        self.next_minute_passengers = []
        for i in range(self.station_number):  
            self.next_minute_passengers.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"] == (self.current_minute + 1)])




class Bus:
    def __init__(self,max_capacity, station_number,start_time):


        self.start_time =start_time #time of issue
        self.current_minute = self.start_time # current minute
        self.station_number = station_number # total number of stations
        self.state_str = ""
        self.max_capacity = max_capacity #Maximum number of people in the bus
        self.position = np.zeros(station_number) #Bus location
        self.position[0]=1 # initialize bus location to 0
        self.pass_position = np.zeros(station_number) #Stations through which vehicles pass for capacity calculations
        self.pass_position[0]=1
        self.passengers_on = pd.DataFrame(columns=passenger_columns) # information of passengers on the bus
        self.arrv_mark = 0 #terminate
        self.estimated_arrival_time = np.zeros(station_number)  #estimated time of arrival at a given station
        self.estimated_arrival_time[0] = start_time
        self.last_station = 0 #last station visited by the bus

        # We calculate an estimated arrival time using the traffic data
        for i in range(1, station_number):
            self.estimated_arrival_time[i] = start_time +sum(trf_con.iloc[(start_time // 15), 6:6+i]) 

            # If the time difference between stations is 0 we add one minute
            if  (self.estimated_arrival_time[i]-self.estimated_arrival_time[i-1])<=0:
                self.estimated_arrival_time[i]= self.estimated_arrival_time[i-1] + 1
            



    def update(self,stations:Station):
        #print(f"Bus time: {self.current_minute//60}:{self.current_minute%60}")
        arrival_times = self.estimated_arrival_time
        waiting_time = 0 

        if self.arrv_mark!=1:

            # We update the bus passengers info if the bus arrives at a station in the current minute
            if self.current_minute in arrival_times:
                
                # we find the station that the bus is at
                station_index = np.where(arrival_times==self.current_minute)[0][0]
                
                # We update the current position of the bus
                self.position = np.zeros(self.station_number)
                self.position[station_index] = 1
                self.pass_position[station_index] = 1

                self.last_station = station_index
                self.state_str = f"in station {station_index}"
                
                # If the bus has been to all stations then its done
                if sum(self.pass_position)==self.station_number:
                        self.arrv_mark=1
                        self.state_str = "Done for today"
                        
                
                # We get current people in the station
                people_in_station =  stations.current_minute_passengers[station_index]

                # Drop Passengers
                self.passengers_on = self.passengers_on._append(self.passengers_on[self.passengers_on["Alighting station"] == station_index])
                self.passengers_on = self.passengers_on.drop_duplicates(subset=["Label", 'Boarding time', 'Boarding station', 'Alighting station'], keep=False)

                # If there is enough room to fit everyone we empty the station s
                if len(people_in_station) + len(self.passengers_on) <= self.max_capacity:

                    # we add passengers
                    boarding_passengers = people_in_station
                    self.passengers_on = pd.concat([self.passengers_on, boarding_passengers])
                    
                    # station has now zero people
                    stations.current_minute_passengers[station_index] = pd.DataFrame(columns=passenger_columns)
                    
                   
                
                else:

                    # If there is not enough room we fit the bus to be full and leave people in the statoin
                    append_mark = (self.max_capacity - len(self.passengers_on))
                    boarding_passengers = people_in_station.iloc[:append_mark, :]


                    self.passengers_on = pd.concat([self.passengers_on, boarding_passengers])
                    
                    # station has some people
                    stations.current_minute_passengers[station_index] = people_in_station.iloc[append_mark:, :]
                   
                    
                # we update waiting time
                waiting_time += self.current_minute*len(boarding_passengers) - np.sum(boarding_passengers.iloc[:,4])
            else:
                self.state_str = f"travelling between station  {self.last_station} and {self.last_station+1}" 


        self.current_minute=self.current_minute+1
        assert len(self.passengers_on)<= self.max_capacity, "BUS HAS MORE PEOPLE THAN IT SHOULD"
        return waiting_time
    


class BusLine:

    
    BETA = 1
    MIN_INTERVAL =10
    MAX_INTERVAL =25
    EARLY_STOP = 60
    C = 50
    OMEGA = 0.009
    def __init__(self, passenger_data_path:str = "data/line3/passenger_dataframe_direction1.csv", traffic_data_path:str="data/line3/traffic-1.csv" ,c_max:int=47,beta:int=1,first_minute:str="6:30",last_minute:str="22:00") -> None:
        

        self.first_time=first_minute
        self.last_time=last_minute
        self.passenger_data = pd.read_csv(passenger_data_path)
        self.traffic_data = pd.read_csv(traffic_data_path)
        self.num_stations = self.passenger_data['Boarding station'].nunique()
        self.first_minute = (int(self.first_time[:-3]) - int(self.traffic_data.iloc[0, 0])) * 60 + (int(self.first_time[-2:]) - int(self.traffic_data.iloc[0, 1]))
        self.last_minute = (int(self.last_time[:-3]) - int(self.traffic_data.iloc[0, 0])) * 60 + (int(self.last_time[-2:]) - int(self.traffic_data.iloc[0, 1]))
        self.current_minute = self.first_minute 

        self.buses_on_road = []
        self.stations = Station(self.num_stations,passenger_data_path,self.first_minute)
        self.c_max_m = 0 
        self.c_max = c_max
        self.e_m = 1*(self.num_stations-1)*self.c_max
        self.waiting_time = 0
        self.beta=beta
        self.current_carrying_capacity = 0
        self.last_bus_departure = self.current_minute
        self.current_no_waiting_passengers = 0
    
    def reset(self,passenger_data_path="data/line3/passenger_dataframe_direction1.csv"):
        
        self.current_minute = self.first_minute 
        self.buses_on_road = []
        self.stations = Station(self.num_stations,passenger_data_path,self.first_minute)
        self.waiting_time = 0
        self.current_carrying_capacity = 0
        self.last_bus_departure = self.current_minute
        self.current_no_waiting_passengers = 0
        initial_state = self.update_environment(0)[1]
        return initial_state

    
    def get_reward(self,action):
        
        reward = 0 
        if action == 1:
            reward = self.current_carrying_capacity/self.e_m - self.current_no_waiting_passengers/(self.num_stations*self.C)
        elif action == 0: 
            if self.current_no_waiting_passengers==0:
                reward = 1 - self.current_carrying_capacity/self.e_m - self.current_no_waiting_passengers/(self.num_stations*self.C) 
            else:
                reward = 1 - self.current_carrying_capacity/self.e_m - self.current_no_waiting_passengers/(self.num_stations*self.C) - (self.waiting_time/self.current_no_waiting_passengers)*self.OMEGA
        return -self.current_no_waiting_passengers + self.current_carrying_capacity
    
    def update_environment(self, action):

        logger.debug(f"\nCurrent time {self.current_minute//60}:{self.current_minute%60}\n")
        
        actual_action = action
        """
        if (( self.last_minute - self.current_minute) <self.EARLY_STOP):
            actual_action = 0
        elif ((self.current_minute == self.first_minute) or (self.current_minute == self.last_minute) or ((self.current_minute - self.last_bus_departure)>=self.MAX_INTERVAL) ):
            actual_action = 1
            self.last_bus_departure = self.current_minute
        elif (action ==1 ) and ((self.current_minute - self.last_bus_departure)<=self.MIN_INTERVAL):
            actual_action = 0
        elif action ==1:
            self.last_bus_departure = self.current_minute
        """
        if actual_action==1:
            self.last_bus_departure = self.current_minute
        

        
         
        if actual_action == 1:
            self.buses_on_road.append(Bus(self.c_max,self.num_stations,self.current_minute))


        self.waiting_time = 0 
        self.current_carrying_capacity = 0
        self.c_max_m = 0 
        self.current_no_waiting_passengers = 0

        for idx,bus in enumerate(self.buses_on_road):
            if bus.arrv_mark!=1:
                waiting_time = bus.update(self.stations)
                self.waiting_time += waiting_time 
                self.current_carrying_capacity += len(bus.passengers_on)
                if len(bus.passengers_on) >= self.c_max_m:
                    self.c_max_m=len(bus.passengers_on)
                logger.debug(f" Bus {idx} ({bus.start_time//60}:{bus.start_time%60}) with {len(bus.passengers_on)} passengers is {bus.state_str}")

        self.current_no_waiting_passengers = np.sum([len(self.stations.current_minute_passengers[i]) for i  in range(self.num_stations) ])
        self.waiting_time = self.waiting_time +  np.sum([self.current_minute*len(self.stations.current_minute_passengers[i]) - np.sum(self.stations.current_minute_passengers[i]["Arrival time"])  for i in range(self.num_stations) ] )
        
        
        self.current_minute+=1
        # First we update passengers at each station,
        self.stations.forward_one_step()

        logger.debug(f"Current carrying capacity: {self.current_carrying_capacity}")
        logger.debug(f"Current e_m : {self.e_m}")
        logger.debug(f"Waiting time {self.waiting_time}")
        logger.debug(f"Current people waiting passengers: {self.current_no_waiting_passengers}")
        new_state = torch.cat([torch.Tensor([(self.current_minute//60)/24]),torch.Tensor([(self.current_minute%60)/60]),torch.Tensor([(self.c_max_m)/self.c_max ]),torch.Tensor([self.waiting_time]),
                torch.Tensor([(self.current_carrying_capacity)/self.e_m])])

        reward = self.get_reward(actual_action)
        logger.debug(f"Actual action: {actual_action} WTF reward 1 {reward} {self.current_carrying_capacity/self.e_m} - {self.current_no_waiting_passengers/(self.num_stations*self.C)} \n0 1 - {self.current_carrying_capacity/self.e_m} - {self.current_no_waiting_passengers/(self.num_stations*self.C)} - {(self.waiting_time/self.current_no_waiting_passengers)*self.OMEGA}")
            

        return (reward,new_state)
    
    def plot(self):
        bus_image = mpimg.imread('busImage.png')
        fig, ax = plt.subplots(figsize=(17,5))
        passengers = [ len(x) for x in self.stations.current_minute_passengers]
        ax.set_xlim(-1,3*len(passengers))
        ax.set_ylim(0,10)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_yticks([])
        ax.set_xticks([])

        for x, text in enumerate(passengers):
            ax.text(3*x,-1.25,text)

        ax.text(0.5,7.5,f"Current time: {self.current_minute//60}:{self.current_minute%60}",color='red')

        if len(self.buses_on_road)!= 0:
            for bus in self.buses_on_road:
                if bus.arrv_mark!=1:
                    num_passengers = len(bus.passengers_on)
                    state_str = re.findall(r'\d+', bus.state_str)
                    if len(state_str)==1:
                        position = int(state_str[0])
                    else:
                        position = (int(state_str[0]) +  int(state_str[1]))/2
                    
                    ax.text(3*position+0.2, 2.7, str(num_passengers))
                    ax.imshow(bus_image,extent=[3*position, 3*position+2, 1,3])
        return fig, ax
    
    def step(self, action):
        # take action, update the environment, and return next state, reward, and done flag
        reward, next_state = self.update_environment(action)
        done = self.is_done()  # if the episode is done
        return next_state, reward, done

    def is_done(self):
        if self.current_minute >= self.last_minute:
            return True
        else:
            return False
    




first_time="6:30"
last_time="22:00"
max_capacity = 47
trf_path="data/line1/traffic-0.csv"
passenger_info_path = "data/line1/passenger_dataframe_direction0.csv"
trf_con = pd.DataFrame(pd.read_csv(trf_path))
first_minute_th = (int(first_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(first_time[-2:]) - int(trf_con.iloc[0, 1]))
last_minute_th = (int(last_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(last_time[-2:]) - int(trf_con.iloc[0, 1]))
current_minute_th = first_minute_th
passenger_columns = ['Label', 'Boarding time', 'Boarding station', 'Alighting station','Arrival time']


rewards = []
Line = BusLine()
current_minute = first_minute_th
history = pd.DataFrame(columns=["Time","Action","Reward"])
while current_minute<last_minute_th:
    #action = get_action(state)
    if current_minute%10==0:
        action = 1
    else:
        action = 0 
    
    reward,new_state = Line.update_environment(action)
    logger.debug(f"Reward = {reward} New state = {new_state}")
    current_minute+=1
    history = history._append({"Time": current_minute,"Action":action,"Reward":reward},ignore_index=True)
    #fig,ax = Line.plot()
    #fig.savefig(f"plots/{current_minute}.png")


# Filter DataFrame by Action
#action_A = history[history["Action"] == 0]
#action_B = history[history["Action"] == 1]



# Plotting
plt.plot(history["Time"], history["Reward"])
#plt.plot(action_B["Time"], action_B["Reward"], label='Action 1')

# Adding labels and title
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Reward over Time (Current Timetable)')

# Adding legend
plt.legend()
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.savefig("initialReward1.png",dpi=100)
plt.show()
