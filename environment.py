import pandas as pd
import numpy as  np
import torch
import warnings
import csv
warnings.filterwarnings("ignore")



class Station:

    def __init__(self, station_number,all_passenger_info_path, first_minute):

        self.station_number = station_number  #Total number of stations
        self.init_first_minute_passengers = [] #Passengers waiting at the station when the train departs
        self.current_minute_passengers = [] #Passengers waiting at the station at the current minute
        self.next_minute_passengers = [] #The next minute, a passenger about to arrive at the station
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
            
            # Passengers in the next minute
            self.next_minute_passengers.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"] == (self.first_minute+1)])
        
        self.current_minute_passengers =  self.init_first_minute_passengers
        

    def forward_one_step(self):

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
        self.current_minute = self.start_time
        self.station_number = station_number
        self.state_str = ""
        self.max_capacity = max_capacity #Maximum number of people in the bus
        self.position = np.zeros(station_number) #Bus location
        self.position[0]=1
        self.pass_position = np.zeros(station_number) #Stations through which vehicles pass for capacity calculations
        self.pass_position[0]=1
        self.passengers_on = pd.DataFrame(columns=passenger_columns)
        self.arrv_mark = 0 #terminate
        self.estimated_arrival_time = np.zeros(station_number)  #estimated time of arrival at a given station
        self.estimated_arrival_time[0] = start_time
        self.last_station = 0

        # We calculate an estimated arrival time using the traffic data
        for i in range(1, station_number):
            self.estimated_arrival_time[i] = start_time +sum(trf_con.iloc[(start_time // 15), 6:6+i]) 
            if  (self.estimated_arrival_time[i]-self.estimated_arrival_time[i-1])<=0:
                self.estimated_arrival_time[i]= self.estimated_arrival_time[i-1] + 1
            



    def update(self,stations:Station):
        #print(f"Bus time: {self.current_minute//60}:{self.current_minute%60}")
        arrival_times = self.estimated_arrival_time
        waiting_time = 0 

        if self.arrv_mark!=1:
            if self.current_minute in arrival_times:
                
                self.position = np.zeros(self.station_number)
                station_index = np.where(arrival_times==self.current_minute)[0][0]
                self.position[station_index] = 1
                self.last_station = station_index
                self.state_str = f"in station {station_index}"
                self.pass_position[station_index] = 1
                if sum(self.pass_position)==self.station_number:
                        self.arrv_mark=1
                        self.state_str = "Done for today"
                        
                        
                people_in_station =  stations.current_minute_passengers[station_index]

                # Drop Passengers
                self.passengers_on = self.passengers_on._append(self.passengers_on[self.passengers_on["Alighting station"] == station_index])
                self.passengers_on = self.passengers_on.drop_duplicates(subset=["Label", 'Boarding time', 'Boarding station', 'Alighting station'], keep=False)

                if len(people_in_station) + len(self.passengers_on) <= self.max_capacity:

                    # we add passengers
                    boarding_passengers = people_in_station
                    self.passengers_on = pd.concat([self.passengers_on, boarding_passengers])
                    
                    # station has now zero people
                    stations.current_minute_passengers[station_index] = pd.DataFrame(columns=passenger_columns)
                    
                   
                
                else:
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
    def __init__(self, passenger_data_path:str, traffic_data_path:str ,first_minute:str,last_minute:str,c_max:int,beta:int) -> None:
        
        self.passenger_data = pd.read_csv(passenger_data_path)
        self.traffic_data = pd.read_csv(traffic_data_path)
        self.num_stations = self.passenger_data['Boarding station'].nunique()
        self.first_minute = first_minute
        self.last_minute = last_minute
        self.current_minute = self.first_minute 

        self.buses_on_road = []
        self.stations = Station(self.num_stations,passenger_data_path,first_minute)
        self.c_max_m = 0 
        self.c_max = c_max
        self.e_m = 1*(self.num_stations-1)*self.c_max
        self.waiting_time = 0
        self.beta=beta
        self.current_carrying_capacity = 0
        self.stranded_passengers = 0 
        self.last_bus_departure = self.current_minute

    def get_reward(self,action):
        
        reward = 0 
        if action == 1:
            reward = self.current_carrying_capacity/self.e_m - self.beta*self.waiting_time
        elif action == 0: 
            reward = 1 - self.current_carrying_capacity/self.e_m -  self.beta*self.waiting_time - self.beta*self.stranded_passengers
        return reward
    
    def update_environment(self, action):

        print(f"\nCurrent time {self.current_minute//60}:{self.current_minute%60}\n")
        
        actual_action = action
        if (( self.last_minute - self.current_minute) <self.EARLY_STOP):
            actual_action = 0
        elif ((self.current_minute == self.first_minute) or (self.current_minute == self.last_minute) or ((self.current_minute - self.last_bus_departure)>=self.MAX_INTERVAL) ):
            actual_action = 1
            self.last_bus_departure = self.current_minute
        elif (action ==1 ) and ((self.current_minute - self.last_bus_departure)<=self.MIN_INTERVAL):
            actual_action = 0
        elif action ==1:
            self.last_bus_departure = self.current_minute
        
        

        
         
        if actual_action == 1:
            self.buses_on_road.append(Bus(self.c_max,self.num_stations,self.current_minute))


        self.waiting_time = 0 
        self.current_carrying_capacity = 0
        self.c_max_m = 0 

        for idx,bus in enumerate(self.buses_on_road):
            if bus.arrv_mark!=1:
                waiting_time = bus.update(self.stations)
                self.waiting_time += waiting_time 
                self.current_carrying_capacity += len(bus.passengers_on)
                if len(bus.passengers_on) >= self.c_max_m:
                    self.c_max_m=len(bus.passengers_on)
                print(f" Bus {idx} ({bus.start_time//60}:{bus.start_time%60}) with {len(bus.passengers_on)} passengers is {bus.state_str}")

       
        self.waiting_time = self.waiting_time +  np.sum([self.current_minute*len(self.stations.current_minute_passengers[i]) - np.sum(self.stations.current_minute_passengers[i])  for i in range(self.num_stations) ] )
        self.stranded_passengers = self.stranded_passengers +  np.sum([len(self.stations.current_minute_passengers[i])   for i in range(self.num_stations) ] )
        
        with open('dict.csv', 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in self.stranded_dict.items():
                writer.writerow([key, value])
        self.current_minute+=1
        # First we update passengers at each station,
        self.stations.forward_one_step()

        print(f"Current carrying capacity: {self.current_carrying_capacity}")
        print(f"Current people waiting passengers: {self.stranded_passengers}")
        new_state = torch.cat([torch.Tensor([(self.current_minute//60)/24]),torch.Tensor([(self.current_minute%60)/60]),torch.Tensor([(self.c_max_m)/self.c_max ]),torch.Tensor([self.waiting_time]),
                torch.Tensor([(self.current_carrying_capacity)/self.e_m])])

        reward = self.get_reward(actual_action)

        return (reward,new_state)


first_time="6:30"
last_time="22:00"
max_capacity = 47
trf_path="data/line3/traffic-1.csv"
passenger_info_path = "data/line3/passenger_dataframe_direction1.csv"
trf_con = pd.DataFrame(pd.read_csv(trf_path))
first_minute_th = (int(first_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(first_time[-2:]) - int(trf_con.iloc[0, 1]))
last_minute_th = (int(last_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(last_time[-2:]) - int(trf_con.iloc[0, 1]))
current_minute_th = first_minute_th
passenger_columns = ['Label', 'Boarding time', 'Boarding station', 'Alighting station','Arrival time']

Line = BusLine(passenger_info_path,trf_path,first_minute_th,last_minute_th,max_capacity,1)
current_minute = first_minute_th
while current_minute<last_minute_th:
    reward,new_state = Line.update_environment(1)
    print(f"Reward = {reward} New state = {new_state}")
    current_minute+=1

