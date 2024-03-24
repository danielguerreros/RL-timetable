import pandas as pd
import numpy as  np
import random
import torch
import copy

#coding:utf-8
__all__ = ["Environment_Calculation"]



#初始发车时间，最后发车时间
first_time="6:30"
last_time="22:00"

station_num  = 36  #线路的站数

max_capacity = 47 #每辆车的最大运载人数

weight_time_wait_time = 5000

min_Interval =5

max_Interval =22


All_passenger_wait_time = 0

departure_label = 0

All_cap_out = 0

All_cap_uesd = 0

#交通情况
trf_path="data/line1/traffic-0.csv"
trf_con = pd.DataFrame(pd.read_csv(trf_path))


passenger_info_path = "data/line1/passenger_dataframe_direction0.csv"

# We need minute and hour in the dataframe
first_minute_th = (int(first_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(first_time[-2:]) - int(trf_con.iloc[0, 1]))
last_minute_th = (int(last_time[:-3]) - int(trf_con.iloc[0, 0])) * 60 + (int(last_time[-2:]) - int(trf_con.iloc[0, 1]))
current_minute_th = first_minute_th



class Station:
    def __init__(self, station_number=station_num,all_passenger_info_path=passenger_info_path, first_minute = first_minute_th):

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

    def reset(self):

        self.init_first_minute_passengers = [] #Passengers waiting at the station when the train departs
        self.current_minute_passengers = [] #Passengers waiting at the station at the current minute
        self.next_minute_passengers = [] #The next minute, a passenger about to arrive at the station
        self.all_passenger_info_dataframe =  pd.read_csv(self.all_passenger_info_path)
        self.all_station_all_minute_passenger = []  # All stations, passengers boarding every minute ([[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==1]],[all_passenger_info_dataframe[all_passenger_info_dataframe["Boarding station"]==2]],[],[],])

        for i in range(self.station_number): 
            # All stops, passengers per minute
            self.all_station_all_minute_passenger.append(self.all_passenger_info_dataframe[self.all_passenger_info_dataframe["Boarding station"]==i])#Every station, all minutes of

            #For the first minute, passengers at all stops
            self.init_first_minute_passengers.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"]<=self.first_minute])
            
            #The next minute, a passenger about to arrive at the station
            self.next_minute_passengers.append(self.all_station_all_minute_passenger[i][self.all_station_all_minute_passenger[i]["Arrival time"] == (self.first_minute+1)])
        
        self.current_minute_passengers =  self.init_first_minute_passengers

    def forward_one_step(self):
        for i in range(self.station_number):
            self.current_minute_passengers[i]=pd.concat([self.current_minute_passengers[i],self.next_minute_passengers[i]])
        self.current_minute = self.current_minute + 1
        
        self.next_minute_passengers = []
        for i in range(self.station_number):  # 下一分钟，即将到达车站的乘客
            self.next_minute_passengers.append(self.all_station_all_minute_passenger[i][
                                                  self.all_station_all_minute_passenger[i]["Arrival time"] == (
                                                              self.current_minute + 1)])


    def Estimated_psger_procs(self,station_no,time_start,time_finish):
        psger_procs1=self.all_station_all_minute_passenger[station_no][self.all_station_all_minute_passenger[station_no]["Arrival time"]>time_start]
        psger_procs2 = psger_procs1[psger_procs1["Arrival time"]<=time_finish]
        return psger_procs2


station = Station()
print(f" Time {(station.current_minute+1)} {[len(x) for x in station.next_minute_passengers]}")
station.forward_one_step()
print(f" Time {(station.current_minute+1)} {[len(x) for x in station.next_minute_passengers]}")
station.forward_one_step()
print(f" Time {(station.current_minute+1)} {[len(x) for x in station.next_minute_passengers]}")
station.forward_one_step()
print(f" Time {(station.current_minute+1)} {[len(x) for x in station.next_minute_passengers]}")



class Bus:
    def __init__(self,max_capacity=max_capacity, station_number=station_num,start_time= current_minute_th):


        self.start_time =start_time #发出的时间

        self.station_number = station_number

        self.max_capacity = max_capacity #最大人数

        self.passengers_on =pd.DataFrame(columns=["No_num", "Get_in_time_th","Boarding station","Get_off_station","Arrival time"])   #车上人员

        self.num_passengers_on = 0 #车上人数

        self.position = np.zeros(station_number) #车辆位置
        self.position[0]=1

        self.pass_position = np.zeros(station_number) #车辆通过的站点，用于计算运力
        self.pass_position[0]=1

        self.left_every_station_pn_on = np.zeros(station_number)

        self.there_label = 1

        self.pass_minute = 0

        self.passenger_on_board_leaving_station = pd.DataFrame(columns=["No_num", "Get_in_time_th","Boarding station","Get_off_station","Arrival time"])

        self.arrv_mark = 0 #到达终点站

        self.onlie_mark =1 #目前仍未到达终点站

        self.bus_milage = 0 #走过某站后的距离

        self.cant_taken_once = 0

        self.Estimated_arrival_time = np.zeros(station_number)  #预计到达某站的时间
        self.Estimated_arrival_time[0] = current_minute_th
        for i in range(1, station_number):
            self.Estimated_arrival_time[i] = current_minute_th +sum(trf_con.iloc[(current_minute_th // 15), 6:6+i]) - self.pass_minute

        self.To_be_process_all_station = []
        for i in range(0, station_number):
            self.To_be_process_all_station.append(pd.DataFrame(columns=["No_num", "Get_in_time_th","Boarding station","Get_off_station","Arrival time"]))

        self.Left_after_pass_the_station = []
        for i in range(0, station_number):
            self.Left_after_pass_the_station.append(pd.DataFrame(
                columns=["No_num", "Get_in_time_th", "Boarding station", "Get_off_station", "Arrival time"]))

    def forward_one_step(self):
        self.there_label = 0
        for i in range(len(self.position) - 1):
            if self.position[i] == 1:
                self.bus_milage = self.bus_milage + (1 / trf_con.iloc[(current_minute_th // 15), 6 + i])
                self.pass_minute = self.pass_minute + 1


                if self.bus_milage >= 1:
                    self.pass_minute = 0
                    self.bus_milage = self.bus_milage - 1
                    self.position[i + 1] = 1
                    self.pass_position[i + 1] = 1  # 过站标志位

                    self.position[i] = 0
                    self.there_label = 1

                    if sum(self.pass_position)==self.station_number:
                        self.arrv_mark=1
                    break
                break



    def up_down(self,station=Station):

        global All_cap_out

        global All_passenger_wait_time

        global All_cap_uesd


        for i in range(station.station_number):
            if self.there_label ==1 and self.position[i]==1:
                #print("222222",(station.current_minute_passengers[i]),len(self.passengers_on))
                #print(current_minute_th,"入站",len(self.passengers_on))
                #乘客下车

                self.passengers_on = self.passengers_on.append(self.passengers_on[self.passengers_on["Get_off_station"]==i])
                self.passengers_on = self.passengers_on.drop_duplicates(subset=["No_num", "Get_in_time_th","Boarding station","Get_off_station"], keep=False)


                if len(station.current_minute_passengers[i])+len(self.passengers_on)<=self.max_capacity:
                    station.sum1 = station.sum1 + len(station.current_minute_passengers[i])
                    self.passengers_on=pd.concat([self.passengers_on,station.current_minute_passengers[i]])
                    All_cap_out = All_cap_out + self.max_capacity
                    All_passenger_wait_time =All_passenger_wait_time + current_minute_th*len(station.current_minute_passengers[i])-np.sum(station.current_minute_passengers[i].iloc[:,4])
                    station.current_minute_passengers[i] = pd.DataFrame(columns=["No_num", "Get_in_time_th","Boarding station","Get_off_station","Arrival time"])
                    All_cap_uesd = All_cap_uesd + len(self.passengers_on)
                else:

                    station.sum1 = station.sum1 + len(station.current_minute_passengers[i])
                    append_mark=self.max_capacity - len(self.passengers_on)
                    self.passengers_on = self.passengers_on.append(station.current_minute_passengers[i].iloc[:append_mark,:])
                    All_cap_out = All_cap_out + self.max_capacity

                    All_passenger_wait_time = All_passenger_wait_time + current_minute_th * append_mark - np.sum(station.current_minute_passengers[i].iloc[:append_mark, 4])

                    station.current_minute_passengers[i] = station.current_minute_passengers[i].iloc[append_mark:,:]
                    All_cap_uesd = All_cap_uesd + len(self.passengers_on)





class BUS_LINE_SYSTEM:
    # Line of the bus
    # To initiate the class we need a Station and station number
    def __init__(self, station_number=station_num, station=Station):
        self.station_number = station_number
        self.bus_online = []
        self.Interval = 0
        self.end_label = 0
        self.station = station
        self.bus_online_test = []
        self.wait_time = 0
        self.All_psger_wait_time = 0
        self.All_cap_take = 0
        self.All_cap_uesd = 0

        self.All_psger_wait_time_depart_once= 0
        self.All_cap_take_depart_once = 0
        self.All_cap_uesd_depart_once = 0
        self.Cant_taken_once  = 0
        self.Cap_used = 0
        self.if_depart_wait_time = 0


    def Departure(self,):
        # If we depart We initialize the bus (Max number of passengers, number of starting station and starting time)
        self.bus_online.append(Bus(max_capacity=max_capacity,station_number=station_num,start_time= current_minute_th))
        self.All_cap_take = self.All_cap_take + max_capacity*(station_num-1)
        self.All_cap_take_depart_once = self.All_cap_take_depart_once + max_capacity * (station_num - 1)
        if   len(self.bus_online) == 0:
            pass
        elif len(self.bus_online) == 1:

            # We iterate over all stations
            for i in range(self.station_number):


                self.bus_online[0].To_be_process_all_station[i] = pd.concat([self.station.current_minute_passengers[i],self.station.Estimated_psger_procs(i, current_minute_th, self.bus_online[0].Estimated_arrival_time[i])])
                self.bus_online[0].passengers_on = self.bus_online[0].passengers_on._append(self.bus_online[0].passengers_on[self.bus_online[0].passengers_on["Get_off_station"] == i])
                self.bus_online[0].passengers_on = self.bus_online[0].passengers_on.drop_duplicates(subset=["No_num", "Get_in_time_th", "Boarding station", "Get_off_station"], keep=False)

                if len(self.bus_online[0].To_be_process_all_station[i]) + len(self.bus_online[0].passengers_on) <= self.bus_online[0].max_capacity:
                    self.bus_online[0].passengers_on = pd.concat([self.bus_online[0].passengers_on, self.bus_online[0].To_be_process_all_station[i]])
                    self.bus_online[0].Left_after_pass_the_station[i] = pd.DataFrame(columns=["No_num", "Get_in_time_th", "Boarding station", "Get_off_station", "Arrival time"])
                    self.All_psger_wait_time = self.All_psger_wait_time +self.bus_online[0].Estimated_arrival_time[i]*len(self.bus_online[0].To_be_process_all_station[i]) - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:,4])
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[0].Estimated_arrival_time[i] * len(self.bus_online[0].To_be_process_all_station[i]) - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].passengers_on)
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].passengers_on)
                    self.bus_online[-1].left_every_station_pn_on[i]=len(self.bus_online[-1].passengers_on)
                else:
                    append_mark = (self.bus_online[0].max_capacity - len(self.bus_online[0].passengers_on))
                    self.bus_online[0].passengers_on.append(self.bus_online[0].To_be_process_all_station[i].iloc[:append_mark, :])
                    self.bus_online[0].Left_after_pass_the_station[i] = self.bus_online[0].To_be_process_all_station[i].iloc[append_mark:, :]
                    self.All_psger_wait_time = self.All_psger_wait_time + self.bus_online[0].Estimated_arrival_time[i] * append_mark - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[0].Estimated_arrival_time[i] * append_mark - np.sum(self.bus_online[0].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].passengers_on)
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].passengers_on)
                    self.bus_online[-1].left_every_station_pn_on[i] = len(self.bus_online[-1].passengers_on)
                    self.bus_online[-1].cant_taken_once = self.bus_online[-1].cant_taken_once + len(self.bus_online[0].To_be_process_all_station[i])-append_mark

        elif len(self.bus_online)  > 1:
            for i in range(self.station_number):
                self.bus_online[-1].To_be_process_all_station[i]= pd.concat([self.bus_online[-2].Left_after_pass_the_station[i],self.station.Estimated_psger_procs(i, self.bus_online[-2].Estimated_arrival_time[i], self.bus_online[-1].Estimated_arrival_time[i])])


                self.bus_online[-1].passengers_on = self.bus_online[-1].passengers_on._append(self.bus_online[-1].passengers_on[self.bus_online[-1].passengers_on["Get_off_station"] == i])
                self.bus_online[-1].passengers_on = self.bus_online[-1].passengers_on.drop_duplicates(subset=["No_num", "Get_in_time_th", "Boarding station", "Get_off_station"], keep=False)


                if len(self.bus_online[-1].To_be_process_all_station[i]) + len(self.bus_online[-1].passengers_on) <= self.bus_online[-1].max_capacity:

                    self.bus_online[-1].passengers_on = pd.concat([self.bus_online[-1].passengers_on, self.bus_online[-1].To_be_process_all_station[i]])
                    self.bus_online[-1].Left_after_pass_the_station[i] = pd.DataFrame(columns=["No_num", "Get_in_time_th", "Boarding station", "Get_off_station", "Arrival time"])
                    self.All_psger_wait_time = self.All_psger_wait_time + self.bus_online[-1].Estimated_arrival_time[i] * len(self.bus_online[-1].To_be_process_all_station[i]) - np.sum(self.bus_online[-1].To_be_process_all_station[i].iloc[:, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].passengers_on)
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[-1].Estimated_arrival_time[
                        i] * len(self.bus_online[-1].To_be_process_all_station[i]) - np.sum(
                        self.bus_online[-1].To_be_process_all_station[i].iloc[:, 4])
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].passengers_on)


                    self.bus_online[-1].left_every_station_pn_on[i] = len(self.bus_online[-1].passengers_on)


                else:
                    append_mark = (self.bus_online[-1].max_capacity - len(self.bus_online[-1].passengers_on))
                    self.bus_online[-1].passengers_on._append(self.bus_online[-1].To_be_process_all_station[i].iloc[:append_mark, :])
                    self.bus_online[-1].Left_after_pass_the_station[i] = self.bus_online[-1].To_be_process_all_station[i].iloc[append_mark:, :]
                    self.All_psger_wait_time = self.All_psger_wait_time + self.bus_online[-1].Estimated_arrival_time[i] * append_mark - np.sum(self.bus_online[-1].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_cap_uesd = self.All_cap_uesd + len(self.bus_online[-1].passengers_on)
                    self.All_psger_wait_time_depart_once = self.All_psger_wait_time_depart_once + self.bus_online[-1].Estimated_arrival_time[
                        i] * append_mark - np.sum(
                        self.bus_online[-1].To_be_process_all_station[i].iloc[:append_mark, 4])
                    self.All_cap_uesd_depart_once = self.All_cap_uesd_depart_once + len(self.bus_online[-1].passengers_on)
                    self.bus_online[-1].left_every_station_pn_on[i] = len(self.bus_online[-1].passengers_on)
                    self.bus_online[-1].cant_taken_once = self.bus_online[-1].cant_taken_once + len(
                        self.bus_online[-1].To_be_process_all_station[i]) - append_mark
        #print("fache")

    def Arrival_test(self):
        for i in range (len(self.bus_online)):
            if self.bus_online[i].position[-1]==1:
                self.bus_online[i].arrv_mark = 1
                self.bus_online[i].onlie_mark = 0
                break

    def Action(self, Departure_factor=0, min_Interval=min_Interval, max_Interval=max_Interval):
        # 人进站
        global departure_label
        if Departure_factor != 1 and Departure_factor != 0:  # 确定发车因子
            af = random.random()
            if af <= 0.5:
                Departure_factor = 0
            else:
                Departure_factor = 1
        else:
            pass
        # 结束标志位
        if current_minute_th > last_minute_th + 50 :
            self.end_label = 1

        # 发车
        if (current_minute_th == first_minute_th or current_minute_th == last_minute_th or self.Interval == max_Interval) and current_minute_th < last_minute_th+1:   #
            self.Departure()
            self.Interval = 0
            departure_label = 1
        else:
            if self.Interval < min_Interval or Departure_factor == 0 or current_minute_th > last_minute_th:
                self.Interval = self.Interval + 1
            else:
                self.Departure()
                self.Interval = 0
                departure_label = 1
        # 人上下车



    def get_state(self):
        #print("hhhhhhh",len(self.bus_online),self.bus_online[0].position)
        self.bus_online_test =copy.deepcopy(self.bus_online)


        bus_on_line_now = []   #目前在线路上运行的公交车
        for j in range(len(self.bus_online_test)):
            if self.bus_online_test[j].position[-1]== 0:
                bus_on_line_now.append((self.bus_online_test[j]))
                #print("第",j,"辆车",bus_on_line_now[j].Estimated_arrival_time)

        Capacity_need = 0

        Passenger_num_on_board_leaving_station = pd.DataFrame(columns=["No_num", "Get_in_time_th","Boarding station","Get_off_station","Arrival time"])

        Passenger_num_on_board_leaving_station_num = []

        Passenger_tobe_proc_per_station = []

        right_deapart_estimated_arrival_time=np.zeros(self.station_number)  # 预计到达某站的时间

        right_deapart_estimated_arrival_time[0] = current_minute_th


        for i in range(1,self.station_number-1):
            right_deapart_estimated_arrival_time[i] =  current_minute_th +sum(trf_con.iloc[(current_minute_th // 15), 6:6+i])

        #print(current_minute_th, "预发车辆到站时间", right_deapart_estimated_arrival_time)
        #print(current_minute_th, "上一辆车离开的时间", self.bus_online[-1].Estimated_arrival_time)

        last_bus_left_per_station = []
        #print("在线车数",len(bus_on_line_now))


        This_deaparture_pre_wait_time = 0

        Passenger_cant_takeon_once = 0



        for i in range(self.station_number):
                Passenger_tobe_proc_per_station.append(pd.concat([self.bus_online[-1].Left_after_pass_the_station[i], self.station.Estimated_psger_procs(i,self.bus_online[-1].Estimated_arrival_time[i],right_deapart_estimated_arrival_time[i])]))

                # print("下车前",i,len(Passenger_num_on_board_leaving_station))
                Passenger_num_on_board_leaving_station = Passenger_num_on_board_leaving_station._append(Passenger_num_on_board_leaving_station[Passenger_num_on_board_leaving_station["Get_off_station"] == i])
                Passenger_num_on_board_leaving_station = Passenger_num_on_board_leaving_station.drop_duplicates(subset=["No_num", "Get_in_time_th", "Boarding station", "Get_off_station"], keep=False)
                # print("下车后",i,len(Passenger_num_on_board_leaving_station))

                if len(Passenger_tobe_proc_per_station[i]) + len(Passenger_num_on_board_leaving_station) <= max_capacity:

                    # print(current_minute_th,1,len(self.passengers_on),len(station.current_minute_passengers[i]))
                    Passenger_num_on_board_leaving_station = pd.concat([Passenger_num_on_board_leaving_station, Passenger_tobe_proc_per_station[i]])
                    This_deaparture_pre_wait_time = This_deaparture_pre_wait_time + right_deapart_estimated_arrival_time[i] * len(Passenger_tobe_proc_per_station[i]) - np.sum(Passenger_tobe_proc_per_station[i].iloc[:, 4])


                else:
                    append_mark = (max_capacity - len(Passenger_num_on_board_leaving_station))
                    Passenger_num_on_board_leaving_station = pd.concat([Passenger_num_on_board_leaving_station,Passenger_tobe_proc_per_station[i].iloc[:append_mark, :]])
                    This_deaparture_pre_wait_time = This_deaparture_pre_wait_time + right_deapart_estimated_arrival_time[i] * append_mark - np.sum(Passenger_tobe_proc_per_station[i].iloc[:append_mark, 4])
                    Passenger_cant_takeon_once = Passenger_cant_takeon_once + len(Passenger_tobe_proc_per_station[i]) - append_mark

                Passenger_num_on_board_leaving_station_num.append(len(Passenger_num_on_board_leaving_station))
        state = torch.cat([torch.Tensor([(current_minute_th//60)/24]),torch.Tensor([(current_minute_th%60)/60]),torch.Tensor([(np.max(Passenger_num_on_board_leaving_station_num))/max_capacity ]),torch.Tensor([(This_deaparture_pre_wait_time)/weight_time_wait_time]),
                 torch.Tensor([(np.sum(Passenger_num_on_board_leaving_station_num))/1750])])
        self.Cant_taken_once = Passenger_cant_takeon_once
        self.Cap_used = (np.sum(Passenger_num_on_board_leaving_station_num))
        self.if_depart_wait_time = This_deaparture_pre_wait_time
        return state
            # print(current_minute_th, "每站待处理人数     ", [len(i) for i in Passenger_tobe_proc_per_station])
            # print(current_minute_th, "每站离开时车上的人数", Passenger_num_on_board_leaving_station_num)




"""
XM_2_station = Station(all_passenger_info_path=passenger_info_path)

#print(XM_2_station.next_minute_passenger)

XM_2system = BUS_LINE_SYSTEM(station=XM_2_station)

#print(XM_2_station.next_minute_passenger)

bus_on_line_now = []
for i in range(100):
    #print([len(k) for k in XM_2_station.next_minute_passenger])
    if i % 10 == 0 :
        XM_2system.Action(1)
    #print( XM_2system.bus_online[0].passengers_on)
    bus_on_line_now=[]
    for j in range(len(XM_2system.bus_online)):
        if XM_2system.bus_online[j].arrv_mark!= 1:
            bus_on_line_now.append(len(XM_2system.bus_online[j].passengers_on))

    XM_2system.get_state()
    XM_2system.station.forward_one_step()
    print("Number of buses on line:",len(bus_on_line_now))
    #print(current_minute_th,XM_2system.bus_online[-1].left_every_station_pn_on)
    #print(current_minute_th,[len(k) for k in XM_2_station.current_minute_passengers])
    #print(XM_2system.get_state())
    print(current_minute_th,"Capacity and waiting time",XM_2system.All_cap_take,XM_2system.All_cap_uesd,XM_2system.All_psger_wait_time)
    current_minute_th = current_minute_th + 1
    #print(current_minute_th)
"""


