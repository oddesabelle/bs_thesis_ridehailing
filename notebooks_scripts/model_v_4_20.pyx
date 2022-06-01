#this isa print build

import numpy as np
cimport numpy as np
from libc.stdlib cimport RAND_MAX,rand
import random as rnd    
# cimport PedestrianCrowding.random
# cdef PedestrianCrowding.random.mt19937 gen = PedestrianCrowding.random.mt19937(rnd.randint(0, RAND_MAX))
# cdef PedestrianCrowding.random.uniform_real_distribution[double] dist = PedestrianCrowding.random.uniform_real_distribution[double](0.0,1.0)

STUFF="HI"
cdef int max_decel = 2
cdef class Vehicle:
    cdef readonly:
        int pos,lane, prev_lane, marker, vel, vmax
        float p_lambda, p_slow
    
    cdef public:
        np.int_t[:, :] road
        int prev_vel, id
    
    cdef float rng
    cdef Road Road

    def __cinit__(self, Road Road, int pos, int lane, int vel, float p_slow, float p_lambda, **kwargs):
        self.pos = pos
        self.lane = lane
        self.prev_lane = 0
        self.p_lambda = p_lambda
        self.p_slow = p_slow
        self.marker = 1
        self.vel = vel
        self.vmax = Road.vmax
        self.road = Road.road
        self.Road = Road
        self.id = Road.get_id()

    cdef void accelerate(self):
        self.prev_vel = self.vel
        self.vel = min(self.vel+1, self.vmax)
        self.rng = np.random.random()

    cdef void decelerate(self):
        self.vel = min(self.vel, self.headway(self.lane))

       
    cdef void random_slow(self):
        #self.rng = np.random.random()
        if (self.rng < self.p_slow):
            self.vel = max(self.vel-1, 0)

    cdef void movement(self):
        self.remove()
        self.pos += self.vel
        if self.Road.periodic:
            self.pos %= self.Road.roadlength
        self.place()

    cdef int headway(self, int lane):
        cdef int _pos, headwaycount
        headwaycount = 0
        _pos = self.pos+1
        if self.Road.periodic:
            _pos %= self.Road.roadlength
        while (self.road[lane, _pos]==0) and (headwaycount<(self.vmax*2)):
            _pos += 1
            if self.Road.periodic:
                _pos %= self.Road.roadlength
            headwaycount += 1
        return headwaycount


    cdef void place(self):
        self.road[self.lane, self.pos] = self.id

    cdef void remove(self):
        self.road[self.lane, self.pos] = 0

    cdef bint lanechange(self):
        cdef int where, max_headway, i, headway, current_headway
        where = 0; max_headway = 0
        cdef np.int_t[:] s
        s = np.zeros(self.Road.num_lanes, dtype=int)
        current_headway = self.headway(self.lane)

        if self.vel>current_headway:
            # scan for suitable lanes
            for i in range(self.Road.num_lanes):
                headway = self.headway(i)
                s[i] = headway
                if headway>max_headway:
                    where = i
                    max_headway = headway
            # self.lookback(where)

            if (where != self.lane) and (max_headway>current_headway):  # desired lane is different
                if self.lane > where and ((self.road[self.lane-1, self.pos]==0)) and self.lookback(self.lane-1):  # left
                    self.remove()
                    self.lane -= 1
                    self.place()
                    return True
                elif self.lane < where and ((self.road[self.lane+1, self.pos])==0) and self.lookback(self.lane+1):  # right
                    self.remove()
                    self.lane += 1
                    self.place()
                    return True
        return False

    cdef bint lookback(self, int target_lane):
        cdef int _pos, headwaycount
        headwaycount = 0
        _pos = self.pos-1
        if self.Road.periodic:
            _pos %= self.Road.roadlength
        while (self.road[target_lane, _pos]==0) and (headwaycount<(self.vmax)):
            _pos -= 1
            if self.Road.periodic:
                _pos %= self.Road.roadlength
            headwaycount += 1
        back_id = self.road[target_lane, _pos]
        if back_id == 0: # all clear!
            return True
        elif headwaycount+self.vel >= self.Road.get_vehicle(back_id).vel-max_decel: # will it crash?
            return True
        else: # too risky!
            return False

#############################################################################################################################
cdef class TNV(Vehicle):
    cdef public:
        np.int_t [:,:] pedestrian
        float p_1
        float p_2
        int num_passengers
        int wait_counter
        int tnv_wait_time
        int trip
        int trip_end
        
    #cdef float rng

    cdef float rng2
    
    def __cinit__(self, Road Road, int pos, int lane, int vel, float p_slow, float p_lambda, float p_1, float p_2, int tnv_wait_time, **kwargs):
        self.pos = pos
        self.lane = lane
        self.prev_lane = 0
        self.p_lambda = p_lambda
        self.p_slow = p_slow
        self.marker = 2
        self.vel = vel
        self.vmax = Road.vmax
        self.road = Road.road
        self.Road = Road
        self.pedestrian = Road.pedestrian
        self.num_passengers = 0
        #self.wait_time = Road.tnv_wait_time
        self.tnv_wait_time = Road.tnv_wait_time
        self.wait_counter = 0
        self.p_1 = self.Road.p_1
        self.p_2 = self.Road.p_2
        self.trip = 0
        self.trip_end = 0
        self.id = Road.get_id()

    #cdef void accelerate(self):
        #self.prev_vel = self.vel
        #self.vel = min(self.vel+1, self.vmax)
        
        
        
    cpdef void decelerate(self):
        hw_pass = self.passenger_headway()
        hw = self.headway(self.lane)

        # skip when too fast
        if (2*self.prev_vel-3*max_decel)>hw_pass:# and (self.prev_vel-max_decel)>(max_decel):
            self.vel = min([self.vel, hw])
        else:
            # anticipate the stop
            if (self.prev_vel+max_decel)>=hw_pass>=max_decel:
                self.vel = min([self.vel, hw, max(self.prev_vel-max_decel, max_decel)]) 
            else:
                self.vel = min([self.vel, hw, hw_pass])
        # print(self.prev_vel, self.vel, hw, hw_pass, c)

    cdef int passenger_headway(self):
        cdef int _pos, headwaycount
        headwaycount = 0
        _pos = self.pos+1
        if self.Road.periodic:
            _pos %= self.Road.roadlength
        while (self.pedestrian[self.lane, _pos]==0) and (headwaycount<(self.vmax*2)):
            _pos += 1
            if self.Road.periodic:
                _pos %= self.Road.roadlength
            headwaycount += 1
        return headwaycount + 1
    
    
        
    cpdef void strategy(self):
        #self.rng2 = rand()/RAND_MAX
        #self.rng = self.get_rng
        #cdef float rng2
        #what happens when TNV has passenger
        if self.num_passengers > 0:
            #self.trip += 1
            self.trip += self.vel
            #self.wait_counter = 0
        if self.num_passengers > 0 and self.trip >= self.trip_end:
            self.trip = 0
            self.num_passengers -= 1 #assumes they can alight in the middle of the road lmao
            self.Road.total_trips += 1
            #self.rng = np.random.random()
            if self.rng <= self.p_1:#condition for wander
                pass #do this
            elif self.rng <= self.p_1 + self.p_2: #condition for disappear
                self.vel = 0
                self.remove()
                #self.wait_time = 5
                self.wait_counter += 1
                if self.wait_counter >= self.tnv_wait_time: #self.wait_time
                    self.place() 
                #do this
            else: #condition for hazard
                self.wait_counter +=1
                if self.wait_counter < self.tnv_wait_time:
                    self.vel = 0 
                #self.wait_time = 5
                #do this
          
    
    cpdef void load(self):
        if self.pedestrian[self.lane, self.pos] != 0:
            self.Road.waiting_times.append(self.pedestrian[self.lane, self.pos])
            self.pedestrian[self.lane, self.pos] = 0
            self.vel = 0
            self.num_passengers += 1
            self.wait_counter = 0
            self.trip += self.vel
            #self.trip_end = self.pos + 1 + ((self.Road.roadlength-2)//2)
            self.trip_end = self.Road.roadlength//2

#############################################################################################################################
cdef class Road:
    cdef public:
        int roadlength, num_lanes, vmax, station_period, max_passengers, tnv_wait_time
        float alpha, frac_tnv, density, p_slow, p_1, p_2
        float total_trips
        bint periodic
        np.int_t[:,:] road, road_id_map
        np.int_t[:,:] pedestrian
        list waiting_times
        list vehicle_array
        list full_tnvs
        list wait_counter_array
        list trips
        list trip_ends
        list vehs_on_road
        list onboard_pass_list
        list rng2_list
        list rng1_list
        list rng_list
        list wait_time_list
       # float ave_speed
       # float veh_count
       # np.int_t[:] speeds
        

    cdef int id_counter

    def __cinit__(self, int roadlength, int num_lanes, int vmax, float alpha, float frac_tnv, bint periodic, float density, float p_slow, float p_1, float p_2, int tnv_wait_time, int station_period=1, int max_passengers=1):
        self.roadlength = roadlength
        self.vehicle_array = []
        self.full_tnvs = []
        self.wait_counter_array = []
        self.trips = []
        self.trip_ends = []
        self.vehs_on_road = []
        self.onboard_pass_list = []
        self.rng2_list = []
        self.rng1_list = []
        self.rng_list = []
        self.wait_time_list = []
        self.total_trips = 0.0
        #self.speeds = np.zeros(0)
        #self.ave_speed = 0
        #self.veh_count = 0
        self.road = np.zeros((num_lanes, roadlength), dtype=int)
        self.pedestrian = np.zeros((num_lanes, roadlength), dtype=int)
        self.vmax = vmax
        self.num_lanes = num_lanes
        self.alpha = alpha
        self.periodic = periodic
        self.p_slow = p_slow
        self.p_1 = p_1
        self.p_2 = p_2
        self.waiting_times = []
        self.station_period = station_period
        self.max_passengers = max_passengers
        self.tnv_wait_time = tnv_wait_time
        self.id_counter = 0
        self.frac_tnv = frac_tnv
        self.density = density
        #if frac_tnv <= 1./num_lanes:
        #    self.frac_tnv = frac_tnv
        #else:
        #    raise ValueError("Invalid TNV Fraction")
        cdef int num_tnvs, num_vehicles, num_cars
        if self.periodic:
            num_vehicles = int(density*roadlength*num_lanes)
            num_tnvs = int(num_vehicles*frac_tnv)
            num_cars = num_vehicles - num_tnvs
            self.place_vehicle_type(TNV, num_tnvs)
            self.place_vehicle_type(Vehicle, num_cars)

    cdef int get_id(self):
        self.id_counter = self.id_counter + 1
        return self.id_counter

    cdef place_vehicle_type(self, type veh_type, int number):
        cdef int i, pos, lane
        cdef Vehicle vehicle
        for i in range(number):
            pos=0; lane=self.num_lanes-1
            while not self.place_check(pos, lane):
                pos = np.random.randint(self.roadlength)
                if veh_type != TNV: # type checking
                    lane = np.random.randint(self.num_lanes)
                else:
                    lane = self.num_lanes-1
            vehicle = veh_type(Road=self, pos=pos, lane=lane, vel=self.vmax, p_slow=self.p_slow, p_lambda=1, p_1=self.p_1, p_2=self.p_2, tnv_wait_time=self.tnv_wait_time)
            self.vehicle_array.append(vehicle)
            vehicle.place()

    cdef bint place_check(self, int pos, int lane):
        return False if self.road[lane, pos] else True

#############################################################################################################################
    
    cpdef timestep_parallel(self):
        #self.ave_speed = 0
        #self.veh_count = 0
        if self.frac_tnv>0:
            self.spawn_pedestrian(self.station_period)
        np.random.shuffle(self.vehicle_array)
        cdef list reached_end = []
        cdef int i
        cdef Vehicle vehicle
        self.waiting_times = []
        lcs = np.zeros_like(self.vehicle_array)
        for i, vehicle in enumerate(self.vehicle_array):
            vehicle.accelerate()
            if type(vehicle) == TNV:
                vehicle.strategy()
                if vehicle.num_passengers<self.max_passengers:
                    vehicle.load()

            if np.random.random() < vehicle.p_lambda:
                lcs[i] = vehicle.lanechange()*1
                
        for i, vehicle in enumerate(self.vehicle_array):

            vehicle.decelerate()

            if not lcs[i]:
                vehicle.random_slow()

        for i, vehicle in enumerate(self.vehicle_array):

            vehicle.movement()
            #self.ave_speed += vehicle.vel
            #self.veh_count += 1

            if vehicle.pos >= (self.roadlength-self.vmax-1):
                reached_end.append(i)
        if not self.periodic:
            self.clear(reached_end)
        #self.ave_speed = self.ave_speed/veh_count
        #self.speeds = np.append(self.speeds,ave_speed)
        #self.full_tnvs.append(self.get_num_full_tnvs())
        #self.wait_counter_array.append(self.get_wait_counter())
        #self.trips.append(self.get_trips())
        #self.trip_ends.append(self.get_trip_ends())
        #self.vehs_on_road.append(self.get_on_road())
        #self.onboard_pass_list.append(self.get_onboard_pass())
        #self.rng2_list.append(self.get_rng2())
        #self.rng1_list.append(self.get_rng1())
        #self.rng_list.append(self.get_rng())
        #self.wait_time_list.append(self.get_wait_time())
            
    
#############################################################################################################################

    def clear(self, reached_end):
        for i in reached_end:
            self.vehicle_array[i].remove()
        self.vehicle_array = [veh for i, veh in enumerate(
            self.vehicle_array) if i not in reached_end]

    cdef spawn_pedestrian(self, int period=1):
        cdef int i
        for i in range(0, len(self.pedestrian[self.num_lanes-1]), period):
            if self.pedestrian[self.num_lanes-1][i] == 0:
                self.pedestrian[self.num_lanes-1][i] += (self.road[self.num_lanes-1,i] == 0) * (np.random.random() < self.alpha)*1
            else:
                # increment waiting time
                self.pedestrian[self.num_lanes-1][i] += 1


    cpdef np.float64_t throughput(self):
        return 1.*sum([i.vel for i in self.vehicle_array])/self.roadlength/self.num_lanes

    cpdef np.ndarray[np.float64_t, ndim=1] throughput_per_lane(self):
        cdef np.ndarray[np.float64_t, ndim=1] lane = np.empty(self.num_lanes, dtype=np.float64)
        for i in range(self.num_lanes):
            lane[i] = 1.*sum([j.vel for j in self.vehicle_array if j.lane==i])/self.roadlength
        return lane



    cpdef np.float64_t get_density(self):
        cdef int count = 0
        cdef int i,j
        for i in range(len(self.road)):
            for j in range(len(self.road[0])):
                if self.road[i,j] != 0:
                    count += 1
        return count/self.road.size
    
    cpdef int get_on_road(self):
        cdef int count = 0
        cdef int i,j
        for i in range(len(self.road)):
            for j in range(len(self.road[0])):
                if self.road[i,j] != 0:
                    count += 1
        return count
    
    cpdef np.float64_t get_ave_trips(self):
        return self.total_trips/(self.density*self.roadlength*self.num_lanes*self.frac_tnv)
    
#####################################################################
    
    cpdef int get_num_full_tnvs(self):
        cdef Vehicle veh
        cdef int count = 0
        for veh in self.vehicle_array:
            if type(veh) == TNV:
                if veh.num_passengers == self.max_passengers:
                    count += 1
        return count
    
    cpdef int get_wait_counter(self):
        cdef Vehicle veh
        cdef int wait_count = 0
        for veh in self.vehicle_array:
            if type(veh) == TNV:
                wait_count = veh.wait_counter
        return wait_count
    
    cpdef int get_trips(self):
        cdef Vehicle veh
        cdef int trip_count = 0
        for veh in self.vehicle_array:
            if type(veh) == TNV:
                trip_count = veh.trip
        return trip_count
    
    cpdef int get_trip_ends(self):
        cdef Vehicle veh
        cdef int trip_end_count = 0
        for veh in self.vehicle_array:
            if type(veh) == TNV:
                trip_end_count = veh.trip_end
        return trip_end_count
    
    cpdef int get_onboard_pass(self):
        cdef Vehicle veh
        cdef int pass_count = 0
        for veh in self.vehicle_array:
            if type(veh) == TNV:
                pass_count = veh.num_passengers
        return pass_count
    
    cpdef float get_rng2(self):
        cdef Vehicle veh
        cdef float rng_count = 0
        for veh in self.vehicle_array:
            if type(veh) == TNV and veh.num_passengers > 0 and veh.trip >= veh.trip_end:
                rng_count = veh.rng2
        return rng_count
    
    cpdef float get_rng1(self):
        cdef Vehicle veh
        cdef float rng_count = 0
        for veh in self.vehicle_array:
            if type(veh) != TNV:
                rng_count = veh.rng
        return rng_count
    
    cpdef float get_rng(self):
        cdef Vehicle veh
        cdef float rng_count = 0
        for veh in self.vehicle_array:
            rng_count = veh.rng
        return rng_count
    
    cpdef int get_wait_time(self):
        cdef Vehicle veh
        cdef int wait_time_count = 0
        for veh in self.vehicle_array:
            if type(veh) == TNV:
                wait_time_count = veh.tnv_wait_time
        return wait_time_count
#####################################################################    
    cpdef list get_on_road_list(self):
        return self.vehs_on_road

    cpdef list get_full_tnv_list(self):
        return self.full_tnvs
    
    cpdef list get_wait_count_list(self):
        return self.wait_counter_array
    
    cpdef list get_trips_list(self):
        return self.trips
    
    cpdef list get_trip_ends_list(self):
        return self.trip_ends
    
    cpdef list get_onboard_pass_list(self):
        return self.onboard_pass_list
    
    cpdef list get_rng2_list(self):
        return self.rng2_list
    
    cpdef list get_rng1_list(self):
        return self.rng1_list
    
    cpdef list get_rng_list(self):
        return self.rng_list
    
    cpdef list get_wait_time_list(self):
        return self.wait_time_list
    
    cpdef float get_ave_speed(self):
        return np.mean(self.speeds)
#####################################################################
    cpdef np.int_t[:,:] get_road(self):
        cdef np.int_t[:,:] road
        cdef Vehicle veh
        road = np.zeros((self.num_lanes, self.roadlength), dtype=int)
        for veh in self.vehicle_array:
            road[veh.lane, veh.pos] = veh.marker
        return road

    cdef Vehicle get_vehicle(self, int veh_id):
        cdef Vehicle veh
        for veh in self.vehicle_array:
            if veh.id == veh_id:
                return veh