import csv
from multiprocessing import Pool
import itertools
from model_v_4_20 import Road
import numpy as np

#########################################################################################

def get_road(density, frac_tnv, alpha, strat_list, tnv_wait_time, glob_vars):
    #global variables
    roadlength = glob_vars['roadlength']
    vmax = glob_vars['vmax']
    p_slow = glob_vars['p_slow']
    periodic = glob_vars['periodic']
    sim_time = glob_vars['sim_time']
    trans_time = glob_vars['trans_time']
    num_lanes = glob_vars['num_lanes']
    #other variables
    p_1 = strat_list[0] #wander
    p_2 = strat_list[1] #disappear
    p_3 = 1-p_1-p_2     #hazard
     
    roads = []
    throughputs = []
    #full_tnvs = []
    road = Road(roadlength, num_lanes, vmax, alpha, 
                        frac_tnv, periodic, density, p_slow, p_1, p_2, tnv_wait_time)
    for t in range(sim_time+trans_time):
        road.timestep_parallel()
        if t >= trans_time:
            throughputs.append(road.throughput())
            temp_road = np.array(road.get_road())
            roads.append(temp_road[0])
            full_tnvs.append(road.get_num_full_tnvs)
            
            
            
    
    
    res = {
            "throughput": np.mean(throughputs),
            "frac_tnv": frac_tnv,
            "density": density,
            "true_density": road.get_density(),
            "alpha": alpha, 
            "p_slow": p_slow,
            "p_1": p_1,
            "p_2": p_2,
            "tnv_wait_time": tnv_wait_time,
            "ave_trips": road.get_ave_trips()
          }    
    return res

#########################################################################################

#########################################################################################

def simulate(density, frac_tnv, trial, alpha, strat_list, tnv_wait_time, glob_vars):
    #global variables
    roadlength = glob_vars['roadlength']
    vmax = glob_vars['vmax']
    p_slow = glob_vars['p_slow']
    periodic = glob_vars['periodic']
    sim_time = glob_vars['sim_time']
    trans_time = glob_vars['trans_time']
    num_lanes = glob_vars['num_lanes']
    #other variables
    p_1 = strat_list[0] #wander
    p_2 = strat_list[1] #disappear
    p_3 = 1-p_1-p_2     #hazard
     
    #roads = []
    throughputs = []
    #full_tnvs = []
    road = Road(roadlength, num_lanes, vmax, alpha, 
                        frac_tnv, periodic, density, p_slow, p_1, p_2, tnv_wait_time)
    for t in range(sim_time+trans_time):
        road.timestep_parallel()
        if t >= trans_time:
            throughputs.append(road.throughput())
            #temp_road = np.array(road.get_road())
            #roads.append(temp_road[0])
            #full_tnvs.append(road.get_num_full_tnvs)
            
            
            
    
    
    res = {
            "throughput": np.mean(throughputs),
            "frac_tnv": frac_tnv,
            "density": density,
            "true_density": road.get_density(),
            "trial": trial,
            "alpha": alpha, 
            "p_slow": p_slow,
            "p_1": p_1,
            "p_2": p_2,
            "tnv_wait_time": tnv_wait_time,
            "ave_trips": road.get_ave_trips()
          }    
    return res

#########################################################################################

def g(tup):
    return simulate(*tup)
p = Pool()

#########################################################################################

def sim_to_csv(filename, densities, frac_tnv, trials, alphas, strategies, tnv_wait_time, glob_vars):
    filepath = "data/"+filename+".csv"
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        lines = 0
        #for same argument, such as glob_vars, enclose in []
        for result in p.imap_unordered(g, itertools.product(densities, frac_tnv, trials, alphas, strategies, tnv_wait_time, glob_vars)):
            if lines == 0:
                writer.writerow(result.keys())
                lines += 1
            writer.writerow(result.values())