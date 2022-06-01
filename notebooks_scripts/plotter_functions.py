import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def qdens_plotter(densities, med_array, err_array, marker, in_color, linestyle, plot_label):
    final_err_erray = np.array(err_array).T
    plt.errorbar(densities, med_array,final_err_erray, marker = marker, markerfacecolor = in_color, linestyle = linestyle, color = 'black', markeredgecolor='black', ecolor=in_color, label=plot_label)
    
    
def df_extractor(df,num_trials):
    sorted_df = df.sort_values("density")
    q_arr = sorted_df['throughput'].to_numpy()
    mean_q_arr = np.mean(q_arr.reshape(-1, num_trials), axis=1)
    
    composite_list = [q_arr[x:x+num_trials] for x in range(0, len(q_arr),num_trials)]
    #composite_list
    errorbars = []

    for grp in range(len(composite_list)):
        errbars = []
        errbars.append(abs(np.percentile(composite_list[grp],25) - mean_q_arr[grp]))
        errbars.append(abs(np.percentile(composite_list[grp],75) - mean_q_arr[grp]))
        errorbars.append(errbars)

    results = {
        "q_arr" : q_arr,
        "mean_q_arr" : mean_q_arr,
        "errorbars" : errorbars
    }
    
    
    return results    
    
    
#def df_extractor(df,alpha_value,num_trials):
#    sorted_df = df[df['alpha'] == alpha_value].sort_values("density")
#    q_arr = sorted_df['throughput'].to_numpy()
#    mean_q_arr = np.mean(q_arr.reshape(-1, num_trials), axis=1)
    
#    composite_list = [q_arr[x:x+num_trials] for x in range(0, len(q_arr),num_trials)]
    #composite_list
#    errorbars = []

#    for grp in range(len(composite_list)):
#        errbars = []
#        errbars.append(abs(np.percentile(composite_list[grp],25) - mean_q_arr[grp]))
#        errbars.append(abs(np.percentile(composite_list[grp],75) - mean_q_arr[grp]))
#        errorbars.append(errbars)
#
#    results = {
#        "q_arr" : q_arr,
#        "mean_q_arr" : mean_q_arr,
#        "errorbars" : errorbars
#    }
    
    
#    return results