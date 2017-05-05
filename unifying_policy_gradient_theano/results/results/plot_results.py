import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt



eps = range(2991)
eps2 = range(2798)

ddpg_hopper = genfromtxt('/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/polyrl_results/DDPG_Hopper/Hopper/progress.csv', delimiter=',')
ddpg_hopper_avg_rwd = ddpg_hopper[1:, 22]
ddpg_hopper_max_return = ddpg_hopper[1:, 17]
ddpg_hopper_std_return = ddpg_hopper[1:, 13]


ddpg_humanoid_simple = genfromtxt('/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/polyrl_results/DDPG_Humanoid/Humanoid_Simple/progress.csv', delimiter=',')
ddpg_humanoid_avg_rwd = ddpg_humanoid_simple[1:, 5]
ddpg_humanoid_max_return = ddpg_humanoid_simple[1:, 2]
ddpg_humanoid_std_return = ddpg_humanoid_simple[1:, 1]


ddpg_humanoid_complex = genfromtxt('/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/polyrl_results/DDPG_Humanoid_Complex/Humanoid_Complex/progress.csv', delimiter=',')
ddpg_humanoid_complex_avg_rwd = ddpg_humanoid_complex[1:, 2]
ddpg_humanoid_complex_max_return = ddpg_humanoid_complex[1:, 15]
ddpg_humanoid_complex_std_return = ddpg_humanoid_complex[1:, 3]


ddpg_walker = genfromtxt('/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/polyrl_results/DDPG_Walker/Walker/progress.csv', delimiter=',')
ddpg_walker_avg_rwd = ddpg_walker[1:, 10]
ddpg_walker_max_return = ddpg_walker[1:, 15]
ddpg_walker_std_return = ddpg_walker[1:, 18]



ddpg_halfcheetah = genfromtxt('/Users/Riashat/Documents/PhD_Research/PolyRL/rllab/polyrl_results/DDPG_HalfCheetah/HalfCheetah/progress.csv', delimiter=',')
ddpg_halfcheetah_avg_rwd = ddpg_halfcheetah[1:, 22]
ddpg_halfcheetah_max_return = ddpg_halfcheetah[1:,13]
ddpg_halfcheetah_std_return = ddpg_halfcheetah[1:,16]



 

unified_ddpg_hopper = genfromtxt('/Users/Riashat/Documents/PhD_Research/Unified_Policy_Gradient/results/Unified_Policy_Gradients/Unified_DDPG_Hopper/progress.csv', delimiter=',')
unified_ddpg_hopper_avg_rwd = unified_ddpg_hopper[1:, 9]
unified_ddpg_hopper_max_return = unified_ddpg_hopper[1:, 12]
unified_ddpg_hopper_std_return = unified_ddpg_hopper[1:, 23]


unified_ddpg_walker = genfromtxt('/Users/Riashat/Documents/PhD_Research/Unified_Policy_Gradient/results/Unified_Policy_Gradients/Unified_DDPG_Walker/progress.csv', delimiter=',')
unified_ddpg_walker_avg_rwd = unified_ddpg_walker[1:, 8]
unified_ddpg_walker_max_return = unified_ddpg_walker[1:,6]
unified_ddpg_walker_std_return = unified_ddpg_walker[1:,1]


unified_ddpg_humanoid_simple = genfromtxt('/Users/Riashat/Documents/PhD_Research/Unified_Policy_Gradient/results/Unified_Policy_Gradients/Unified_DDPG_Humanoid/progress.csv', delimiter=',')
unified_ddpg_humanoid_avg_rwd = unified_ddpg_humanoid_simple[1:, 10]
unified_ddpg_humnaoid_max_return = unified_ddpg_humanoid_simple[1:, 20]
unified_ddpg_humanoid_std_return = unified_ddpg_humanoid_simple[1:, 8]





def single_plot(stats1, smoothing_window=50, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps2, rewards_smoothed_1, label="Unified On-Policy and Off-Policy DDPG")    
    plt.fill_between( eps2, rewards_smoothed_1 + unified_ddpg_walker_std_return,   rewards_smoothed_1 - unified_ddpg_walker_std_return, alpha=0.3, edgecolor='blue', facecolor='blue')

    plt.legend(handles=[cum_rwd_1])
    plt.xlabel("Epsiode")
    plt.ylabel("Average Return")
    plt.title("Walker Environment")
  
    plt.show()
    
    return fig



def multipe_plot(stats1, stats2, smoothing_window=50, noshow=False):

    fig = plt.figure(figsize=(30, 20))
    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()

    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, label="DDPG")    
    plt.fill_between( eps, rewards_smoothed_1 + ddpg_walker_std_return,   rewards_smoothed_1 - ddpg_walker_std_return, alpha=0.3, edgecolor='blue', facecolor='blue')

    cum_rwd_2, = plt.plot(eps2, rewards_smoothed_2, label="Unified DDPG")    
    plt.fill_between( eps2, rewards_smoothed_2 + unified_ddpg_walker_std_return,   rewards_smoothed_2 - unified_ddpg_walker_std_return, alpha=0.3, edgecolor='blue', facecolor='red')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2])
    plt.xlabel("Epsiode")
    plt.ylabel("Average Return")
    plt.title("Walker Environment")
  
    plt.show()
    
    return fig






def main():

   #single_plot(unified_ddpg_walker_avg_rwd)
   multipe_plot(ddpg_walker_avg_rwd, unified_ddpg_walker_avg_rwd)





if __name__ == '__main__':
    main()