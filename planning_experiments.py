#!/usr/bin/env python

# Python imports.
import random
from collections import defaultdict
import os
import time

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_multi_task
from simple_rl.tasks import FourRoomMDP
from simple_rl.planning import ValueIteration
from AbstractValueIterationClass import AbstractValueIteration
from hierarch.make_abstr_mdp import make_abstr_mdp
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction
from AbstractionWrapperClass import AbstractionWrapper
from state_abs import indicator_funcs as ind_funcs
from abstraction_experiments import get_sa, get_directed_option_sa_pair

def clear_files(dir_name="plan_results"):
    '''
    Args:
        dir_name (str)

    Summary:
        Removes all csv files in @dir_name.
    '''
    for extension in ["iters", "times"]:
        for mdp_type in ["vi", "vi+sa", "vi+aa", "vi+sa+aa"]:
            if os.path.exists(os.path.join(dir_name, extension, mdp_type) + ".csv"):
                os.remove(os.path.join(dir_name, extension, mdp_type) + ".csv")

def write_datum(file_name, datum):
    '''
    Args:
        file_name (str)
        datum (object)
    '''
    out_file = open(file_name, "a+")
    out_file.write(str(datum) + ",")
    out_file.close()

def main():

    clear_files()

    # Grab experiment params.
    mdp_class = "hall"
    max_grid_dim = 20
    gamma = 0.95
    vanilla_file = "vi.csv"
    sa_file = "vi+sa.csv"

    for grid_dim in xrange(6, 21):
        # ======================
        # == Make Environment ==
        # ======================
        environment = make_mdp.make_mdp(mdp_class=mdp_class, grid_dim=grid_dim)
        environment.set_gamma(gamma)

        # =======================
        # == Make Abstractions ==
        # =======================
        sa_qds = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.05)

        # ============
        # == Run VI ==
        # ============
        vanilla_vi = ValueIteration(environment, delta=0.0001, sample_rate=5)
        sa_vi = AbstractValueIteration(ground_mdp=environment, state_abstr=sa_qds)

        print "Running VIs."
        start_time = time.clock()
        vanilla_iters, vanilla_val = vanilla_vi.run_vi()
        vanilla_time = round(time.clock() - start_time, 2)
        
        start_time = time.clock()        
        sa_iters, sa_val = sa_vi.run_vi()
        sa_time = round(time.clock() - start_time, 2)

        print "vanilla", vanilla_iters, vanilla_val, vanilla_time
        print "sa:", sa_iters, sa_val, sa_time

        write_datum("results/iters/" + vanilla_file, vanilla_iters)
        write_datum("results/iters/" + sa_file, sa_iters)

        write_datum("results/times/" + vanilla_file, vanilla_time)
        write_datum("results/times/" + sa_file, sa_time)

if __name__ == "__main__":
    main()
