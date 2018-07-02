#!/usr/bin/env python

# Python imports.
import os
import time

# Other imports.
from utils import make_mdp
from simple_rl.planning import ValueIteration
from utils.AbstractValueIterationClass import AbstractValueIteration
from state_abs import indicator_funcs as ind_funcs
from abstraction_experiments import get_sa


def clear_files(dir_name):
    '''
    Args:
        dir_name (str)

    Summary:
        Removes all csv files in @dir_name.
    '''
    for extension in ["iters", "times"]:
        dir_w_extension = os.path.join(dir_name, extension)  # , mdp_type) + ".csv"
        if not os.path.exists(dir_w_extension):
            os.makedirs(dir_w_extension)

        for mdp_type in ["vi", "vi-$\phi_{Q_d^*}$"]:
            if os.path.exists(os.path.join(dir_w_extension, mdp_type) + ".csv"):
                os.remove(os.path.join(dir_w_extension, mdp_type) + ".csv")


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
    # Grab experiment params.
    # Switch between Upworld and Trench
    mdp_class = "upworld"
    # mdp_class = "trench"
    grid_lim = 20 if mdp_class == 'upworld' else 7
    gamma = 0.95
    vanilla_file = "vi.csv"
    sa_file = "vi-$\phi_{Q_d^*}.csv"
    file_prefix = "results/planning-" + mdp_class + "/"
    clear_files(dir_name=file_prefix)

    for grid_dim in xrange(3, grid_lim):
        # ======================
        # == Make Environment ==
        # ======================
        environment = make_mdp.make_mdp(mdp_class=mdp_class, grid_dim=grid_dim)
        environment.set_gamma(gamma)

        # =======================
        # == Make Abstractions ==
        # =======================
        sa_qds = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.01)

        # ============
        # == Run VI ==
        # ============
        vanilla_vi = ValueIteration(environment, delta=0.0001, sample_rate=15)
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

        write_datum(file_prefix + "iters/" + vanilla_file, vanilla_iters)
        write_datum(file_prefix + "iters/" + sa_file, sa_iters)

        write_datum(file_prefix + "times/" + vanilla_file, vanilla_time)
        write_datum(file_prefix + "times/" + sa_file, sa_time)


if __name__ == "__main__":
    main()
