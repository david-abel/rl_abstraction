#!/usr/bin/env python

# Python imports.
import random
from collections import defaultdict
import os
import time

# Other imports.
import make_mdp
from simple_rl.agents import RMaxAgent, DelayedQAgent
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.tasks import ChainMDP
from simple_rl.planning import ValueIteration
from state_abs.StateAbstractionClass import StateAbstraction
from abstraction_experiments import get_sa


def main():

    # Grab experiment params.
    mdp = ChainMDP(3)
    gamma = 0.95

    # =======================
    # == Make Abstractions ==
    # =======================
    sa_qds = get_sa(mdp, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.01)

      

if __name__ == "__main__":
    main()
