#!/usr/bin/env python

# Python imports.
import random
from collections import defaultdict
import os

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_multi_task
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction
from AbstractionWrapperClass import AbstractionWrapper
from state_abs import indicator_funcs as ind_funcs
from abstraction_experiments import get_sa, get_directed_option_sa_pair

def get_sa_experiment_agents(environment, gamma):
    '''
    Args:
        environment (simple_rl.MDPDistribution)
        gamma (float)

    Returns:
        (list)
    '''

    actions = environment.get_actions()

    # State Abstractions.
    sa_qds_test = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.00)
    sa_hand_test = get_sa(environment, indic_func=ind_funcs._four_rooms, epsilon=0.00)
    sa_rand_test = get_sa(environment, indic_func=ind_funcs._random, epsilon=0.00)

    # QLearners.
    ql_agent = QLearnerAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    ql_sa_qds_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_qds_test, name_ext="$\phi_{Q_d^*}$")
    ql_sa_hand_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_hand_test, name_ext="$\phi_u$")

    # R-Max.
    rm_agent = RMaxAgent(actions, gamma=gamma)
    rm_sa_qds_agent = AbstractionWrapper(RMaxAgent, actions, str(environment), state_abstr=sa_qds_test, name_ext="$\phi_{Q_d^*}$")
    rm_sa_hand_agent = AbstractionWrapper(RMaxAgent, actions, str(environment), state_abstr=sa_hand_test, name_ext="$\phi_u$")

    agents = [ql_agent, ql_sa_qds_agent, ql_sa_hand_agent, rm_agent, rm_sa_qds_agent, rm_sa_hand_agent]

    return agents

def get_combo_experiment_agents(environment, gamma):
    '''
    Args:
        environment (simple_rl.MDPDistribution)
        gamma (float)

    Returns:
        (list)
    '''
    actions = environment.get_actions()

    sa, aa = get_directed_option_sa_pair(environment, indic_func=ind_funcs._q_disc_approx_indicator, max_options=100)

    # QLearner.
    ql_agent = QLearnerAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)

    # Combos.
    sa_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa, name_ext="sa")
    aa_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), action_abstr=aa, name_ext="aa")
    sa_aa_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa, action_abstr=aa, name_ext="sa+aa")

    agents = [ql_agent, sa_agent, aa_agent, sa_aa_agent]

    return agents

def main():

    # Grab experiment params.
    mdp_class = "four_room"
    task_samples = 5
    episodes = 200
    steps = 250
    grid_dim = 7
    gamma = 0.95
    experiment_type = "combo" # One of {"sa","combo"}.

    # ========================
    # === Make Environment ===
    # ========================
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=grid_dim)
    environment.set_gamma(gamma)

    # ===================
    # === Make Agents ===
    # ===================
    agents = []
    if experiment_type == "sa":
        # SA experiment.
        agents = get_sa_experiment_agents(environment, gamma=gamma)
    elif experiment_type == "combo":
        # AA experiment.
        agents = get_combo_experiment_agents(environment, gamma=gamma)
    else:
        print "Experiment Error: experiment type unknown (" + experiment_type + "). Must be one of {sa, combo}."
        quit()

    # Run!
    run_agents_multi_task(agents, environment, task_samples=task_samples, steps=steps, episodes=episodes, reset_at_terminal=True)

if __name__ == "__main__":
	main()
