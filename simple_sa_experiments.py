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
from abstraction_experiments import get_sa

def main():

    # Grab experiment params.
    mdp_class = "four_room"
    task_samples = 100
    episodes = 200
    steps = 250
    grid_dim = 15
    gamma = 0.95

    # ========================
    # === Make Environment ===
    # ========================
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=grid_dim)
    actions = environment.get_actions()
    environment.set_gamma(gamma)
    sa_qds_test = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.00)
    sa_hand_test = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.00)
    sa_rand_test = get_sa(environment, indic_func=ind_funcs._four_rooms, epsilon=0.00)

    # ===================
    # === Make Agents ===
    # ===================
    rand_agent = RandomAgent(actions)
    baseline_agent = QLearnerAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    sa_qds_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_qds_test, name_ext="-$\phi_{Q_d^*}$")
    sa_hand_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_hand_test, name_ext="-$\phi_u$")
    sa_rand_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_rand_test, name_ext="-$\phi_h$")
    agents = [baseline_agent, rand_agent, sa_qds_agent, sa_hand_agent, sa_rand_agent]

    # Run!
    run_agents_multi_task(agents, environment, task_samples=task_samples, steps=steps, episodes=episodes, reset_at_terminal=True)


if __name__ == "__main__":
	main()
