#!/usr/bin/env python

# Python imports.
import random as r
from collections import defaultdict
import os
import argparse

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearningAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_lifelong, run_agents_on_mdp
from simple_rl.tasks import TaxiOOMDP
from simple_rl.mdp import State, MDPDistribution
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction
import state_abs
import action_abs
from state_abs import indicator_funcs as ind_funcs

# -----------------------
# -- Make Abstractions --
# -----------------------

def get_abstractions(mdp, indic_func, directed=True, max_options=100):
    '''
    Args:
        mdp (MDP or MDPDistribution)
        indic_func (lambda): Property tester for the state abstraction.
        directed (bool)
        max_options (int)

    Returns:
        (StateAbstraction, ActionAbstraction)
    '''
    if directed:
        return get_directed_option_sa_pair(mdp, indic_func=indic_func, max_options=max_options)
    else:
        sa = get_sa(mdp, indic_func=indic_func)
        aa = get_aa(mdp)
        return sa, aa

def get_directed_option_sa_pair(mdp_distr, indic_func, max_options=100):
    '''
    Args:
        mdp_distr (MDPDistribution)
        indic_func
        max_options (int)

    Returns:
        (StateAbstraction, ActionAbstraction)
    '''

    # Get Abstractions by iterating over epsilons.
    found_small_option_set = False
    sa_epsilon, sa_eps_incr = 0.1, 0.01

    if isinstance(mdp_distr.get_all_mdps()[0], TaxiOOMDP):
        sa_epsilon = 0.02

    if "whirlpool" in str(mdp_distr.get_all_mdps()[0]):
        sa_eps_incr = 0.002

    if "color" in str(mdp_distr.get_all_mdps()[0]):
        sa_epsilon = 0.00

    while sa_epsilon <= 1.0 / (1 - mdp_distr.get_gamma()):
        print "Epsilon:", sa_epsilon

        # Compute the SA-AA pair.
        # NOTE: Track act_opt_pr is TRUE
        sa = get_sa(mdp_distr, indic_func=indic_func, default=False, epsilon=sa_epsilon, track_act_opt_pr=False)

        if sa.get_num_abstr_states() == 1:
            # We can't have only 1 abstract state.
            print "Abstraction Error: only 1 abstract state."
            quit()

        aa = get_directed_aa(mdp_distr, sa, max_options=max_options)
        if aa:
            # If this is a good aa, we're done.
            break

        sa_epsilon += sa_eps_incr

    print "\nFound", len(aa.get_actions()), "Options."

    return sa, aa

# ------------------------
# -- State Abstractions --
# ------------------------

def get_sa(mdp_distr, indic_func=None, default=False, epsilon=0.0):
    '''
    Args:
        mdp_distr (MDPDistributon)
        indicator_func (lambda): Indicator function from state_abs/indicator_funcs.py
        default (bool): If true, returns a blank StateAbstraction
        epsilon (float): Determines approximation for clustering.

    Returns:
        (StateAbstraction)
    '''

    if default:
        return StateAbstraction(phi={})

    state_abstr = state_abs.sa_helpers.make_sa(mdp_distr, indic_func=indic_func, state_class=State, epsilon=epsilon)

    return state_abstr

def compute_pac_sa(mdp_distr, indic_func=None, default=False, phi_epsilon=0.05, pac_delta=0.2):
    '''
    Args:
        mdp_distr (MDPDistributon)
        indicator_func (lambda): Indicator function from state_abs/indicator_funcs.py
        default (bool): If true, returns a blank StateAbstraction
        phi_epsilon (float): Determines approximation for Q^*_epsilon clustering.
        pac_delta (float): Determines how confident the resulting p_hat should be.

    Returns:
        (StateAbstraction)
    '''

    state_abstr = state_abs.sa_helpers.get_pac_sa_from_samples(mdp_distr, indic_func=indic_func, phi_epsilon=epsilon, pac_delta=0.2)

    return state_abstr

# -------------------------
# -- Action Abstractions --
# -------------------------

def get_aa(mdp_distr, default=False):
    '''
    Args:
        mdp (defaultdict)
        default (bool): If true, returns a blank ActionAbstraction

    Returns:
        (ActionAbstraction)
    '''
    
    if default:
        return ActionAbstraction(options=mdp_distr.get_actions(), prim_actions=mdp_distr.get_actions())

    return action_abs.aa_helpers.make_greedy_options(mdp_distr)

def get_directed_aa(mdp_distr, state_abs, incl_prim_actions=False, max_options=100):
    '''
    Args:
        mdp_distr (dict)
        state_abs (StateAbstraction)
        incl_prim_actions (bool)
        max_options (int)

    Returns:
        (ActionAbstraction)
    '''
    directed_options = action_abs.aa_helpers.get_directed_options_for_sa(mdp_distr, state_abs, incl_self_loops=True, max_options=max_options)
    term_prob = 1 - mdp_distr.get_gamma()

    if not directed_options:
        # No good option set found.
        return False

    if incl_prim_actions:
        # Include the primitives.
        aa = ActionAbstraction(options=mdp_distr.get_actions(), prim_actions=mdp_distr.get_actions(), prims_on_failure=False, term_prob=term_prob)
        for o in directed_options:
            aa.add_option(o)
        return aa
    else:
        # Return just the options.
        return ActionAbstraction(options=directed_options, prim_actions=mdp_distr.get_actions(), prims_on_failure=True, term_prob=term_prob)


def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", type = str, default = "octo", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-samples", type = int, default = 500, nargs = '?', help = "Number of samples from the MDP Distribution.")
    parser.add_argument("-steps", type = int, default = 100, nargs = '?', help = "Number of steps for the experiment.")
    parser.add_argument("-episodes", type = int, default = 1, nargs = '?', help = "Number of episodes for the experiment.")
    parser.add_argument("-grid_dim", type = int, default = 11, nargs = '?', help = "Dimensions of the grid world.")
    parser.add_argument("-track_options", type = bool, default = False, nargs = '?', help = "Plot in terms of option executions (if True).")
    parser.add_argument("-agent", type = str, default='ql', nargs = '?', help = "Specify agent class (one of {'ql', 'rmax'})..")
    parser.add_argument("-max_options", type = int, default=50, nargs = '?', help = "Specify maximum number of options.")
    parser.add_argument("-exp_type", type = str, default="core", nargs = '?', help = "Choose which experiment we're running. One of {core, combo}.")
    args = parser.parse_args()

    return args.task, args.samples, args.episodes, args.steps, args.grid_dim, bool(args.track_options), args.agent, args.max_options, args.exp_type

def main():

    # Grab experiment params.
    mdp_class, task_samples, episodes, steps, grid_dim, x_axis_num_options, agent_class_str, max_options, exp_type = parse_args()

    gamma = 0.9

    # ========================
    # === Make Environment ===
    # ========================
    multi_task = True
    max_option_steps = 50 if x_axis_num_options else 0
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=grid_dim) if multi_task else make_mdp.make_mdp(mdp_class=mdp_class)
    actions = environment.get_actions()
    environment.set_gamma(gamma)

    # Indicator functions.
    v_indic = ind_funcs._v_approx_indicator
    q_indic = ind_funcs._q_eps_approx_indicator
    v_disc_indic = ind_funcs._v_disc_approx_indicator
    rand_indic = ind_funcs._random

    # =========================
    # === Make Abstractions ===
    # =========================

        # Directed Variants.
    v_directed_sa, v_directed_aa = get_abstractions(environment, v_disc_indic, directed=True, max_options=max_options)
    # v_directed_sa, v_directed_aa = get_abstractions(environment, v_indic, directed=True, max_options=max_options)

        # Identity action abstraction.
    identity_sa, identity_aa = get_sa(environment, default=True), get_aa(environment, default=True)

    if exp_type == "core":
        # Core only abstraction types.
        q_directed_sa, q_directed_aa = get_abstractions(environment, q_indic, directed=True, max_options=max_options)
        rand_directed_sa, rand_directed_aa = get_abstractions(environment, rand_indic, directed=True, max_options=max_options)
        pblocks_sa, pblocks_aa = get_sa(environment, default=True), action_abs.aa_baselines.get_policy_blocks_aa(environment, incl_prim_actions=True, num_options=max_options)

    # ===================
    # === Make Agents ===
    # ===================

    # Base Agents.
    agent_class = QLearningAgent if agent_class_str == "ql" else RMaxAgent
    rand_agent = RandomAgent(actions)
    baseline_agent = agent_class(actions, gamma=gamma)

    if mdp_class == "pblocks":
        baseline_agent.epsilon = 0.01

    # Abstraction Extensions.
    agents = []
    vabs_agent_directed = AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=v_directed_sa, action_abstr=v_directed_aa, name_ext="v-sa+aa")

    if exp_type == "core":
        # Core only agents.
        qabs_agent_directed = AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=q_directed_sa, action_abstr=q_directed_aa, name_ext="q-sa+aa")
        rabs_agent_directed = AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=rand_directed_sa, action_abstr=rand_directed_aa, name_ext="rand-sa+aa")
        pblocks_agent = AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=pblocks_sa, action_abstr=pblocks_aa, name_ext="pblocks")
        agents = [vabs_agent_directed, qabs_agent_directed, rabs_agent_directed, pblocks_agent, baseline_agent]
    elif exp_type == "combo":
        # Combo only agents.
        aa_agent = AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=identity_sa, action_abstr=v_directed_aa, name_ext="aa")
        sa_agent = AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=v_directed_sa, action_abstr=identity_aa, name_ext="sa")
        agents = [vabs_agent_directed, sa_agent, aa_agent, baseline_agent]

    # Run experiments.
    if multi_task:
        steps = 999999 if x_axis_num_options else steps
        run_agents_multi_task(agents, environment, task_samples=task_samples, steps=steps, episodes=episodes, reset_at_terminal=True)
    else:
        run_agents_on_mdp(agents, environment, instances=20, episodes=30, reset_at_terminal=True)


if __name__ == "__main__":
    main()