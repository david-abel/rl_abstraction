#!/usr/bin/env python

# Python imports.
import random as r
from collections import defaultdict
import os

# Other imports.
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.utils.ValueIterationClass import ValueIteration
from AbstractionWrapperClass import AbstractionWrapper
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction
from simple_rl.mdp.StateClass import State
import state_abs
import action_abs
import make_mdp

# -------------------------
# --- Make Abstractions ---
# -------------------------

def get_sa(mdp, make_new_sa=True, default=False, epsilon=0.0):
    '''
    Args:
        mdp (MDP)
        make_new_sa (bool)
        default (bool): If true, returns a blank StateAbstraction
        epsilon (float): Determines approximation for clustering.

    Returns:
        (StateAbstraction)
    '''

    if default:
        return StateAbstraction()

    # Make State Abstraction.
    should_save_sa = False
    q_equiv_sa = StateAbstraction(state_class=State)
    if make_new_sa:
        state_abs.sa_helpers.make_and_save_sa(mdp, state_class=State, epsilon=epsilon)
    else:
        mdp_name = str(mdp)
        if type(mdp) is dict:
            mdp_name = str("multitask-") + str(mdp.keys()[0])
        q_equiv_sa = state_abs.sa_helpers.load_sa(str(mdp_name) + ".p")
        if q_equiv_sa is None:
            state_abs.sa_helpers.make_and_save_sa(mdp, state_class=State)


    return q_equiv_sa

def get_aa(mdp_distr, actions, default=False):
    '''
    Args:
        mdp (defaultdict)
        actions (list of str)
        default (bool): If true, returns a blank ActionAbstraction

    Returns:
        (ActionAbstraction)
    '''
    
    if default:
        return ActionAbstraction(options=actions)
    # Test.
    new_aa = action_abs.aa_helpers.make_greedy_options(mdp_distr)

    # Default.
    return new_aa

def get_directed_aa(mdp_distr, state_abs):
    '''
    Args:
        mdp_distr (dict)
        state_abs (StateAbstraction)

    Returns:
        (ActionAbstraction)
    '''
    if type(mdp_distr) is dict:
        first_mdp = mdp_distr.keys()[0]
    else:
        first_mdp = mdp_distr
    directed_options = action_abs.aa_helpers.get_directed_options_for_sa(first_mdp, state_abs)

    if not directed_options:
        return False

    return ActionAbstraction(options=directed_options)

def compare_planning_abstr(mdp, abstr_mdp):
    '''
    Args;
        mdp (MDP)
        abstr_mdp (MDP)

    Returns:
        (int, int): num iters for VI on mdp vs. abstr_mdp
    '''
    # Run VI
    vi = ValueIteration(mdp, delta=0.001, max_iterations=1000)
    iters, value = vi.run_vi()

    abstr_vi = ValueIteration(abstr_mdp, delta=0.001, max_iterations=1000)
    abstr_iters, abstr_value = vi.run_vi()

    return iters, abstr_iters

def write_datum_to_file(exp_dir, datum, file_name):
    if not os.path.isdir("results/" + exp_dir + "/"):
        os.makedirs("results/" + exp_dir)
    out_file = open("results/" + exp_dir + "/" + file_name + ".csv", "a+")
    out_file.write(str(datum) + ",")
    out_file.close()


def main():

    # MDP Setting.
    multi_task = False
    mdp_class = "four_room"

    # Single Task.
    mdp = make_mdp.make_mdp(mdp_class=mdp_class)
    actions = mdp.actions
    gamma = mdp.gamma

    # Multi Task
    if multi_task:
        mdp_distr = make_mdp.make_mdp_distr(mdp_class=mdp_class, num_mdps=4)
        actions = mdp_distr.keys()[0].actions
        gamma = mdp_distr.keys()[0].gamma

    # Get Abstractions.
    found_small_option_set = False
    sa_epsilon, sa_eps_incr = 0.0, 0.01
    while True:
        print "\tEpsilon:", sa_epsilon
        sa = get_sa(mdp, make_new_sa=True, default=False, epsilon=sa_epsilon)
        aa = get_directed_aa(mdp, sa)

        if aa:
            break

        if not aa:
            sa_epsilon += sa_eps_incr

    # aa = get_aa(mdp_distr, actions, default=True)

    print "Found", len(aa.get_actions()), "Options."

    # Make Agents.
    rand_agent = RandomAgent(actions)
    rmax_agent = RMaxAgent(actions, gamma=gamma, s_a_threshold=1)
    ql_agent = QLearnerAgent(actions, gamma=gamma)

    # Make Abstraction Agents.
    abstr_random_agent = AbstractionWrapper(RandomAgent, actions, state_abs=sa, action_abs=aa)
    abs_rmax_agent = AbstractionWrapper(RMaxAgent, actions, state_abs=sa, action_abs=aa)
    abs_ql_agent = AbstractionWrapper(QLearnerAgent, actions, state_abs=sa, action_abs=aa)
    
    agents = [abs_ql_agent, ql_agent, abs_rmax_agent, rmax_agent]

    # Run experiments.
    if multi_task:
        run_agents_multi_task(agents, mdp_distr, instances=200, num_switches=1, steps=250, clear_old_results=False)
    else:
        run_agents_on_mdp(agents, mdp, instances=3, episodes=100, steps=20, clear_old_results=True)


if __name__ == "__main__":
    main()
