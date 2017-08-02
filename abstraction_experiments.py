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
from simple_rl.mdp.StateClass import State
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction
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
        return ActionAbstraction(options=actions, prim_actions=actions)

    return action_abs.aa_helpers.make_greedy_options(mdp_distr)

def get_directed_aa(mdp_distr, state_abs, incl_prim_actions=False):
    '''
    Args:
        mdp_distr (dict)
        state_abs (StateAbstraction)
        incl_prim_actions (bool)

    Returns:
        (ActionAbstraction)
    '''
    # Compute directed options for mdp/mdp_distr.
    if type(mdp_distr) is dict:
        first_mdp = mdp_distr.keys()[0]
    else:
        first_mdp = mdp_distr
    directed_options = action_abs.aa_helpers.get_directed_options_for_sa(first_mdp, state_abs)

    if not directed_options:
        # No good option set found.
        return False

    if incl_prim_actions:
        # Include the primitives.
        aa = ActionAbstraction(options=first_mdp.get_actions(), prim_actions=first_mdp.get_actions())
        for o in directed_options:
            aa.add_option(o)
        return aa
    else:
        # Return just the options.
        return ActionAbstraction(options=directed_options, prim_actions=first_mdp.get_actions())

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

def get_directed_option_sa_pair(mdp):
    '''
    Args:
        mdp (MDP)
        make_new_abs (bool)

    Returns:
        (StateAbstraction, ActionAbstraction)
    '''

     # Get Abstractions by iterating over epsilons.
    found_small_option_set = False
    sa_epsilon, sa_eps_incr = 0.00, 0.01

    while True:
        print "Epsilon:", sa_epsilon

        # Compute the SA-AA pair.
        sa = get_sa(mdp, make_new_sa=True, default=False, epsilon=sa_epsilon)

        if sa.get_num_abstr_states() == 1:
            # We can't have only 1 abstract state.
            print "Error: only 1 abstract state."
            quit()

        aa = get_directed_aa(mdp, sa)
        if aa:
            # If this is a good aa, we're done.
            break

        sa_epsilon += sa_eps_incr

    print "\nFound", len(aa.get_actions()), "Options."

    return sa, aa

def get_abstractions(mdp, directed=True):
    '''
    Args:

    Returns:
        (StateAbstraction, ActionAbstraction)
    '''
    if directed:
        return get_directed_option_sa_pair(mdp)
    else:
        sa = get_sa(mdp)
        aa = get_aa(mdp)
        return sa, aa

def write_datum_to_file(exp_dir, datum, file_name):
    '''
    Args:
        exp_dir (str)
        datum (object)
        file_name (str)

    Summary:
        Writes @datum to the file stored in join(exp_dir, file_name).
    '''
    if not os.path.isdir("results/" + exp_dir + "/"):
        os.makedirs("results/" + exp_dir)
    out_file = open("results/" + exp_dir + "/" + file_name + ".csv", "a+")
    out_file.write(str(datum) + ",")
    out_file.close()

def print_aa(action_abstr, state_space):
    '''
    Args:
        action_abstr (ActionAbstraction)
        state_space (list of State)

    Summary:
        Prints out options in a convenient way.
    '''

    options = action_abstr.get_actions()
    for o in options:
        inits = [s for s in state_space if o.is_init_true(s)]
        terms = [s for s in state_space if o.is_term_true(s)]
        print o
        print "\tinit:",
        for s in inits:
            print s,
        print
        print "\tterm:",
        for s in terms:
            print s,
        print
        print

def main():

    # MDP Setting.
    multi_task = True
    mdp_class = "grid"

    # Single Task.
    mdp = make_mdp.make_mdp(mdp_class=mdp_class)
    actions = mdp.actions
    gamma = mdp.gamma

    if multi_task:
        # Multi Task
        mdp = make_mdp.make_mdp_distr(mdp_class=mdp_class, num_mdps=4)
        actions = mdp.keys()[0].actions
        gamma = mdp.keys()[0].gamma

    # Grab SA and AA for each abstraction agent.   
    directed_sa, directed_aa = get_abstractions(mdp, directed=True)
    regular_sa, regular_aa = directed_sa, get_aa(mdp, actions, default=True)

    # Make Agents.
    rand_agent = RandomAgent(actions)
    rmax_agent = RMaxAgent(actions, gamma=gamma, s_a_threshold=1)
    ql_agent = QLearnerAgent(actions, gamma=gamma)

    # Make Abstraction Agents.
    abs_rmax_agent_directed = AbstractionWrapper(RMaxAgent, actions, state_abs=directed_sa, action_abs=directed_aa, name_ext="sa+aa")
    abs_rmax_agent = AbstractionWrapper(RMaxAgent, actions, state_abs=directed_sa, action_abs=regular_aa, name_ext="sa")
    abs_ql_agent_directed = AbstractionWrapper(QLearnerAgent, actions, state_abs=directed_sa, action_abs=directed_aa, name_ext="sa+aa")
    abs_ql_agent = AbstractionWrapper(QLearnerAgent, actions, state_abs=directed_sa, action_abs=regular_aa, name_ext="sa")
    
    # agents = [abs_rmax_agent_directed, abs_rmax_agent, rmax_agent]
    agents = [abs_ql_agent_directed, ql_agent]

    # Run experiments.
    if multi_task:
        run_agents_multi_task(agents, mdp, task_samples=20, episodes=1000, steps=20)
    else:
        run_agents_on_mdp(agents, mdp, instances=20, episodes=1000, steps=75)

if __name__ == "__main__":
    main()
