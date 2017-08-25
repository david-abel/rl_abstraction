#!/usr/bin/env python

# Python imports.
import random as r
from collections import defaultdict
import os
import argparse


# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.mdp import State, MDPDistribution
import AbstractionWrapperClass as AWC
from state_abs.StateAbstractionClass import StateAbstraction
from action_abs.ActionAbstractionClass import ActionAbstraction
import state_abs
import action_abs
from state_abs import indicator_funcs as ind_funcs

# -------------------------
# --- Make Abstractions ---
# -------------------------

def get_sa(mdp_distr, indic_func=None, default=False, epsilon=0.0, track_act_opt_pr=False):
    '''
    Args:
        mdp_distr (MDPDistributon)
        indicator_func (lambda): Indicator function from state_abs/indicator_funcs.py
        default (bool): If true, returns a blank StateAbstraction
        epsilon (float): Determines approximation for clustering.
        track_act_prob (bool): If true, the state abstraction keeps track
            of the probability of each action's probability in ground states.

    Returns:
        (StateAbstraction)
    '''

    if default:
        return StateAbstraction()

    state_abstr = state_abs.sa_helpers.make_sa(mdp_distr, indic_func=indic_func, state_class=State, epsilon=epsilon, track_act_opt_pr=track_act_opt_pr)

    return state_abstr

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

def get_directed_option_sa_pair(mdp, indic_func, max_options=100):
    '''
    Args:
        mdp (MDP) or (MDPDistribution)
        indic_func
        max_options (int)

    Returns:
        (StateAbstraction, ActionAbstraction)
    '''

     # Get Abstractions by iterating over epsilons.
    found_small_option_set = False
    sa_epsilon, sa_eps_incr = 0.00, 0.01

    while True:
        print "Epsilon:", sa_epsilon

        # Compute the SA-AA pair.
        # NOTE: Track act_opt_pr is TRUE
        sa = get_sa(mdp, indic_func=indic_func, default=False, epsilon=sa_epsilon, track_act_opt_pr=False)

        if sa.get_num_abstr_states() == 1:
            # We can't have only 1 abstract state.
            print "Error: only 1 abstract state."
            quit()

        aa = get_directed_aa(mdp, sa, max_options=max_options)
        if aa:
            # If this is a good aa, we're done.
            break

        sa_epsilon += sa_eps_incr

    print "\nFound", len(aa.get_actions()), "Options."

    return sa, aa

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
        print


def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", type = str, default = "octo", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-samples", type = int, default = 500, nargs = '?', help = "Number of samples from the MDP Distribution.")
    parser.add_argument("-steps", type = int, default = 100, nargs = '?', help = "Number of steps for the experiment.")
    parser.add_argument("-grid_dim", type = int, default = 11, nargs = '?', help = "Dimensions of the grid world.")
    parser.add_argument("-track_options", type = bool, default = False, nargs = '?', help = "Plot in terms of option executions (if True).")
    parser.add_argument("-agent", type = str, default='ql', nargs = '?', help = "Specify agent class (one of {'ql', 'rmax'})..")
    args = parser.parse_args()

    return args.task, args.samples, args.steps, args.grid_dim, bool(args.track_options), args.agent

def main():

    # Grab experiment params.
    mdp_class, task_samples, steps, grid_dim, x_axis_num_options, agent_class_str = parse_args()

    # ========================
    # === Make Environment ===
    # ========================
    multi_task = True
    max_option_steps = 50 if x_axis_num_options else 0
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=grid_dim) if multi_task else make_mdp.make_mdp(mdp_class=mdp_class)
    actions = environment.get_actions()
    gamma = environment.get_gamma()

    # Indicator functions.
    v_indic = ind_funcs._v_approx_indicator
    q_indic = ind_funcs._q_eps_approx_indicator
    rand_indic = ind_funcs._random

    # =========================
    # === Make Abstractions ===
    # =========================

        # Directed Variants.
    max_options = 50
    q_directed_sa, q_directed_aa = get_abstractions(environment, q_indic, directed=True, max_options=max_options)
    v_directed_sa, v_directed_aa = get_abstractions(environment, v_indic, directed=True, max_options=max_options)
    rand_directed_sa, rand_directed_aa = get_abstractions(environment, rand_indic, directed=True, max_options=max_options)

        # Policy Blocks.
    pblocks_sa, pblocks_aa = get_sa(environment, default=True), action_abs.aa_baselines.get_policy_blocks_aa(environment, incl_prim_actions=True)

        # Identity action abstraction.
    identity_sa, identity_aa = get_sa(environment, default=True), get_aa(environment, default=True)

    # ===================
    # === Make Agents ===
    # ===================

    # Base Agents.
    agent_class = QLearnerAgent if agent_class_str == "ql" else RMaxAgent
    rand_agent = RandomAgent(actions)
    baseline_agent = agent_class(actions, gamma=gamma)

    # Abstraction Extensions.
    qabs_agent_directed = AWC.AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=q_directed_sa, action_abstr=q_directed_aa, name_ext="q-sa+aa")
    vabs_agent_directed = AWC.AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=v_directed_sa, action_abstr=v_directed_aa, name_ext="v-sa+aa")
    rabs_agent_directed = AWC.AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=rand_directed_sa, action_abstr=rand_directed_aa, name_ext="rand-sa+aa")
    aa_agent = AWC.AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=identity_sa, action_abstr=v_directed_aa, name_ext="aa")
    sa_agent = AWC.AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=v_directed_sa, action_abstr=identity_aa, name_ext="sa")
    pblocks_agent = AWC.AbstractionWrapper(agent_class, actions, str(environment), max_option_steps=max_option_steps, state_abstr=pblocks_sa, action_abstr=pblocks_aa, name_ext="pblocks")
    core_agents = [vabs_agent_directed, qabs_agent_directed, rabs_agent_directed, pblocks_agent, baseline_agent]

    # table_agents = [vabs_agent_directed, sa_agent, aa_agent, baseline_agent]
    # agents = [qabs_ql_agent_directed]

    # if mdp_class == "four_room":
    #     # Add handmade four room if needed.
    #     hand_directed_sa, hand_directed_aa = get_abstractions(environment, ind_funcs._four_rooms, directed=True)
    #     habs_ql_agent_directed = AWC.AbstractionWrapper(agent_class, actions, state_abstr=hand_directed_sa, action_abstr=hand_directed_aa, name_ext="hand-sa+aa")
    #     agents.append(habs_ql_agent_directed)

    # Run experiments.
    if multi_task:
        # This doesn't make sense... Need some way to terminate.
        steps = 999999 if x_axis_num_options else steps
        run_agents_multi_task(core_agents, environment, task_samples=task_samples, steps=steps, episodes=1, reset_at_terminal=True)
    else:
        run_agents_on_mdp(agents, environment, instances=20, episodes=100, reset_at_terminal=True)


if __name__ == "__main__":
    main()
