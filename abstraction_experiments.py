#!/usr/bin/env python

# Python imports.
import random as r
from collections import defaultdict
import os

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.mdp import State, MDPDistribution
from AbstractionWrapperClass import AbstractionWrapper
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

def get_directed_aa(mdp_distr, state_abs, incl_prim_actions=False):
    '''
    Args:
        mdp_distr (dict)
        state_abs (StateAbstraction)
        incl_prim_actions (bool)

    Returns:
        (ActionAbstraction)
    '''
    directed_options = action_abs.aa_helpers.get_directed_options_for_sa(mdp_distr, state_abs, inc_self_loops=True)

    if not directed_options:
        # No good option set found.
        return False

    if incl_prim_actions:
        # Include the primitives.
        aa = ActionAbstraction(options=mdp_distr.get_actions(), prim_actions=mdp_distr.get_actions(), prims_on_failure=True)
        for o in directed_options:
            aa.add_option(o)
        return aa
    else:
        # Return just the options.
        return ActionAbstraction(options=directed_options, prim_actions=mdp_distr.get_actions(), prims_on_failure=True)

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

def get_directed_option_sa_pair(mdp, indic_func):
    '''
    Args:
        mdp (MDP) or (MDPDistribution)
        indic_func

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

        aa = get_directed_aa(mdp, sa)
        if aa:
            # If this is a good aa, we're done.
            break

        sa_epsilon += sa_eps_incr

    print "\nFound", len(aa.get_actions()), "Options."

    return sa, aa

def get_abstractions(mdp, indic_func, directed=True):
    '''
    Args:
        mdp (MDP or MDPDistribution)

    Returns:
        (StateAbstraction, ActionAbstraction)
    '''
    if directed:
        return get_directed_option_sa_pair(mdp, indic_func=indic_func)
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

def main():

    # ========================
    # === Make Environment ===
    # ========================
    multi_task = True
    mdp_class = "four_room"
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=15) if multi_task else make_mdp.make_mdp(mdp_class=mdp_class)
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
    q_directed_sa, q_directed_aa = get_abstractions(environment, q_indic, directed=True)
    v_directed_sa, v_directed_aa = get_abstractions(environment, v_indic, directed=True)
    rand_directed_sa, rand_directed_aa = get_abstractions(environment, rand_indic, directed=True)

        # Policy Blocks.
    pblocks_sa, pblocks_aa = get_sa(environment, default=True), action_abs.aa_baselines.get_policy_blocks_aa(environment, incl_prim_actions=True)

        # Identity action abstraction.
    identity_aa = get_aa(environment, default=True)

    # ===================
    # === Make Agents ===
    # ===================

    # Base Agents.
    agent_class = QLearnerAgent
    rand_agent = RandomAgent(actions)
    ql_agent = QLearnerAgent(actions, gamma=gamma)

    # Abstraction Extensions.
    qabs_ql_agent_directed = AbstractionWrapper(agent_class, actions, state_abstr=q_directed_sa, action_abstr=q_directed_aa, name_ext="q-sa+aa")
    vabs_ql_agent_directed = AbstractionWrapper(agent_class, actions, state_abstr=v_directed_sa, action_abstr=v_directed_aa, name_ext="v-sa+aa")
    rabs_ql_agent_directed = AbstractionWrapper(agent_class, actions, state_abstr=rand_directed_sa, action_abstr=rand_directed_aa, name_ext="rand-sa+aa")
    abs_ql_agent = AbstractionWrapper(agent_class, actions, state_abstr=q_directed_sa, action_abstr=identity_aa, name_ext="sa")
    pblocks_ql_agent = AbstractionWrapper(agent_class, actions, state_abstr=pblocks_sa, action_abstr=pblocks_aa, name_ext="pblocks")
    agents = [qabs_ql_agent_directed, vabs_ql_agent_directed, rabs_ql_agent_directed, pblocks_ql_agent, abs_ql_agent, ql_agent]

    if mdp_class == "four_room":
        # Add handmade four room if needed.
        hand_directed_sa, hand_directed_aa = get_abstractions(environment, ind_funcs._four_rooms, directed=True)
        habs_ql_agent_directed = AbstractionWrapper(agent_class, actions, state_abstr=hand_directed_sa, action_abstr=hand_directed_aa, name_ext="hand-sa+aa")
        agents.append(habs_ql_agent_directed)

    # Run experiments.
    if multi_task:
        run_agents_multi_task(agents, environment, task_samples=400, steps=500, episodes=1, reset_at_terminal=True)
    else:
        run_agents_on_mdp(agents, environment, instances=20, episodes=100, reset_at_terminal=True)


if __name__ == "__main__":
    main()
