# Python imports.
from collections import defaultdict
import cPickle
import os
import sys
import itertools
import numpy as np

# Other imports.
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.mdp import State
from simple_rl.mdp import MDPDistribution
import indicator_funcs as ind_funcs
from StateAbstractionClass import StateAbstraction


def get_pac_sa_from_samples(mdp_distr, indic_func=ind_funcs._q_eps_approx_indicator, phi_epsilon=0.0, delta=0.2):
    '''
    Args:
        mdp_distr (MDPDistribution)
        indicator_func (S x S --> {0,1})
        epsilon (float)
        delta (float)

    Returns:
        (StateAbstraction)


    Summary:
        Computes a PAC state abstraction.
    '''
    sample_eps = 1.0
    pac_sample_bound = max(int(np.log(1 / delta) / sample_eps**2), 2)
    print("PAC sample bound:", pac_sample_bound)
    sa_list = []
    for sample in xrange(pac_sample_bound):
        mdp = mdp_distr.sample()
        sa = make_singletask_sa(mdp, indic_func, phi_epsilon) #, prob_of_mdp=mdp_distr.get_prob_of_mdp(mdp))
        sa_list += [sa]

    pac_state_abstr = merge_state_abs(sa_list)

    return pac_state_abstr

def merge_state_abs(list_of_sa):
    '''
    Args:
        list_of_sa (list of StateAbstraction)

    Returns:
        (StateAbstraction)
    '''
    merged = list_of_sa[0]

    for sa in list_of_sa[1:]:
        merged = merged + sa

    return merged

def compute_planned_state_abs(mdp_class="grid", num_mdps=30):
    '''
    Args:
        mdp_class (str)
        num_mdps (int)
    '''

    # Setup grid params for MDPs.
    goal_locs = []
    width, height = 7, 4
    for element in itertools.product(range(1, width + 1), [height]):
        goal_locs.append(element)

    # Compute the optimal Q^* abstraction for each MDP.
    state_abstrs = []
    for i in xrange(num_mdps):
        left = goal_locs[:len(goal_locs) / 2]
        right = goal_locs[len(goal_locs) / 2:]
        mdp = GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=r.choice([left, right]))
        state_abstrs.append(make_sa(mdp, ind_funcs._q_eps_approx_indicator))

    # Merge
    merged_sa = merge_state_abs(state_abstrs)

    return merged_sa

def make_sa(mdp, indic_func=ind_funcs._q_eps_approx_indicator, state_class=State, epsilon=0.0):
    '''
    Args:
        mdp (MDP)
        state_class (Class)
        epsilon (float)

    Summary:
        Creates and saves a state abstraction.
    '''
    print "  Making state abstraction... "
    new_sa = StateAbstraction(phi={})
    if isinstance(mdp, MDPDistribution):
        new_sa = make_multitask_sa(mdp, state_class=state_class, indic_func=indic_func, epsilon=epsilon)
    else:
        new_sa = make_singletask_sa(mdp, state_class=state_class, indic_func=indic_func, epsilon=epsilon)

    print "  (final SA) Num abstract states:", new_sa.get_num_abstr_states()

    return new_sa

def make_multitask_sa(mdp_distr, state_class=State, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.0, aa_single_act=True):
    '''
    Args:
        mdp_distr (MDPDistribution)
        state_class (Class)
        indicator_func (S x S --> {0,1})
        epsilon (float)
        aa_single_act (bool): If we should track optimal actions.

    Returns:
        (StateAbstraction)
    '''
    sa_list = []
    for mdp in mdp_distr.get_mdps():
        sa = make_singletask_sa(mdp, indic_func, state_class, epsilon, aa_single_act=aa_single_act, prob_of_mdp=mdp_distr.get_prob_of_mdp(mdp))
        sa_list += [sa]

    multitask_sa = merge_state_abs(sa_list)

    return multitask_sa

def make_singletask_sa(mdp, indic_func, state_class, epsilon=0.0, aa_single_act=False, prob_of_mdp=1.0):
    '''
    Args:
        mdp (MDP)
        indic_func (S x S --> {0,1})
        state_class (Class)
        epsilon (float)

    Returns:
        (StateAbstraction)
    '''

    print "\tRunning VI...",
    sys.stdout.flush()
    # Run VI
    if isinstance(mdp, MDPDistribution):
        mdp = mdp.sample()

    vi = ValueIteration(mdp)
    iters, val = vi.run_vi()
    print " done."

    print "\tMaking state abstraction...",
    sys.stdout.flush()
    sa = StateAbstraction(phi={}, state_class=state_class)
    clusters = defaultdict(set)
    num_states = len(vi.get_states())
    actions = mdp.get_actions()
    
    # Find state pairs that satisfy the condition.
    for i, state_x in enumerate(vi.get_states()):
        sys.stdout.flush()
        clusters[state_x].add(state_x)

        for state_y in vi.get_states()[i:]:
            if not(state_x == state_y) and indic_func(state_x, state_y, vi, actions, epsilon=epsilon):
                clusters[state_x].add(state_y)
                clusters[state_y].add(state_x)

    print "making clusters...",
    sys.stdout.flush()
    
    # Build SA.
    for i, state in enumerate(clusters.keys()):
        new_cluster = clusters[state]
        sa.make_cluster(new_cluster)

        # Destroy old so we don't double up.
        for s in clusters[state]:
            if s in clusters.keys():
                clusters.pop(s)
    
    print " done."
    print "\tGround States:", num_states
    print "\tAbstract:", sa.get_num_abstr_states()
    print

    return sa

# ------------ Indicator Functions ------------

def agent_q_estimate_equal(state_x, state_y, agent, state_abs, action_abs=[], epsilon=0.0):
    '''
    Args:
        state_x (State)
        state_y (State)
        agent (Agent)
        state_abs (StateAbstraction)
        action_abs (ActionAbstraction)

    Returns:
        (bool): true iff:
            max |agent.Q(state_x,a) - agent.Q(state_y, a)| <= epsilon
    '''
    for a in agent.actions:
        q_x = agent.get_q_value(state_abs(state_x), a)
        q_y = agent.get_q_value(state_abs(state_y), a)
        if abs(q_x - q_y) > epsilon:
            return False

    return True

def agent_always_false(state_x, state_y, agent):
    return False
