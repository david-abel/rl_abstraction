# Python imports.
from collections import defaultdict
import cPickle
import os
import sys
import itertools

# Other imports.
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.mdp import State
from simple_rl.mdp import MDPDistribution
import indicator_funcs as ind_funcs
from StateAbstractionClass import StateAbstraction

def merge_state_abs(list_of_sa, track_act_opt_pr=False):
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

    # Visualize on the expected MDP.
    avged_mdp = GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=goal_locs)
    visualize_mdp(avged_mdp, merged_sa.phi, file_name="abstr-mdp.png")

def make_sa(mdp, indic_func=ind_funcs._q_eps_approx_indicator, state_class=State, epsilon=0.0, save=False, track_act_opt_pr=False):
    '''
    Args:
        mdp (MDP)
        state_class (Class)
        epsilon (float)

    Summary:
        Creates and saves a state abstraction.
    '''
    print "  Making state abstraction... "
    q_equiv_sa = StateAbstraction(phi={}, track_act_opt_pr=track_act_opt_pr)
    if isinstance(mdp, MDPDistribution):
        q_equiv_sa = make_multitask_sa(mdp, state_class=state_class, indic_func=indic_func, epsilon=epsilon, track_act_opt_pr=track_act_opt_pr)
    else:
        q_equiv_sa = make_singletask_sa(mdp, state_class=state_class, indic_func=indic_func, epsilon=epsilon, track_act_opt_pr=track_act_opt_pr)

    if save:
        save_sa(q_equiv_sa, str(mdp) + ".p")

    return q_equiv_sa

def make_multitask_sa(mdp_distr, state_class=State, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.0, aa_single_act=True, track_act_opt_pr=False):
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
        sa = make_singletask_sa(mdp, indic_func, state_class, epsilon, aa_single_act=aa_single_act, prob_of_mdp=mdp_distr.get_prob_of_mdp(mdp), track_act_opt_pr=track_act_opt_pr)
        sa_list += [sa]

    multitask_sa = merge_state_abs(sa_list, track_act_opt_pr=track_act_opt_pr)

    return multitask_sa

def make_singletask_sa(mdp, indic_func, state_class, epsilon=0.0, aa_single_act=False, prob_of_mdp=1.0, track_act_opt_pr=False):
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
    sa = StateAbstraction(phi={}, state_class=state_class, track_act_opt_pr=track_act_opt_pr)
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
    
    if aa_single_act:
        # Put all optimal actions in a set associated with the ground state.
        for ground_s in sa.get_ground_states():
            a_star_set = set(vi.get_max_q_actions(ground_s))
            sa.set_actions_state_opt_dict(ground_s, a_star_set, prob_of_mdp)

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

def load_sa(file_name):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(this_dir + "/cached_sa/" + file_name):
        return cPickle.load( open( this_dir + "/cached_sa/" + file_name, "rb" ) )
    else:
        print "Warning: no saved State Abstraction with name '" + file_name + "'."
        
def save_sa(sa, file_name):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    cPickle.dump( sa, open( this_dir + "/cached_sa/" + file_name, "w" ) )