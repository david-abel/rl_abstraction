# Python imports.
from collections import defaultdict
import cPickle
import os
import sys

# Other imports.
from simple_rl.utils.ValueIterationClass import ValueIteration
from StateAbstractionClass import StateAbstraction
from simple_rl.mdp.StateClass import State

def merge_state_abs(list_of_sa):
    '''
    Args:
        list_of_sa (list of StateAbstraction)

    Returns:
        (StateAbstraction)
    '''
    merged = list_of_sa[0]
    
    for sa in list_of_sa:
        merged = merged + sa

    return merged

def _q_eps_approx_indicator(state_x, state_y, vi, mdp, epsilon=0.00):
    '''
    Args:
        state_x (State)
        state_y (State)
        vi (ValueIteration)
        mdp (MDP)

    Returns:
        (bool): true iff:
            max |Q(state_x,a) - Q(state_y, a)| <= epsilon
    '''
    for a in mdp.actions:
        q_x = vi.get_q_value(state_x, a)
        q_y = vi.get_q_value(state_y, a)

        if abs(q_x - q_y) > epsilon:
            return False

    return True

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
        state_abstrs.append(make_sa(mdp, _q_eps_approx_indicator))

    # Merge
    merged_sa = merge_state_abs(state_abstrs)

    # Visualize on the expected MDP.
    avged_mdp = GridWorldMDP(width=width, height=height, init_loc=(1, 1), goal_locs=goal_locs)
    visualize_mdp(avged_mdp, merged_sa.phi, file_name="abstr-mdp.png")

def make_and_save_sa(mdp, state_class=State, epsilon=0.0):
    '''
    Args:
        mdp (MDP)
        state_class (Class)
        epsilon (float)

    Summary:
        Creates and saves a state abstraction.
    '''
    print "  Making and saving Q equivalence state abstraction... "
    q_equiv_sa = StateAbstraction()
    mdp_name = str(mdp)
    if not type(mdp) is dict:
        q_equiv_sa = make_sa(mdp, _q_eps_approx_indicator, state_class=state_class, epsilon=epsilon)
    else:
        q_equiv_sa = make_multitask_sa(mdp, _q_eps_approx_indicator, state_class=state_class, epsilon=epsilon)
        mdp_name = "multitask-" + str(mdp.keys()[0])
    save_sa(q_equiv_sa, str(mdp_name) + ".p")

def make_multitask_sa(mdp_distr, indicator_func, state_class, epsilon=0.0):
    '''
    Args:
        mdp_distr (defaultdict)
        indicator_func (S x S --> {0,1})
        state_class (Class)
        epsilon (float)

    Returns:
        (StateAbstraction)
    '''
    sa_list = []
    for mdp in mdp_distr.keys():
        sa = make_sa(mdp, indicator_func, state_class, epsilon)

        sa_list += [sa]

    return merge_state_abs(sa_list)

def make_sa(mdp, indicator_func, state_class, epsilon=0.0):
    '''
    Args:
        mdp (MDP)
        indicator_func (S x S --> {0,1})
        state_class (Class)
        epsilon (float)

    Returns:
        (StateAbstraction)
    '''

    print "\tRunning VI...",
    sys.stdout.flush()
    # Run VI
    vi = ValueIteration(mdp, delta=0.0001, max_iterations=5000)
    iters, val = vi.run_vi()
    print " done."

    print "\tMaking state abstraction...",
    sys.stdout.flush()
    sa = StateAbstraction(state_class=state_class)
    clusters = defaultdict(list)
    num_states = len(vi.get_states())
    # Find state pairs that satisfy the condition.
    for i, state_x in enumerate(vi.get_states()):
        clusters[state_x] = [state_x]
        for state_y in vi.get_states():
            if not (state_x == state_y) and indicator_func(state_x, state_y, vi, mdp, epsilon=epsilon):
                clusters[state_x].append(state_y)
                clusters[state_y].append(state_x)

    # Build SA.
    for i, state in enumerate(clusters.keys()):
        new_cluster = clusters[state]
        sa.make_cluster(new_cluster)

        # Destroy old so we don't double up.
        for s in clusters[state]:
            if s in clusters.keys():
                clusters.pop(s) #

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