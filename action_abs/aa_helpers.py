# Python imports.
from collections import defaultdict
import Queue
import random as r
import os
import sys
import cPickle

# Other imports.
from ActionAbstractionClass import ActionAbstraction
from OptionClass import Option
from simple_rl.utils.ValueIterationClass import ValueIteration
from simple_rl.mdp.MDPClass import MDP
from EqPredicateClass import EqPredicate
from PolicyFromDictClass import *

def get_directed_options_for_sa(mdp, state_abstr, opt_size_limit=100):
    '''
    Args:
        mdp (MDP)
        state_abstr (StateAbstraction)

    Returns:
        (ActionAbstraction)
    '''

    print "  Computing directed options."
    sys.stdout.flush()
    
    abs_states = state_abstr.get_abs_states()
    g_start_state = mdp.get_init_state()

    # Compute all directed options that transition between abstract states.
    options = []
    state_pairs = []
    random_policy = lambda s : r.choice(mdp.actions)
    for s_a in abs_states:
        for s_a_prime in abs_states:
            if not(s_a == s_a_prime):
                init_predicate = EqPredicate(y=s_a, func=state_abstr.phi)
                term_predicate = EqPredicate(y=s_a_prime, func=state_abstr.phi)
                o = Option(init_predicate=init_predicate,
                            term_predicate=term_predicate,
                            policy = random_policy)

                options.append(o)
                state_pairs.append((s_a, s_a_prime))

    if len(options) > max(state_abstr.get_num_ground_states() / 3.0, 100):
        print "\tToo many options (" + str(len(options)) + "). Increasing epsilon and continuing.\n"
        return False

    print "\tMade", len(options), "options (formed clique over S_A)."

    print "\tPruning...",
    sys.stdout.flush()

    pruned_option_set = _prune_non_directed_options(options, state_pairs, state_abstr, mdp)

    print "done. Reduced to", len(pruned_option_set), "options."

    return pruned_option_set

def _prune_non_directed_options(options, state_pairs, state_abstr, mdp):
    '''
    Args:
        Options(list)
        state_pairs (list)
        state_abstr (StateAbstraction)
        mdp (MDP)

    Returns:
        (list of Options)

    Summary:
        Removes redundant options. That is, if o_1 goes from s_A1 to s_A2, and
        o_2 goes from s_A1 *through s_A2 to s_A3, then we get rid of o_2.
    '''

    good_options = []

    for i, o in enumerate(options):
        pre_abs_state, post_abs_state = state_pairs[i]

        ground_init_states = state_abstr.get_ground_states_in_abs_state(pre_abs_state)
        ground_term_states = state_abstr.get_ground_states_in_abs_state(post_abs_state)

        def _directed_option_reward_lambda(s, a):
            s_prime = mdp.get_transition_func()(s,a)
            return int(s_prime in ground_term_states and not s in ground_term_states)

        def new_trans_func(s, a):
            original = s.is_terminal()
            s.set_terminal(s in ground_term_states)
            s_prime = mdp.get_transition_func()(s,a)
            s.set_terminal(original)
            return s_prime


        rand_init_g_state = r.choice(ground_init_states)
        mini_mdp = MDP(actions=mdp.actions,
                        init_state=rand_init_g_state,
                        transition_func=new_trans_func,
                        reward_func=_directed_option_reward_lambda)

        # Solve the MDP defined by the terminal abstract state.
        mini_mdp_vi = ValueIteration(mini_mdp, delta=0.0001, max_iterations=5000)
        mini_mdp_vi.run_vi()
        o_policy_dict = make_dict_from_lambda(mini_mdp_vi.policy, state_abstr.get_ground_states())
        o_policy = PolicyFromDict(o_policy_dict)

        # Compute overlap w.r.t. plans from each state.
        good_option = False
        for init_g_state in ground_init_states:
            # Prune overlapping ones.
            plan, state_seq = mini_mdp_vi.plan(init_g_state)
            
            opt_name = str(ground_init_states[0]) + "-" + str(ground_term_states[0])
            o.set_name(opt_name)
            options[i] = o

            if not _check_overlap(o, state_seq, options):
                # The option overlaps, don't include it.
                good_option = True
                break

        if good_option:
            # Give the option the new directed policy and name.
            o.set_policy(o_policy.get_action)
            good_options.append(o)

    return good_options

def _check_overlap(option, state_seq, options):
    '''
    Args:
        state_seq (list of State)
        options

    Returns:
        (bool): If true, we should remove this option.
    '''
    terminal_is_reachable = False

    for i, s_g in enumerate(state_seq):
        for o_prime in options:
            is_in_middle = (not option.is_term_true(s_g)) and (not option.is_init_true(s_g))
            if is_in_middle and o_prime.is_init_true(s_g):
                # We should get rid of @option, because it's path goes through another init.
                return True
            
            # Only keep options whose terminal states are reachable from the initiation set.
            if option.is_term_true(s_g):
                terminal_is_reachable = True

    if not terminal_is_reachable:
        return True

    return False

def compute_sub_opt_func_for_mdp_distr(mdp_distr):
    '''
    Args:
        mdp_distr (dict)

    Returns:
        (list): Contains the suboptimality function for each MDP in mdp_distr.
            subopt: V^*(s) - Q^(s,a)
    '''
    actions = mdp_distr.keys()[0].get_actions()
    sub_opt_funcs = []

    i = 0
    for mdp in mdp_distr.keys():
        print "\t mdp", i + 1, "of", len(mdp_distr.keys())
        vi = ValueIteration(mdp, delta=0.001, max_iterations=1000)
        iters, value = vi.run_vi()

        new_sub_opt_func = defaultdict(float)
        for s in vi.get_states():
            max_q = float("-inf")
            for a in actions:
                next_q = vi.get_q_value(s, a)
                if next_q > max_q:
                    max_q = next_q

            for a in actions:
                new_sub_opt_func[(s, a)] = max_q - vi.get_q_value(s,a)

        sub_opt_funcs.append(new_sub_opt_func)
        i+=1

    return sub_opt_funcs

def _compute_agreement(sub_opt_funcs, mdp_distr, state, action, epsilon=0.00):
    '''
    Args:
        sub_opt_funcs (list of dicts)
        mdp_distr (dict)
        state (simple_rl.State)
        action (str)
        epsilon (float)

    Returns:
        (list)

    Summary:
        Computes the MDPs for which @action is epsilon-optimal in @state.
    '''
    all_sub_opt_vals = [sof[(state, action)] for sof in sub_opt_funcs]
    eps_opt_mdps = [int(sov <= epsilon) for sov in all_sub_opt_vals]

    return eps_opt_mdps

def add_next_option(mdp_distr, next_decis_state, sub_opt_funcs):
    '''
    Args:

    Returns:
        (Option)
    '''

    # Init func and terminal func.
    init_func = lambda s : s == next_decis_state
    term_func = lambda s : True
    term_func_states = []

    # Misc. 
    reachable_states = Queue.Queue()
    reachable_states.put(next_decis_state)
    visited_states = set([next_decis_state])
    policy_dict = defaultdict(str)
    actions = mdp_distr.keys()[0].get_actions()
    transition_func = mdp_distr.keys()[0].get_transition_func()

    # Tracks which MDPs share near-optimal action sequences.
    mdps_active = [1 for m in range(len(sub_opt_funcs))]

    while not reachable_states.empty():
        # Pointers for this iteration.
        cur_state = reachable_states.get()
        next_action = r.choice(actions)
        max_agreement = 0 # agreement for this state.

        # Compute action with max agreement (num active MDPs with shared eps-opt action.)
        for a in actions:
            agreement_ls = _compute_agreement(sub_opt_funcs, mdp_distr, cur_state, a)
            active_agreement_ls = [mdps_active[i] & agreement_ls[i] for i in range(len(agreement_ls))]
            agreement = sum(active_agreement_ls)
            if agreement > max_agreement:
                next_action = a
                max_agreement = agreement

        # Set policy for this state to the action with maximal agreement.
        policy_dict[cur_state] = next_action
        max_agreement_ls = _compute_agreement(sub_opt_funcs, mdp_distr, cur_state, next_action)
        mdps_active = [mdps_active[i] & max_agreement_ls[i] for i in range(len(max_agreement_ls))]
        agreement = sum(mdps_active)

        # Move to the next state according to max agreement action.
        next_state = transition_func(cur_state, next_action)

        if agreement <= 2 or next_state.is_terminal():
            term_func_states.append(next_state)

        if next_state not in visited_states:
            reachable_states.put(next_state)
            visited_states.add(next_state)

    if len(term_func_states):
        term_func_states.append(next_state)

    # Turn policy dict into a function and make the option.
    o = Option(init_func, term_func=term_func_states, policy=policy_dict)

    return o


def make_greedy_options(mdp_distr):
    '''
    Assumptions:
        Shared S, A, start state, T, gamma between all M in mdp_distr.
    '''

    if isinstance(mdp_distr, MDP):
        print "Warning: attempting to create options for a single MDP."
        mdp_distr = {1.0:mdp_distr}

    # Grab relevant MDP distr. components.
    init_state = mdp_distr.keys()[0].get_init_state()
    transition_func = mdp_distr.keys()[0].get_transition_func()
    actions = mdp_distr.keys()[0].get_actions()

    # Setup data structures.
    print "Computing advantage functions."
    sub_opt_funcs = compute_sub_opt_func_for_mdp_distr(mdp_distr)
    decision_states = Queue.Queue()
    decision_states.put(init_state)
    new_aa = ActionAbstraction(options=actions, prim_actions=actions)

    visited_states = set([init_state])
    # Loop over reachable states.
    num_options = 0
    print "Learning:"
    while num_options < 2 and (not decision_states.empty()):
        print "\toption", num_options + 1
        # Add option as long as we have a decision state.
        # A decision state is a state where we don't have a good option.
        
        next_decis_state = decision_states.get()
        o = add_next_option(mdp_distr, next_decis_state, sub_opt_funcs)
        new_aa.add_option(o)
        num_options += 1
        new_state = o.act_until_terminal(next_decis_state, transition_func)
        if new_state not in visited_states:
            decision_states.put(new_state)
            visited_states.add(new_state)

    return new_aa


def load_aa(file_name):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(this_dir + "/cached_aa/" + file_name):
        return cPickle.load( open( this_dir + "/cached_aa/" + file_name, "rb" ))
    else:
        print "Warning: no saved Action Abstraction found with name '" + file_name + "'."
        
def save_aa(aa, file_name):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    cPickle.dump(aa, open( this_dir + "/cached_aa/" + file_name, "w" ))

