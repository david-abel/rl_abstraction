# Python imports.
from collections import defaultdict
import Queue
import random
import os
import sys
import cPickle

# Other imports.
from ActionAbstractionClass import ActionAbstraction
from OptionClass import Option
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.mdp.MDPClass import MDP
from EqPredicateClass import EqPredicate, NeqPredicate
from PolicyFromDictClass import *
from simple_rl.tasks import GridWorldMDP

# ----------------------
# -- Directed Options --
# ----------------------

def get_directed_options_for_sa(mdp_distr, state_abstr, incl_self_loops=False, max_options=100):
    '''
    Args:
        mdp_distr (MDPDistribution)
        state_abstr (StateAbstraction)
        incl_self_loops (bool)
        max_options (int)

    Returns:
        (ActionAbstraction)
    '''

    print "  Computing directed options."
    sys.stdout.flush()
    
    abs_states = state_abstr.get_abs_states()
    g_start_state = mdp_distr.get_init_state()

    # Compute all directed options that transition between abstract states.
    options = []
    state_pairs = []
    random_policy = lambda s : random.choice(mdp_distr.get_actions())
    for s_a in abs_states:
        for s_a_prime in abs_states:
            if not(s_a == s_a_prime):
                init_predicate = EqPredicate(y=s_a, func=state_abstr.phi)
                term_predicate = EqPredicate(y=s_a_prime, func=state_abstr.phi)
                o = Option(init_predicate=init_predicate,
                            term_predicate=term_predicate,
                            policy=random_policy)
                options.append(o)
                state_pairs.append((s_a, s_a_prime))

            elif incl_self_loops:
                # Self loop.
                init_predicate = EqPredicate(y=s_a, func=state_abstr.phi)
                term_predicate = NeqPredicate(y=s_a, func=state_abstr.phi) # Terminate in any other abstract state.
                o = Option(init_predicate=init_predicate,
                            term_predicate=term_predicate,
                            policy=random_policy)
                # Wait. This policy is insane. What?
                options.append(o)
                state_pairs.append((s_a, s_a_prime))

    if len(options) > max_options: #max(state_abstr.get_num_ground_states() / 3.0, max_options):
        print "\tToo many options (" + str(len(options)) + "). Increasing epsilon and continuing.\n"
        return False

    print "\tMade", len(options), "options (formed clique over S_A)."

    print "\tPruning...",
    sys.stdout.flush()

    pruned_option_set = _prune_non_directed_options(options, state_pairs, state_abstr, mdp_distr)

    print "done. Reduced to", len(pruned_option_set), "options."

    return pruned_option_set

def _prune_non_directed_options(options, state_pairs, state_abstr, mdp_distr):
    '''
    Args:
        Options(list)
        state_pairs (list)
        state_abstr (StateAbstraction)
        mdp_distr (MDPDistribution)

    Returns:
        (list of Options)

    Summary:
        Removes redundant options. That is, if o_1 goes from s_A1 to s_A2, and
        o_2 goes from s_A1 *through s_A2 to s_A3, then we get rid of o_2.
    '''


    good_options = []
    bad_options = []

    first_mdp = mdp_distr.get_all_mdps()[0]

    if isinstance(first_mdp, GridWorldMDP):
        original = first_mdp.slip_prob
        first_mdp.slip_prob = 0.0

    transition_func = first_mdp.get_transition_func()

    for i, o in enumerate(options):
        print "Option", i, "of", len(options)
        pre_abs_state, post_abs_state = state_pairs[i]

        ground_init_states = state_abstr.get_ground_states_in_abs_state(pre_abs_state)
        ground_term_states = state_abstr.get_ground_states_in_abs_state(post_abs_state)
        rand_init_g_state = random.choice(ground_init_states)

        def _directed_option_reward_lambda(s, a):
            s_prime = transition_func(s,a)
            return int(s_prime in ground_term_states and not s in ground_term_states)

        def new_trans_func(s, a):
            original = s.is_terminal()
            s.set_terminal(s in ground_term_states)
            s_prime = transition_func(s,a)
            s.set_terminal(original)
            return s_prime

        if pre_abs_state == post_abs_state:

            mini_mdp_init_states = defaultdict(list)

            # Self loop. Make an option per goal in the cluster.
            goal_mdps = []
            goal_state_action_pairs = defaultdict(list)
            for i, mdp in enumerate(mdp_distr.get_all_mdps()):
                add = False
    
                # Is there a goal for this MDP in one of the ground states.
                for s_g in ground_term_states:
                    for a in mdp.get_actions():
                        if mdp.get_reward_func()(s_g, a) > 0.0 and a not in goal_state_action_pairs[s_g]:
                            goal_state_action_pairs[s_g].append(a)
                            if isinstance(mdp, GridWorldMDP):
                                goals = tuple(mdp.get_goal_locs())
                            else:
                                goals = tuple(s_g)
                            mini_mdp_init_states[goals].append(s_g)
                            add = True

                if add:
                    goal_mdps.append(mdp)

            for goal_mdp in goal_mdps:

                def goal_new_trans_func(s, a):
                    original = s.is_terminal()
                    s.set_terminal(s not in ground_term_states or original)
                    s_prime = goal_mdp.get_transition_func()(s,a)
                    s.set_terminal(original)
                    return s_prime

                if isinstance(goal_mdp, GridWorldMDP):
                    cluster_init_state = random.choice(mini_mdp_init_states[tuple(goal_mdp.get_goal_locs())])
                else:
                    cluster_init_state = random.choice(ground_init_states)

                # Make a new 
                mini_mdp = MDP(actions=goal_mdp.get_actions(),
                        init_state=cluster_init_state,
                        transition_func=goal_new_trans_func,
                        reward_func=goal_mdp.get_reward_func())


                o_policy, mini_mdp_vi = _make_mini_mdp_option_policy(mini_mdp, state_abstr)

                # print goal_mdp.get_goal_locs()
                # for s_g in state_abstr.get_ground_states():
                #     print s_g, o_policy(s_g)

                new_option = Option(o.init_predicate, o.term_predicate, o_policy)
                new_option.set_name(str(ground_init_states[0]) + "-sl")
                good_options.append(new_option)

                if isinstance(goal_mdp, GridWorldMDP):
                    goal_mdp.slip_prob = original

            continue
        else:
            # This is a non-self looping option.
            mini_mdp = MDP(actions=mdp_distr.get_actions(),
                            init_state=rand_init_g_state,
                            transition_func=new_trans_func,
                            reward_func=_directed_option_reward_lambda)


            o_policy, mini_mdp_vi = _make_mini_mdp_option_policy(mini_mdp, state_abstr)
            # Compute overlap w.r.t. plans from each state.
            for init_g_state in ground_init_states:

                # Prune overlapping ones.
                plan, state_seq = mini_mdp_vi.plan(init_g_state)
                
                opt_name = str(ground_init_states[0]) + "-" + str(ground_term_states[0])
                o.set_name(opt_name)
                options[i] = o

                if not _check_overlap(o, state_seq, options, bad_options):
                    # Give the option the new directed policy and name.
                    o.set_policy(o_policy)
                    good_options.append(o)   
                    break
                else:
                    # The option overlaps, don't include it.
                    bad_options.append(o)

    if isinstance(first_mdp, GridWorldMDP):
        first_mdp.slip_prob = original

    return good_options

def _make_mini_mdp_option_policy(mini_mdp, state_abstr):
        '''
        Args:
            mini_mdp (MDP)
            state_abstr (StateAbstraction)

        Returns:
            Policy
        '''
        # Solve the MDP defined by the terminal abstract state.
        mini_mdp_vi = ValueIteration(mini_mdp, delta=0.0001, max_iterations=5000)
        iters, val = mini_mdp_vi.run_vi()

        o_policy_dict = make_dict_from_lambda(mini_mdp_vi.policy, state_abstr.get_ground_states())
        o_policy = PolicyFromDict(o_policy_dict)

        return o_policy.get_action, mini_mdp_vi

def _check_overlap(option, state_seq, options, bad_options):
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
            
            if o_prime in bad_options:
                continue

            is_in_middle = (not option.is_term_true(s_g)) and (not option.is_init_true(s_g))
            if is_in_middle and o_prime.is_init_true(s_g):
                # We should get rid of @option, because it's path goes through another init.
                return True
            
            # Only keep options whose terminal states are reachable from the initiation set.
            if option.is_term_true(s_g):
                terminal_is_reachable = True

    if not terminal_is_reachable:
        # Can't reach the terminal state.
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
    actions = mdp_distr.get_actions()
    sub_opt_funcs = []

    i = 0
    for mdp in mdp_distr.get_mdps():
        print "\t mdp", i + 1, "of", mdp_distr.get_num_mdps()
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
    actions = mdp_distr.get_actions()
    transition_func = mdp_distr.get_mdps()[0].get_transition_func()

    # Tracks which MDPs share near-optimal action sequences.
    mdps_active = [1 for m in range(len(sub_opt_funcs))]

    while not reachable_states.empty():
        # Pointers for this iteration.
        cur_state = reachable_states.get()
        next_action = random.choice(actions)
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
