import sys
import numpy as np

from action_abs.ActionAbstractionClass import ActionAbstraction
from action_abs.CovPredicate import CovPredicate
from action_abs.EqPredicateClass import EqPredicate
from action_abs.OptionClass import Option
from action_abs.PolicyFromDictClass import make_dict_from_lambda
from simple_rl.planning import ValueIteration


def get_policy_blocks_aa(mdp_distr, num_options=10, task_samples=20, incl_prim_actions=False):
    pb_options = make_policy_blocks_options(mdp_distr, num_options=num_options, task_samples=task_samples)

    if type(mdp_distr) is dict:
        first_mdp = mdp_distr.keys()[0]
    else:
        first_mdp = mdp_distr

    if incl_prim_actions:
        # Include the primitives.
        aa = ActionAbstraction(options=first_mdp.get_actions(), prim_actions=first_mdp.get_actions())
        for o in pb_options:
            aa.add_option(o)
        return aa
    else:
        # Return just the options.
        return ActionAbstraction(options=pb_options, prim_actions=first_mdp.get_actions())


def policy_blocks_merge_pair(pi1, pi2):
    '''
    Perform pairwise merge between two partial policies to compute their intersection (also a partial policy)
    :param pi1: Partial policy 1
    :param pi2: Partial policy 2
    :return: Merged partial policy
    '''
    ret = {}
    for state in set(pi1.keys() + pi2.keys()):
        a1 = pi1.get(state, None)
        a2 = pi2.get(state, None)
        if a1 == a2 and a1 is not None:
            ret[state] = a1
    return ret

def policy_blocks_subtract_pair(pi1, pi2):
    '''
    Perform pairwise subtraction between two partial policies to compute their difference (also a partial policy)
    :param pi1: Partial policy 1
    :param pi2: Partial policy 2
    :return: Difference partial policy
    '''
    ret = {}
    for state in set(pi1.keys() + pi2.keys()):
        a1 = pi1.get(state, None)
        a2 = pi2.get(state, None)
        if a1 is not None and a2 is None:
            ret[state] = a1
    return ret


def policy_blocks_merge(policy_set):
    '''

    :param policy_set: Set of policies to merge
    :return: partial policy representing the merge of all policies in the policy set
    '''
    policy_set = list(policy_set)
    merged = policy_set[0]
    for i in range(1, len(policy_set)):
        merged = policy_blocks_merge_pair(merged, policy_set[i])
    return merged


def policy_blocks_contains_pairwise(containee, container):
    '''
    Determine if one policy is contained within another
    :param containee: Policy that may be contained
    :param container: Policy that is a container
    :return: True if containee(s) = container(s) for all s in containee, otherwise False
    '''
    for state in containee.keys():
        a = container.get(state, None)
        if a is None:
            return False

        if a != containee[state]:
            return False
    return True


def policy_blocks_num_contains_policy(pi, policy_set):
    '''
    Compute the number of policies in policy set which contain the candidate policy
    :param pi: Policy to search for within policy set
    :param policy_set: Set of policies to check containment
    :return: Number of policies in policy set which contain the candidate policy (value in [0, len(policy_set)]
    '''
    ret = 0
    for policy in policy_set:
        if policy_blocks_contains_pairwise(pi, policy):
            ret += 1
    return ret


def policy_blocks_score_policy(pi_unscored, policy_set):
    '''
    The score is the size of the partial policy multiplied by the number of solution policies that contain it
    :return: Quick metric for determining how good this option policy is with respect to the sampled initial tasks
    '''
    scale_contain = policy_blocks_num_contains_policy(pi_unscored, policy_set)
    return len(pi_unscored.keys()) * scale_contain


def get_power_set(policy_set):
    '''
    Compute subset of the power set of the input policy set by computing all possible pairs and triples
    :param policy_set: Set to compute partial power set
    :return: Partial power set consisting of all pairwise and triplet policy combinations
    '''
    print 'Computing partial power set over solution policies...'
    ret = []
    # Compute all pairs
    for i in xrange(len(policy_set)-1):
        for j in xrange(i+1, len(policy_set)):
            ret.append([policy_set[i], policy_set[j]])

    print 'Finished computing all pairs...starting triples...'

    # Generate all triplets
    for i in xrange(len(ret)):
        for p in policy_set:
            # Sort to maintain consistency for duplicate checks
            to_add = sorted(ret[i] + [p])
            if p not in ret[i] and to_add not in ret:
                ret.append(to_add)

    return ret

def make_policy_blocks_options(mdp_distr, num_options, task_samples):
    option_set = []
    # Fill solution set for task_samples draws from MDP distribution
    L = []
    for new_task in xrange(task_samples):
        print "  Sample " + str(new_task + 1) + " of " + str(task_samples) + "."

        # Sample the MDP.
        mdp_id = np.random.multinomial(1, mdp_distr.values()).tolist().index(1)
        mdp = mdp_distr.keys()[mdp_id]

        # Run VI to get a policy for the MDP as well as the list of states
        print "\tRunning VI...",
        sys.stdout.flush()
        # Run VI
        vi = ValueIteration(mdp, delta=0.0001, max_iterations=5000)
        iters, val = vi.run_vi()
        print " done."

        policy = make_dict_from_lambda(vi.policy, vi.get_states())
        L.append(policy)

    power_L = get_power_set(L)
    num_iters = 1
    print 'Beginning policy blocks for {2} options with {0} solution policies and power set of size {1}'\
        .format(len(L), len(power_L), num_options)

    while len(power_L) > 0 and len(option_set) < num_options:
        print 'Running iteration {0} of policy blocks...'.format(num_iters)
        # Initialize empty set of candidate option policies
        C = []
        # Iterate over the power set of solution policies
        for policy_set in power_L:
            # Compute candidate policy as merge over policy set
            candidate = policy_blocks_merge(policy_set)
            if candidate not in C:
                # Compute score of each candidate policy
                C.append((candidate, policy_blocks_score_policy(candidate, L)))
        # Identify the candidate policy with highest score and add to option set
        C = sorted(C, key=lambda x: x[1])
        pi_star = C[-1][0]
        if pi_star not in option_set:
            option_set.append(pi_star)

        # Subtract chosen candidate from L by iterating through power set
        power_L = map(lambda policy_set: [policy_blocks_subtract_pair(p, pi_star) for p in policy_set], power_L)

        # Remove empty elements of power set
        power_L = filter(lambda policy_set: sum(map(lambda x: len(x), policy_set)) > 0, power_L)

        num_iters += 1

    # Generate true option set
    ret = []
    for o in option_set:
        init_predicate = CovPredicate(y=True, policy=o)
        term_predicate = CovPredicate(y=False, policy=o)
        print map(str, o.keys())
        print o.values()
        print '**'
        opt = Option(init_predicate=init_predicate, term_predicate=term_predicate, policy=o)
        ret.append(opt)

    print 'Policy blocks returning with {0} options'.format(len(ret))

    return ret
