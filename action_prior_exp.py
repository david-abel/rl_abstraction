#!/usr/bin/env python

# Python imports.
from collections import defaultdict
import numpy as np
import random as r

# Other imports.
import make_mdp
from simple_rl.mdp.MDPClass import MDP
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, FixedPolicyAgent
from simple_rl.utils.ValueIterationClass import ValueIteration

def compute_avg_mdp(mdp_distr):
    '''
    Args:
        mdp_distr (defaultdict)

    Returns:
        (MDP)
    '''

    # Get normal components.
    first_mdp = mdp_distr.keys()[0]
    init_state = first_mdp.get_init_state()
    actions = first_mdp.get_actions()
    gamma = first_mdp.get_gamma()
    transition_func = first_mdp.transition_func

    vi = ValueIteration(first_mdp, delta=0.001, max_iterations=1000)
    iters, value = vi.run_vi()
    states = vi.get_states()

    # Compute avg reward.
    avg_rew = defaultdict(float)
    for mdp in mdp_distr.keys():
        prob_of_reward = mdp_distr[mdp]
        for s in states:
            for a in actions:
                r = mdp.reward_func(s,a)

                avg_rew[(s,a)] += prob_of_reward * r


    def rew_dict_to_func(s,a):
        return avg_rew[s,a]

    avg_rew_func = rew_dict_to_func

    avg_mdp = MDP(actions, transition_func, avg_rew_func, init_state, gamma)

    return avg_mdp

def compute_optimal_stoch_policy(mdp_distr):
    '''
    Args:
        mdp_distr (defaultdict)

    Returns:
        (lambda)
    '''

    # Key: state
    # Val: dict
        # Key: action
        # Val: probability
    policy_dict = defaultdict(lambda : defaultdict(float))

    # Compute optimal policy for each MDP.
    for mdp in mdp_distr.keys():
        # Solve the MDP and get the optimal policy.
        vi = ValueIteration(mdp, delta=0.001, max_iterations=1000)
        iters, value = vi.run_vi()
        vi_policy = vi.policy
        states = vi.get_states()

        # Compute the probability each action is optimal in each state.        
        prob_of_mdp = mdp_distr[mdp]
        for s in states:
            a_star = vi_policy(s)
            policy_dict[s][a_star] += prob_of_mdp

    # Create the lambda.
    def policy_from_dict(state):
        action_id = np.random.multinomial(1, policy_dict[state].values()).tolist().index(1)
        action = policy_dict[state].keys()[action_id]

        return action

    opt_stochastic_policy = policy_from_dict

    return opt_stochastic_policy

def make_base_a_list_from_number(number, base_a):
    '''
    Args:
        number (int): Base ten.
        base_a (int): New base to convert to.

    Returns:
        (list): Contains @number converted to @base_a.
    '''
    
    # Make a single 32 bit word.
    result = [0 for bit in range(32)]

    for i in range(len(result)-1, -1, -1):
        quotient, remainder = divmod(number,base_a**i)
        result[len(result) - i - 1] = quotient
        number = remainder

    # Remove trailing zeros before the number.
    first_non_zero_index = next((index for index, element in enumerate(result) if element > 0), None)

    return result[first_non_zero_index:]

def make_all_fixed_policies(states, actions):
    '''
    Args:
        states (list)
        actions (list)

    Returns:
        (list): Contains all deterministic policies.
    '''
    all_policies = defaultdict(list)

    # Each policy is a length |S| list containing a number.
    # That number indicates which action to take in the index-th state.

    num_states = len(states)
    num_actions = len(actions)

    all_policies = [make_base_a_list_from_number(i,num_actions) for i in range(num_states**num_actions)]

    return all_policies

def make_policy_from_action_list(action_ls, actions, states):
    '''
    Args:
        action_ls (ls): Each element is a number from [0:|actions|-1],
            indicating which action to take in that state. Each index
            corresponds to the index-th state in @states.
        actions (list)
        states (list)

    Returns:
        (lambda)
    '''

    policy_dict = defaultdict(str)

    for i, s in enumerate(states):
        try:
            a = actions[action_str[i]]
        except:
            a = actions[0]
        policy_dict[s] = a

    # Create the lambda
    def policy_from_dict(state):
        return policy_dict[state]

    policy_func = policy_from_dict

    return policy_func

def get_all_fixed_policy_agents(mdp):    
    states = mdp.get_states()
    actions = mdp.get_actions()

    all_policies = make_all_fixed_policies(states, actions)
    fixed_agents = []
    for i, p in enumerate(all_policies):
        policy = make_policy_from_action_str(p, actions, states)

        next_agent = FixedPolicyAgent(policy, name="rand-fixed-policy-" + str(i))

        fixed_agents.append(next_agent)

    return fixed_agents

def main():

    # Setup multitask setting.
    mdp_class = "grid"
    mdp_distr = make_mdp.make_mdp_distr(mdp_class=mdp_class, num_mdps=10)
    actions = mdp_distr.keys()[0].actions
    gamma = mdp_distr.keys()[0].gamma

    # Compute average MDP.
    avg_mdp = compute_avg_mdp(mdp_distr)
    avg_mdp_vi = ValueIteration(avg_mdp, delta=0.001, max_iterations=1000)
    iters, value = avg_mdp_vi.run_vi()
    states = avg_mdp_vi.get_states()

    # Optimal stochastic policy.
    opt_stoch_policy = compute_optimal_stoch_policy(mdp_distr)

    # Agents.
    opt_stoch_policy = FixedPolicyAgent(opt_stoch_policy, name="$\pi_D^*$")
    vi_agent = FixedPolicyAgent(avg_mdp_vi.policy, name="$\pi_{avg}^*$")
    rand_agent = RandomAgent(actions, name="$\pi^u$")
    ql_agent = QLearnerAgent(actions)

    # Run task.
    total_reward_each_agent = run_agents_multi_task([vi_agent, opt_stoch_policy, rand_agent], mdp_distr, instances=200, steps=50)

    max_reward = 0
    best_agent = ""
    for agent_name in total_reward_each_agent.keys():
        if total_reward_each_agent[agent_name] > max_reward:
            best_agent = agent_name
            max_reward = total_reward_each_agent[agent_name]

    print best_agent, max_reward


if __name__ == "__main__":
    main()