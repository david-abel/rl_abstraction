#!/usr/bin/env python

# Python imports.
import random
from collections import defaultdict
import os

# Other imports.
from simple_rl.agents import RandomAgent, RMaxAgent, QLearningAgent, DelayedQAgent, DoubleQAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_lifelong, run_agents_on_mdp
from simple_rl.tasks import FourRoomMDP, HanoiMDP
from simple_rl.planning import ValueIteration
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction
from simple_rl.abstraction.AbstractValueIterationClass import AbstractValueIteration
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from state_abs import indicator_funcs as ind_funcs
from abstraction_experiments import compute_pac_sa, get_sa, get_directed_option_sa_pair
from utils.StochasticSAPolicyClass import StochasticSAPolicy
from utils import make_mdp

def get_exact_vs_approx_agents(environment, incl_opt=True):
    '''
    Args:
        environment (simple_rl.MDPDistribution)
        incl_opt (bool)

    Returns:
        (list)
    '''

    actions = environment.get_actions()
    gamma = environment.get_gamma()

    exact_qds_test = get_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.0)
    approx_qds_test = get_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.05)
    
    ql_agent = QLearningAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    ql_exact_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":actions}, state_abstr=exact_qds_test, name_ext="-exact")
    ql_approx_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":actions}, state_abstr=approx_qds_test, name_ext="-approx")
    ql_agents = [ql_agent, ql_exact_agent, ql_approx_agent]

    dql_agent = DoubleQAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    dql_exact_agent = AbstractionWrapper(DoubleQAgent, agent_params={"actions":actions}, state_abstr=exact_qds_test, name_ext="-exact")
    dql_approx_agent = AbstractionWrapper(DoubleQAgent, agent_params={"actions":actions}, state_abstr=approx_qds_test, name_ext="-approx")
    dql_agents = [dql_agent, dql_exact_agent, dql_approx_agent]

    rm_agent = RMaxAgent(actions, gamma=gamma)
    rm_exact_agent = AbstractionWrapper(RMaxAgent, agent_params={"actions":actions}, state_abstr=exact_qds_test, name_ext="-exact")
    rm_approx_agent = AbstractionWrapper(RMaxAgent, agent_params={"actions":actions}, state_abstr=approx_qds_test, name_ext="-approx")
    rm_agents = [rm_agent, rm_exact_agent, rm_approx_agent]

    if incl_opt:
        vi = ValueIteration(environment)
        vi.run_vi()
        opt_agent = FixedPolicyAgent(vi.policy, name="$\pi^*$")

        sa_vi = AbstractValueIteration(environment, sample_rate=50, max_iterations=3000, delta=0.0001, state_abstr=approx_qds_test, action_abstr=ActionAbstraction(options=[], prim_actions=environment.get_actions()))
        sa_vi.run_vi()
        approx_opt_agent = FixedPolicyAgent(sa_vi.policy, name="$\pi_\phi^*$")

        dql_agents += [opt_agent, approx_opt_agent]

    return ql_agents


def get_sa_experiment_agents(environment, AgentClass, pac=False):
    '''
    Args:
        environment (simple_rl.MDPDistribution)
        AgentClass (Class)

    Returns:
        (list)
    '''
    actions = environment.get_actions()
    gamma = environment.get_gamma()

    if pac:
        # PAC State Abstractions.
        sa_qds_test = compute_pac_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.2)
        sa_qs_test = compute_pac_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.2)
        sa_qs_exact_test = compute_pac_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.0)

    else:
        # Compute state abstractions.
        sa_qds_test = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.1)
        sa_qs_test = get_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.1)
        sa_qs_exact_test = get_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.0)

    # Make Agents.
    agent = AgentClass(actions, gamma=gamma)
    params = {"actions":actions} if AgentClass is not RMaxAgent else {"actions":actions, "s_a_threshold":2, "horizon":5}
    sa_qds_agent = AbstractionWrapper(AgentClass, agent_params=params, state_abstr=sa_qds_test, name_ext="$-\phi_{Q_d^*}$")
    sa_qs_agent = AbstractionWrapper(AgentClass, agent_params=params, state_abstr=sa_qs_test, name_ext="$-\phi_{Q_\epsilon^*}$")
    sa_qs_exact_agent = AbstractionWrapper(AgentClass, agent_params=params, state_abstr=sa_qs_exact_test, name_ext="-$\phi_{Q^*}$")
    
    agents = [agent, sa_qds_agent, sa_qs_agent, sa_qs_exact_agent]

    # if isinstance(environment.sample(), FourRoomMDP) or isinstance(environment.sample(), ColorMDP):
    #     # If it's a fourroom add the handcoded one.
    #     sa_hand_test = get_sa(environment, indic_func=ind_funcs._four_rooms)
    #     sa_hand_agent = AbstractionWrapper(AgentClass, agent_params=params, state_abstr=sa_hand_test, name_ext="$-\phi_h$")
    #     agents += [sa_hand_agent]

    return agents

def get_combo_experiment_agents(environment):
    '''
    Args:
        environment (simple_rl.MDPDistribution)

    Returns:
        (list)
    '''
    actions = environment.get_actions()
    gamma = environment.get_gamma()

    sa, aa = get_directed_option_sa_pair(environment, indic_func=ind_funcs._q_disc_approx_indicator, max_options=100)
    sa_qds_test = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.05)
    sa_qs_test = get_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.1)

    # QLearner.
    ql_agent = QLearningAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    rmax_agent = RMaxAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)

    # Combos.
    ql_sa_qds_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":actions}, state_abstr=sa_qds_test, name_ext="$\phi_{Q_d^*}$")
    ql_sa_qs_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":actions}, state_abstr=sa_qs_test, name_ext="$\phi_{Q_\epsilon^*}$")

    # sa_agent = AbstractionWrapper(QLearningAgent, actions, str(environment), state_abstr=sa, name_ext="sa")
    aa_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":actions}, action_abstr=aa, name_ext="aa")
    sa_aa_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":actions}, state_abstr=sa, action_abstr=aa, name_ext="$\phi_{Q_d^*}+aa$")

    agents = [ql_agent, ql_sa_qds_agent, ql_sa_qs_agent, aa_agent, sa_aa_agent]

    return agents

def get_optimal_policies(environment):
    '''
    Args:
        environment (simple_rl.MDPDistribution)

    Returns:
        (list)
    '''

    # Make State Abstraction
    approx_qds_test = get_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.05)

    # True Optimal
    true_opt_vi = ValueIteration(environment)
    true_opt_vi.run_vi()
    opt_agent = FixedPolicyAgent(true_opt_vi.policy, "$\pi^*$")

    # Optimal Abstraction
    opt_det_vi = AbstractValueIteration(environment, state_abstr=approx_qds_test, sample_rate=30)
    opt_det_vi.run_vi()
    opt_det_agent = FixedPolicyAgent(opt_det_vi.policy, name="$\pi_{\phi}^*$")

    stoch_policy_obj = StochasticSAPolicy(approx_qds_test, environment)
    stoch_agent = FixedPolicyAgent(stoch_policy_obj.policy, "$\pi(a \mid s_\phi )$")

    ql_agents = [opt_agent, stoch_agent, opt_det_agent]

    return ql_agents

def parse_args():
    '''
    Summary:
        Parse all arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", type = str, default = "four_room", nargs = '?', help = "Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-samples", type = int, default = 50, nargs = '?', help = "Number of samples from the MDP Distribution.")
    parser.add_argument("-steps", type = int, default = 100, nargs = '?', help = "Number of steps for the experiment.")
    parser.add_argument("-episodes", type = int, default = 1, nargs = '?', help = "Number of episodes for the experiment.")
    parser.add_argument("-grid_dim", type = int, default = 11, nargs = '?', help = "Dimensions of the grid world.")
    parser.add_argument("-agent", type = str, default='ql', nargs = '?', help = "Specify agent class (one of {'ql', 'rmax'})..")
    args = parser.parse_args()

    return args.task, args.samples, args.episodes, args.steps, args.grid_dim, args.agent

def get_params(set_manually=False):
    '''
    Args:
        set_manually (bool)
    
    Returns:
        (tuple)
    '''

    if set_manually:
        # Grab experiment params.
        mdp_class = "four_room"
        task_samples = 5
        episodes = 100
        steps = 250
        grid_dim = 9
        AgentClass = QLearningAgent
    else:
        # Grab experiment params.
        mdp_class, task_samples, episodes, steps, grid_dim, agent_class_str = parse_args()

        if agent_class_str == "dql":
            AgentClass = DelayedQAgent
        else:
            AgentClass = QLearningAgent

    return mdp_class, task_samples, episodes, steps, grid_dim, AgentClass

def main():

    # Set Params.
    mdp_class, task_samples, episodes, steps, grid_dim, AgentClass = get_params(set_manually=True)
    experiment_type = "sa"
    lifelong = False
    resample_at_terminal = False
    reset_at_terminal = False
    gamma = 0.95

    # ======================
    # == Make Environment ==
    # ======================
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=grid_dim) if lifelong else make_mdp.make_mdp(mdp_class=mdp_class, grid_dim=grid_dim)
    environment.set_gamma(gamma)

    # =================
    # == Make Agents ==
    # =================
    agents = []
    if experiment_type == "sa":
        # SA experiment.
        agents = get_sa_experiment_agents(environment, AgentClass)
    elif experiment_type == "combo":
        # AA experiment.
        agents = get_combo_experiment_agents(environment)
    elif experiment_type == "exact_v_approx":
        agents = get_exact_vs_approx_agents(environment, incl_opt=(not multi_task))
    elif experiment_type == "opt":
        agents = get_optimal_policies(environment)
    else:
        print "Experiment Error: experiment type unknown (" + experiment_type + "). Must be one of {sa, combo, exact_v_approx}."
        quit()

    # Run!
    if lifelong:
        run_agents_lifelong(agents, environment, samples=task_samples, steps=steps, episodes=episodes, reset_at_terminal=reset_at_terminal, resample_at_terminal=resample_at_terminal, cumulative_plot=True, clear_old_results=True)
    else:
        run_agents_on_mdp(agents, environment, instances=task_samples, steps=steps, episodes=episodes, reset_at_terminal=reset_at_terminal, track_disc_reward=False)

if __name__ == "__main__":
	main()
