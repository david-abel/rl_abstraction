#!/usr/bin/env python

# Python imports.
import random
from collections import defaultdict
import os

# Other imports.
from simple_rl.utils import make_mdp
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent, DoubleQAgent, FixedPolicyAgent
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.tasks import FourRoomMDP, HanoiMDP
from simple_rl.planning import ValueIteration
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction
from simple_rl.abstraction.AbstractValueIterationClass import AbstractValueIteration
from AbstractionWrapperClass import AbstractionWrapper
from state_abs import indicator_funcs as ind_funcs
from abstraction_experiments import get_sa, get_directed_option_sa_pair
from StochasticSAPolicyClass import StochasticSAPolicy

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
    
    ql_agent = QLearnerAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    ql_exact_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=exact_qds_test, name_ext="-exact")
    ql_approx_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=approx_qds_test, name_ext="-approx")
    ql_agents = [ql_agent, ql_exact_agent, ql_approx_agent]

    dql_agent = DoubleQAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    dql_exact_agent = AbstractionWrapper(DoubleQAgent, actions, str(environment), state_abstr=exact_qds_test, name_ext="-exact")
    dql_approx_agent = AbstractionWrapper(DoubleQAgent, actions, str(environment), state_abstr=approx_qds_test, name_ext="-approx")
    dql_agents = [dql_agent, dql_exact_agent, dql_approx_agent]

    rm_agent = RMaxAgent(actions, gamma=gamma)
    rm_exact_agent = AbstractionWrapper(RMaxAgent, actions, str(environment), state_abstr=exact_qds_test, name_ext="-exact")
    rm_approx_agent = AbstractionWrapper(RMaxAgent, actions, str(environment), state_abstr=approx_qds_test, name_ext="-approx")
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


def get_sa_experiment_agents(environment):
    '''
    Args:
        environment (simple_rl.MDPDistribution)
        gamma (float)

    Returns:
        (list)
    '''

    actions = environment.get_actions()
    gamma = environment.get_gamma()

    # State Abstractions.
    sa_qds_test = get_sa(environment, indic_func=ind_funcs._q_disc_approx_indicator, epsilon=0.05)
    sa_qs_test = get_sa(environment, indic_func=ind_funcs._q_eps_approx_indicator, epsilon=0.05)

    # QLearners.
    ql_agent = QLearnerAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    ql_sa_qds_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_qds_test, name_ext="$\phi_{Q_d^*}$")
    ql_sa_qs_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_qs_test, name_ext="$\phi_{Q_\epsilon^*}$")
    
    # R-Max.
    # rm_agent = RMaxAgent(actions, gamma=gamma)
    # rm_sa_qds_agent = AbstractionWrapper(RMaxAgent, actions, str(environment), state_abstr=sa_qds_test, name_ext="$\phi_{Q_d^*}$")
    # rm_sa_qs_agent = AbstractionWrapper(RMaxAgent, actions, str(environment), state_abstr=sa_qs_test, name_ext="$\phi_{Q_\epsilon^*}$")

    # agents = [rm_agent, rm_sa_qds_agent, ql_sa_qs_agent]
    agents = [ql_agent, ql_sa_qds_agent, ql_sa_qs_agent]

    if isinstance(environment.sample(), FourRoomMDP):
        # If it's a fourroom add the handcoded one.
        sa_hand_test = get_sa(environment, indic_func=ind_funcs._four_rooms)
        ql_sa_hand_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_hand_test, name_ext="$\phi_h$")
        # rm_sa_hand_agent = AbstractionWrapper(RMaxAgent, actions, str(environment), state_abstr=sa_hand_test, name_ext="$\phi_h$")
        agents += [ql_sa_hand_agent]

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
    ql_agent = QLearnerAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)
    rmax_agent = RMaxAgent(actions, gamma=gamma, epsilon=0.1, alpha=0.05)

    # Combos.
    ql_sa_qds_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_qds_test, name_ext="$\phi_{Q_d^*}$")
    ql_sa_qs_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa_qs_test, name_ext="$\phi_{Q_\epsilon^*}$")

    # sa_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa, name_ext="sa")
    aa_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), action_abstr=aa, name_ext="aa")
    sa_aa_agent = AbstractionWrapper(QLearnerAgent, actions, str(environment), state_abstr=sa, action_abstr=aa, name_ext="$\phi_{Q_d^*}+aa$")

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

def main():

    # Grab experiment params.
    mdp_class = "hall"
    task_samples = 3000
    episodes = 1
    steps = 30 # 250 for four room, 30 for hall
    grid_dim = 10
    gamma = 0.95
    experiment_type = "sa" # One of {"sa", "combo", "exact_v_approx"}.
    multi_task = True
    resample_at_terminal = False
    reset_at_terminal = multi_task and not resample_at_terminal # True

    # ======================
    # == Make Environment ==
    # ======================
    environment = make_mdp.make_mdp_distr(mdp_class=mdp_class, grid_dim=grid_dim) if multi_task else make_mdp.make_mdp(mdp_class=mdp_class, grid_dim=grid_dim)
    environment.set_gamma(gamma)

    # =================
    # == Make Agents ==
    # =================
    agents = []
    if experiment_type == "sa":
        # SA experiment.
        agents = get_sa_experiment_agents(environment)
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
    if multi_task:
        run_agents_multi_task(agents, environment, task_samples=task_samples, steps=steps, episodes=episodes, reset_at_terminal=reset_at_terminal, resample_at_terminal=resample_at_terminal, verbose=False)
    else:
        run_agents_on_mdp(agents, environment, instances=task_samples, steps=steps, episodes=episodes, reset_at_terminal=reset_at_terminal, is_rec_disc_reward=True)

if __name__ == "__main__":
	main()
