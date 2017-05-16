# Python imports.
import random as r
from collections import defaultdict
import itertools

# Non-standard imports.
from simple_rl.agents import RandomAgent, RMaxAgent, QLearnerAgent
from simple_rl.tasks import ChainMDP, GridWorldMDP, TaxiOOMDP, RandomMDP
from simple_rl.run_experiments import run_agents_multi_task, run_agents_on_mdp
from simple_rl.utils.ValueIterationClass import ValueIteration
from simple_rl.utils.visualize_mdp import visualize_mdp
from AbstractionWrapperClass import AbstractionWrapper
from StateAbstractionClass import StateAbstraction

def _q_eps_approx_indicator(state_x, state_y, vi, mdp, epsilon=0.0):
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

def make_sa(mdp, indicator_func):
    '''
    Args:
        mdp (MDP)
        indicator_func (S x S --> {0,1})
        epsilon (float)
    '''

    # Run VI
    vi = ValueIteration(mdp)
    vi.run_vi()

    print "Num Ground States:", len(vi.S)

    sa = StateAbstraction()
    clusters = defaultdict(list)

    # Find state pairs that satisfy the condition.
    for state_x in vi.get_states():
        clusters[state_x] = [state_x]
        for state_y in vi.get_states():
            if state_x != state_y and indicator_func(state_x, state_y, vi, mdp):
                clusters[state_x].append(state_y)
                clusters[state_y].append(state_x)

    # Build SA.
    for state in clusters.keys():
        new_cluster = clusters[state]
        sa.make_cluster(new_cluster)

        # Destroy old so we don't double up.
        for s in clusters[state]:
            clusters[s] = []

    return sa

def make_mdp_distr(mdp_class="grid", num_mdps=1):
    '''
    Args:
        mdp_class (str): one of {"grid", "random"}
        num_mdps (int)

    Returns:
        (dict): {key:probability, value:mdp}
    '''
    mdp_distr = {}
    mdp_prob = 1.0/num_mdps

    for i in range(num_mdps):
        mdp_distr[mdp_prob] = {"grid":GridWorldMDP(10,10, (1, 1), (10, r.randint(1,10))),
                                "random":RandomMDP(num_states=40, num_rand_trans=r.randint(1,10))}[mdp_class]

    return mdp_distr


def main():

    # Multi Task
    # mdp_distr = make_mdp_distr(mdp_class="grid")
    # actions = mdp_distr.values()[0].actions
    # gamma = mdp_distr.values()[0].gamma

    # Single MDP
    goal_locs = []
    height, width = 3, 20
    for element in itertools.product([height], range(1, width + 1)):
        goal_locs.append(element)

    mdp = GridWorldMDP(height=height, width=width, init_loc=(1, 1), goal_locs=goal_locs)
    actions = mdp.actions
    gamma = mdp.gamma

    # Make Q equivalence state abstraction.
    q_equiv_sa = make_sa(mdp, _q_eps_approx_indicator)

    # AGENTS
    random_agent = RandomAgent(actions)
    rand_abstr_agent = AbstractionWrapper(random_agent, q_equiv_sa)

    rmax_agent = RMaxAgent(actions, gamma=gamma, horizon=7, s_a_threshold=25)
    abstr_rmax_agent = AbstractionWrapper(rmax_agent, q_equiv_sa)

    qlearner_agent = QLearnerAgent(actions, gamma=gamma, explore="uniform")
    abstr_qlearner_agent = AbstractionWrapper(qlearner_agent, q_equiv_sa)

    # Run single MDP experiment.
    # run_agents_on_mdp([rmax_agent, abstr_rmax_agent], mdp, instances=100, episodes=1, steps=50)

    # Visualize uncompressed
    # visualize_mdp(mdp)

    # Visualize compressed
    visualize_mdp(mdp, q_equiv_sa.phi)

    # Run experiments.
    # run_agents_multi_task([random_agent, rand_abstr_agent], mdp_distr, instances=5, num_switches=10, steps=20)

if __name__ == "__main__":
    main()
