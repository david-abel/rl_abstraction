#!/usr/bin/env python

# Other imports.
from simple_rl.agents import RMaxAgent, DelayedQAgent
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.mdp import MDP
from simple_rl.tasks.chain.ChainStateClass import ChainState
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.abstraction.state_abs import indicator_funcs
from abstraction_experiments import get_sa

class BadChainMDP(MDP):

    ACTIONS = ["left", "right", "loop"]

    def __init__(self, gamma, kappa=0.001):
        MDP.__init__(self, BadChainMDP.ACTIONS, self._transition_func, self._reward_func, init_state=ChainState(1), gamma=gamma)
        self.num_states = 4
        self.kappa = kappa

    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)
            statePrime

        Returns
            (float)
        '''
        if state.is_terminal():
            return 0
        elif action == "right" and state.num + 1 == self.num_states:
            return 1 # RMax.
        elif action == "loop" and state.num < self.num_states:
            return self.kappa
        else:
            return 0

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            # Terminal, done.
            return state
        elif action == "right" and state.num + 1 == self.num_states:
            # Applied right in s2, move to terminal.
            terminal_state = ChainState(self.num_states)
            terminal_state.set_terminal(True)
            return terminal_state
        elif action == "right" and state.num < self.num_states - 1:
            # If in s0 or s1, move to s2.
            return ChainState(state.num + 1)
        elif action == "left" and state.num > 1:
            # If in s1, or s2, move left.
            return ChainState(state.num - 1)
        else:
            # Otherwise, stay in the same state.
            return state

    def __str__(self):
        return "Bad_chain"

def main():

    # Grab experiment params.
    mdp = BadChainMDP(gamma=0.95, kappa=0.001)
    actions = mdp.get_actions()

    # =======================
    # == Make Abstractions ==
    # =======================
    sa_q_eps = get_sa(mdp, indic_func=indicator_funcs._q_eps_approx_indicator, epsilon=0.1)

    # RMax Agents. 
    rmax_agent = RMaxAgent(actions)
    abstr_rmax_agent = AbstractionWrapper(RMaxAgent, state_abstr=sa_q_eps, agent_params={"actions":actions}, name_ext="-$\\phi_{Q_\\epsilon^*}$")

    # Delayed Q Agents.
    del_q_agent = DelayedQAgent(actions)
    abstr_del_q_agent = AbstractionWrapper(DelayedQAgent, state_abstr=sa_q_eps, agent_params={"actions":actions}, name_ext="-$\\phi_{Q_\\epsilon^*}$")

    run_agents_on_mdp([rmax_agent, abstr_rmax_agent, del_q_agent, abstr_del_q_agent], mdp, instances=50, steps=250, episodes=1)

if __name__ == "__main__":
    main()
