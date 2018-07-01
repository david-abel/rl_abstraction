# Python imports
import numpy as np
from collections import defaultdict

# Other imports.
from simple_rl.planning import ValueIteration

class StochasticSAPolicy(object):

    def __init__(self, state_abstr, mdp):
        self.state_abstr = state_abstr
        self.mdp = mdp
        self.vi = ValueIteration(mdp)
        self.vi.run_vi()

    def policy(self, state):
        '''
        Args:
            (simple_rl.State)

        Returns:
            (str): An action

        Summary:
            Chooses an action among the optimal actions in the cluster. That is, roughly:

                \pi(a \mid s_a) \sim Pr_{s_g \in s_a} (a = a^*(s_a))
        '''

        abstr_state = self.state_abstr.phi(state)
        ground_states = self.state_abstr.get_ground_states_in_abs_state(abstr_state)

        action_distr = defaultdict(float)
        for s in ground_states:
            a = self.vi.policy(s)
            action_distr[a] += 1.0 / len(ground_states)

        sampled_distr = np.random.multinomial(1, action_distr.values()).tolist()
        indices = [i for i, x in enumerate(sampled_distr) if x > 0]

        return action_distr.keys()[indices[0]]
