# Python imports.
from collections import defaultdict
import numpy as np

# Other imports.
from HierarchyAgentClass import HierarchyAgent

class DynamicHierarchyAgent(HierarchyAgent):
    
    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        if not self.action_abstr_stack.is_next_step_continuing_option(ground_state):
            # We're in a "decision" state, so change levels.
            new_level = self._compute_max_v_hat_level(ground_state)
            self.set_level(new_level)

        action = HierarchyAgent.act(self, ground_state, reward)

        return action

    def _compute_max_v_hat_level(self, ground_state):
        '''
        Args:
            ground_state (simple_rl.mdp.State)

        Returns:
            (int): The level with the highest value estimate.
        '''
        max_q = float("-inf")
        best_lvl = np.random.choice(xrange(self.get_num_levels() + 1))
        for lvl in xrange(self.get_num_levels() + 1):
            abstr_state = self.state_abstr_stack.phi(ground_state, lvl)
            v_hat = self.agent.get_max_q_value(abstr_state)

            if v_hat - (lvl * 0.05) > max_q:
                best_lvl = lvl
                max_q = v_hat

        return best_lvl

