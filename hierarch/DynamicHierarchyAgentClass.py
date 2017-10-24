# Python imports.
from collections import defaultdict
import numpy as np

# Other imports.
from HierarchyAgentClass import HierarchyAgent

class DynamicHierarchyAgent(HierarchyAgent):
    
    def __init__(self, SubAgentClass, sa_stack, aa_stack, cur_level=0, name_ext=""):
        '''
        Args:
            sa_stack (StateAbstractionStack)
            aa_stack (ActionAbstractionStack)
            cur_level (int): Must be in [0:len(state_abstr_stack)]
        '''
        HierarchyAgent.__init__(self, SubAgentClass, sa_stack, aa_stack, cur_level=0, name_ext="")
        self.num_switches = 0
        self.num_actions_since_open = 0
        self.actions_until_open = 5
    
    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''


        if self.num_actions_since_open > self.actions_until_open and not self.action_abstr_stack.is_next_step_continuing_option(ground_state):
            # We're in a "decision" state, so change levels.
            new_level = self._compute_max_v_hat_level(ground_state)
            if self.cur_level != new_level:
                self.num_switches += 1
            self.set_level(new_level)
            self.num_actions_since_open = 0

        action = HierarchyAgent.act(self, ground_state, reward)

        self.num_actions_since_open += 1

        return action

    def _compute_max_v_hat_level(self, ground_state):
        '''
        Args:
            ground_state (simple_rl.mdp.State)

        Returns:
            (int): The level with the highest value estimate.
        '''

        if self.cur_level == 1:
            return 0
        else:
            max_q = float("-inf")
            best_lvl = 0
            for lvl in xrange(self.get_num_levels() + 1):
                abstr_state = self.state_abstr_stack.phi(ground_state, lvl)
                v_hat = self.agent.get_max_q_value(abstr_state)
                # print lvl, v_hat
                change_cost = 0.01 * int(lvl != self.cur_level)
                if v_hat - change_cost > max_q:
                    best_lvl = lvl
                    max_q = v_hat
            return best_lvl

    def reset(self):
        print "num switches this instance:", self.num_switches
        self.num_switches = 0
        HierarchyAgent.reset(self)
