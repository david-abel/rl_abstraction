'''
HRMaxAgentClass.py: Class for an RMaxAgent that uses a hierarchy controller
'''

# Python imports.
import random
from collections import defaultdict

# Local classes.
from simple_rl.agents import RMaxAgent

class HRMaxAgent(RMaxAgent):
    '''
    Implementation for an R-Max Agent that uses a hierarchy
    during its planning phase.
    '''

    def __init__(self, actions, sa_stack, aa_stack, level=0, gamma=0.99, horizon=4, s_a_threshold=1):
        self.sa_stack = sa_stack
        self.aa_stack = aa_stack
        self.level = level
        RMaxAgent.__init__(self, actions=actions, gamma=gamma, horizon=horizon, s_a_threshold=s_a_threshold)
        self.name = "hrmax-h" + str(horizon)

    def act(self, ground_state, reward):
        if not self.aa_stack.is_next_step_continuing_option(ground_state):
            # If we're in a decision state, set the level, update, and pick a new action.
            self._set_level(ground_state)

            # Grab the landing state in the abstract.
            cur_abstr_state = self.sa_stack.phi(ground_state, self.level)

            # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
            self.update(self.prev_state, self.prev_action, reward, cur_abstr_state)
    
            # Update actions.
            self.actions = self.aa_stack.get_actions(self.level)
    
            # Compute best action.
            action = self.get_max_q_action(ground_state)

            self.aa_stack.set_option_executing(action)

            # Update pointers.
            self.prev_action = action
            self.prev_state = cur_abstr_state
        else:
            # In the middle of computing an option.
            action = self.aa_stack.get_next_ground_action(ground_state)

        return action

    def _compute_max_qval_action_pair(self, ground_state, horizon=None, bootstrap=False):
        '''
        Args:
            ground_state (State)
            horizon (int): Indicates the level of recursion depth for computing Q.

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # If this is the first call, use the default horizon.
        horizon = self.horizon if horizon is None else horizon

        if horizon <= 0:
            r_max = float("-inf")
            best_a = self.actions[0]
            for a in self.actions:
                r = self._get_reward(ground_state, a)
                if r > r_max:
                    best_a = a
                    r_max = r
            return r, a

        # Update level and apply phi, omega.
        self._set_level(ground_state, horizon=horizon-1)

        self.actions = self.aa_stack.get_actions(level=self.level)
        decision_state = self.sa_stack.phi(ground_state, level=self.level)

        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = self.get_q_value(decision_state, best_action, horizon)

        # Find best action (action w/ current max predicted Q value)
        for action in self.actions:
            q_s_a = self.get_q_value(decision_state, action, horizon)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action

    def _compute_exp_future_return(self, ground_state, action, horizon=None):
        '''
        Args:
            ground_state (State)
            action (str)
            horizon (int): Recursion depth to compxute Q

        Return:
            (float): Discounted expected future return from applying @action in @state.
        '''
        # If this is the first call, use the default horizon.
        horizon = self.horizon if horizon is None else horizon

        self._set_level(ground_state, horizon=horizon-1)

        # Compute abstracted state.
        abstr_state = self.sa_stack.phi(ground_state, self.level)

        next_state_dict = self.transitions[abstr_state][action]

        denominator = float(sum(next_state_dict.values()))
        state_weights = defaultdict(float)
        for next_state in next_state_dict.keys():
            count = next_state_dict[next_state]
            state_weights[next_state] = (count / denominator)

        weighted_future_returns = [self.get_max_q_value(next_state, horizon-1) * state_weights[next_state] for next_state in next_state_dict.keys()]

        return sum(weighted_future_returns)

    def _set_level(self, ground_state, horizon=None):
        # If this is the first call, use the default horizon.
        horizon = self.horizon if horizon is None else horizon

        max_q = float("-inf")
        best_lvl = 0
        for lvl in xrange(self.sa_stack.get_num_levels() + 1):
            abstr_state = self.sa_stack.phi(ground_state, lvl)
            v_hat = self.get_max_q_value(abstr_state, horizon=horizon)
            if v_hat - (lvl * 0.001) > max_q:
                best_lvl = lvl
                max_q = v_hat

        self.level = best_lvl
