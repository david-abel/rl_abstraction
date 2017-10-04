# Python imports.
from collections import defaultdict

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
        self.known = defaultdict(bool)
        self.seen_transitions = defaultdict(lambda:defaultdict(bool))
        # Could fill with all possible s-s' pairs in the clusters.

        HierarchyAgent.__init__(self, SubAgentClass, sa_stack, aa_stack, cur_level, name_ext=name_ext)
    
    def act(self, ground_state, reward):
        '''
        Args:
            ground_state (State)
            reward (float)

        Return:
            (str)
        '''
        self.prev_state = None
        self.prev_action = None

        if self.prev_state != None:
            if self.sa_stack.phi(self.prev_state, self.cur_level) != self.sa_stack.phi(ground_state, self.cur_level):
                # We've changed clusters. Go up.
                self.incr_level()
            else:
                # Same cluster, track experience.
                abstr_state = self.sa_stack.phi(ground_state, self.cur_level)
                if (self.prev_state, ground_state) in self.seen_transitions[abstr_state]:
                    # If we've seen it...
                    self.known[abstr_state] = True
                    self.incr_level()
                else:
                    self.decr_level()



        action = HierarchyAgent.act(self, ground_state, reward)

        return action
