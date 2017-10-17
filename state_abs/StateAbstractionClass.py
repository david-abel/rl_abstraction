# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP

class StateAbstraction(object):

    def __init__(self, phi, state_class=State, track_act_opt_pr=False):
        '''
        Args:
            phi (dict)
            state_class (Class)
            track_act_opt_pr (bool): If true, tracks the probability with which
            each action is optimal in each ground state w.r.t. the distribution.
        '''
        self._phi = phi # key:state, val:int. (int represents an abstract state).
        self.state_class = state_class
        self.track_act_opt_pr = track_act_opt_pr
        if self.track_act_opt_pr:
            self.phi_act_optimality_dict = defaultdict(lambda:defaultdict(float))
                # Key: Ground State
                # Val: Dict
                    # Key: Action.
                    # Val: Probability it's optimal.
        else:
            self.phi_act_optimality_dict = defaultdict(set)

    def get_act_opt_dict(self):
        return self.phi_act_optimality_dict

    def set_act_opt_dict(self, new_dict):
        if self.track_act_opt_pr and (len(new_dict.keys()) == 0 or isinstance(new_dict.keys()[0], dict)):
            print "State Abstraction Error: Tried setting optimality dict of incorrect type. Must be K:state, V:dict (K: action, V: probability)."
            quit()
        self.phi_act_optimality_dict = new_dict

    def set_actions_state_opt_dict(self, ground_state, action_set, prob_of_mdp=1.0):
        '''
        Args:
            ground_state (State)
            action (str)

        Summary:
            Tracks optimal actions in each abstract state.
        '''
        if self.track_act_opt_pr:
            for a in action_set:
                self.phi_act_optimality_dict[ground_state][a] = prob_of_mdp
        else:
            self.phi_act_optimality_dict[ground_state] = action_set

    def set_phi(self, new_phi):
        self._phi = new_phi

    def phi(self, state):
        '''
        Args:
            state (State)

        Returns:
            state (State)
        '''
        # Setup phi for new states.
        if state not in self._phi.keys():
            if len(self._phi.values()) > 0:
                self._phi[state] = max(self._phi.values()) + 1
            else:
                self._phi[state] = 1

        abstr_state = self.state_class(self._phi[state])

        abstr_state.set_terminal(state.is_terminal())

        return abstr_state

    def make_cluster(self, list_of_ground_states):
        if len(list_of_ground_states) == 0:
            return

        abstract_value = 0
        if len(self._phi.values()) != 0:
            abstract_value = max(self._phi.values()) + 1

        for state in list_of_ground_states:
            self._phi[state] = abstract_value

    def get_ground_states_in_abs_state(self, abs_state):
        '''
        Args:
            abs_state (State)

        Returns:
            (list): Contains all ground states in the cluster.
        '''
        return [s_g for s_g in self.get_ground_states() if self.phi(s_g) == abs_state]
    
    def get_lower_states_in_abs_state(self, abs_state):
        '''
        Args:
            abs_state (State)

        Returns:
            (list): Contains all ground states in the cluster.

        Notes:
            Here to simplify the state abstraction stack subclass.
        '''
        return self.get_ground_states_in_abs_state(abs_state)


    def get_abs_states(self):
        # For each ground state, get its abstract state.
        return set([self.phi(val) for val in set(self._phi.keys())])

    def get_abs_cluster_num(self, abs_state):
        # FIX: Specific to one abstract state class.
        return list(set(self._phi.values())).index(abs_state.data)

    def get_ground_states(self):
        return self._phi.keys()

    def get_lower_states(self):
        return self.get_ground_states()

    def get_num_abstr_states(self):
        return len(set(self._phi.values()))

    def get_num_ground_states(self):
        return len(set(self._phi.keys()))

    def reset(self):
        self._phi = {}

    def __add__(self, other_abs):
        '''
        Args:
            other_abs
        '''
        merged_state_abs = {}

        # Move the phi into a cluster dictionary.
        cluster_dict = defaultdict(set)
        for k, v in self._phi.iteritems():
            # Cluster dict: v is abstract, key is ground.
            cluster_dict[v].add(k)

        # Move the phi into a cluster dictionary.
        other_cluster_dict = defaultdict(set)
        for k, v in other_abs._phi.iteritems():
            other_cluster_dict[v].add(k)

        for ground_state in self._phi.keys():
            # Get the two clusters (ints that define abstr states) associated with a state.
            states_cluster = self._phi[ground_state]
            if ground_state in other_abs._phi.keys():
                # Only add if it's in both clusters.
                states_other_cluster = other_abs._phi[ground_state]
            else:
                continue

            for s_g in cluster_dict[states_cluster]:
                if s_g in other_cluster_dict[states_other_cluster]:
                    # Grab every ground state that's in both clusters and put them in the new cluster.
                    merged_state_abs[s_g] = states_cluster
        
        new_sa = StateAbstraction(phi=merged_state_abs, track_act_opt_pr=self.track_act_opt_pr)

        # Build the new action optimality dictionary.
        if self.track_act_opt_pr:
            # Grab the two action optimality dictionaries.
            opt_dict = self.get_act_opt_dict()
            other_opt_dict = other_abs.get_act_opt_dict()
            
            # If we're tracking the action's probability.
            new_dict = defaultdict(lambda:defaultdict(float))
            for s_g in self.get_ground_states():
                for a_g in opt_dict[s_g].keys() + other_opt_dict[s_g].keys():
                    new_dict[s_g][a_g] = opt_dict[s_g][a_g] + other_opt_dict[s_g][a_g]
            new_sa.set_act_opt_dict(new_dict)

        return new_sa

