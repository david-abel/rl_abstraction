# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP

class StateAbstraction(object):

    def __init__(self, phi={}, state_class=State):
        self._phi = phi # key:state, val:int. (int represents an abstract state).
        self.state_class = state_class

    def phi(self, state):
        '''
        Args:
            state (State)

        Returns:
            state (State)
        '''
        # Setup phi for new states.
        if state not in self._phi.keys():
            self._phi[state] = max(self._phi.values()) + 1

        return self.state_class(self._phi[state])

    def make_cluster(self, list_of_ground_states):
        if len(list_of_ground_states) == 0:
            return

        abstract_value = 0
        if len(self._phi.values()) != 0:
            abstract_value = max(self._phi.values()) + 1

        for state in list_of_ground_states:
            self._phi[state] = abstract_value

    def make_abstr_mdp(self, mdp):
        '''
        Args:
            mdp (MDP)

        Returns:
            mdp (MDP): The abstracted MDP via self.phi.
        '''
        abstr_actions = mdp.get_actions()
        abstr_init_state = self.phi(mdp.get_init_state())
        ground_t, ground_r = mdp.get_transition_func(), mdp.get_reward_func()
        abstr_trans_func = lambda s, a: self.phi(ground_t(s, a))
        abstr_reward_func = lambda s, a: ground_r(s, a)
        abstr_gamma = mdp.get_gamma()

        abstr_mdp = MDP(actions=mdp.get_actions(),
                        transition_func=abstr_trans_func,
                        reward_func=abstr_reward_func,
                        init_state=abstr_init_state,
                        gamma=abstr_gamma)

        return abstr_mdp

    def get_ground_states_in_abs_state(self, abs_state):
        '''
        Args:
            abs_state (State)

        Returns:
            (list): Contains all ground states in the cluster.
        '''
        return [s_g for s_g in self.get_ground_states() if self.phi(s_g) == abs_state]

    def get_abs_states(self):
        # For each ground state, get its abstract state.
        return set([self.phi(val) for val in set(self._phi.keys())])

    def get_abs_cluster_num(self, abs_state):
        # FIX: Specific to one abstract state class.
        return list(set(self._phi.values())).index(abs_state.data)

    def get_ground_states(self):
        return self._phi.keys()

    def get_num_abstr_states(self):
        return len(set(self._phi.values()))

    def get_num_ground_states(self):
        return len(set(self._phi.keys()))

    def reset(self):
        self._phi = {}

    def __add__(self, other_abs):
        '''
        FIX
        '''
        merged_state_abs = {}

        # Move the phi into a cluster dictionary.
        cluster_dict = defaultdict(list)
        for k, v in self._phi.iteritems():
            cluster_dict[v].append(k)

        # Move the phi into a cluster dictionary.
        other_cluster_dict = defaultdict(list)
        for k, v in other_abs._phi.iteritems():
            other_cluster_dict[v].append(k)

        for state in self._phi.keys():

            # Get the two clusters associate with a state.
            states_cluster = self._phi[state]
            states_other_cluster = other_abs._phi[state]

            for s in cluster_dict[states_cluster]:
                if s in other_cluster_dict[states_other_cluster]:
                    # Every state that's in both clusters, merge.
                    merged_state_abs[s] = states_cluster

        return StateAbstraction(phi=merged_state_abs)

