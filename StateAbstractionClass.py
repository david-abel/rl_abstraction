# Python imports.
from collections import defaultdict

# Other imports.
from simple_rl.mdp.StateClass import State

class StateAbstraction(object):

	def __init__(self):
		self._phi = {} # key:state, val:int. (int represents an abstract state).
		
	def phi(self, state):
		
		# Setup phi.
		if len(self._phi.keys()) == 0:
			self._phi[state] = 1
		elif state not in self._phi.keys():
			self._phi[state] = max(self._phi.values()) + 1

		return State(self._phi[state])

	def make_cluster(self, list_of_ground_states):
		if len(list_of_ground_states) == 0:
			return
		
		abstract_value = 0
		if len(self._phi.values()) != 0:
			abstract_value = max(self._phi.values()) + 1

		for state in list_of_ground_states:
			self._phi[state] = abstract_value

	def get_num_abstr_states(self):
		return len(set(self._phi.values()))
