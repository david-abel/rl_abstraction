from PolicyClass import Policy
from collections import defaultdict

class PolicyFromDict(Policy):

	def __init__(self, policy_dict={}):
		self.policy_dict = policy_dict

	def get_action(self, state):
		return self.policy_dict[state]

def make_dict_from_lambda(policy_func, state_list):
	policy_dict = defaultdict(str)
	for s in state_list:
		policy_dict[s] = policy_func(s)

	return policy_dict