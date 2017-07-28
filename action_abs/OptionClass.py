from collections import defaultdict
import random as r
from simple_rl.mdp.StateClass import State


class Option(object):

	def __init__(self, init_func, term_func, policy):
		'''
		Args:
			init_func (S --> {0,1})
			init_func (S --> {0,1})
			policy (S --> A)
		'''
		self.init_func = init_func
		self.term_func = term_func
		self.term_flag = False

		# Special types.
		if type(term_func) is list:
			self.term_list = term_func
			self.term_func = self.term_func_from_list

		if type(policy) is defaultdict:
			self.policy_dict = dict(policy)
			self.policy = self.policy_from_dict
		else:
			self.policy = policy

	def is_init_true(self, ground_state):
		return self.init_func(ground_state)

	def is_term_true(self, ground_state):
		return self.term_func(ground_state) or self.term_flag

	def act(self, ground_state):
		return self.policy(ground_state)

	def set_policy(self, policy):
		self.policy = policy

	def act_until_terminal(self, init_state, transition_func):
		'''
		Summary:
			Executes the option until termination.
		'''
		if self.is_init_true(init_state):
			cur_state = transition_func(init_state, self.act(init_state))
			while not self.is_term_true(cur_state):
				cur_state = transition_func(cur_state, self.act(cur_state))

		return cur_state

	def policy_from_dict(self, state):
		if state not in self.policy_dict.keys():
			self.term_flag = True
			return r.choice(list(set(self.policy_dict.values())))
		else:
			self.term_flag = False
			return self.policy_dict[state]

	def term_func_from_list(self, state):
		return state in self.term_list
