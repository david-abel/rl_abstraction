from collections import defaultdict
import random as r
from simple_rl.mdp.StateClass import State


class Option(object):

	def __init__(self, init_predicate, term_predicate, policy, name="o"):
		'''
		Args:
			init_func (S --> {0,1})
			init_func (S --> {0,1})
			policy (S --> A)
		'''
		self.init_predicate = init_predicate
		self.term_predicate = term_predicate
		self.term_flag = False
		self.name = name

		# Special types.
		# if type(term_predicate) is list:
		# 	self.term_list = term_func
		# 	self.term_predicate = self.term_func_from_list

		if type(policy) is defaultdict:
			self.policy_dict = dict(policy)
			self.policy = self.policy_from_dict
		else:
			self.policy = policy

	def is_init_true(self, ground_state):
		return self.init_predicate.is_true(ground_state)

	def is_term_true(self, ground_state):
		return self.term_predicate.is_true(ground_state) or self.term_flag

	def act(self, ground_state):
		return self.policy(ground_state)

	def set_policy(self, policy):
		self.policy = policy

	def set_name(self, new_name):
		self.name = new_name

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


	def __str__(self):
		return "option." + str(self.name)