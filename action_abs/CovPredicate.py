class CovPredicate(object):
	def __init__(self, y, policy):
		self.y = y
		self.policy = policy

	def is_true(self, x):
		return (x in self.policy.keys()) == self.y
