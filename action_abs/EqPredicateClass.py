class EqPredicate(object):

	def __init__(self, y, func):
		self.y = y
		self.func = func

	def is_true(self, x):
		return self.func(x) == self.y


