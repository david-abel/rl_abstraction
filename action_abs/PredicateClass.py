class Predicate(object):

	def __init__(self, func, params={}):
		self.func = func

	def is_true(self, x):
		return self.func(x)

