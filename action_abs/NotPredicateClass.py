from PredicateClass import Predicate

class NotPredicate(Predicate):

	def __init__(self, predicate):
		self.predicate = predicate

	def is_true(self, x):
		return not self.predicate.is_true(x)

