class ContainsPredicate(object):

	def __init__(self, list_of_items):
		self.list_of_items = list_of_items

	def is_true(self, x):
		return x in self.list_of_items

