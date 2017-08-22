class Policy(object):

	def __init__(self, action=""):
		self.action = action

	def get_action(self, state):
		return self.action