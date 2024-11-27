from .. import *

class MinifyNamesOptimizer(ASTIdentifierNode):
	level = 5

	identifier: str

	def optimize(self, ns):
		cached = self._minify.is_cached(self.identifier)
		identifier = self._minify(self.identifier)
		if (not cached): ns.rename(self.identifier, identifier)
		self.identifier = identifier
