from .. import *

class ConstantFoldingOptimizer(ASTUnOpExprNode & ASTBinOpExprNode):
	lhs: ASTNode | None
	op: ...
	rhs: ASTNode | None

	def optimize(self, ns):
		if (self.lhs and self.lhs not in ns): return
		if (self.rhs and self.rhs not in ns): return

		try: res = ASTLiteralNode.fold(f"{ns.value(self.lhs) if (self.lhs) else ''} {self.op} {ns.value(self.rhs) if (self.rhs) else ''}", self)
		except SyntaxError: pass
		else:
			if (self.lhs): ns.unref(self.lhs)
			if (self.rhs): ns.unref(self.rhs)
			return res
