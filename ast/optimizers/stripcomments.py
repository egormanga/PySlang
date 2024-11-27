from .. import *

class StripCommentsOptimizer(ASTCodeNode & ASTClassCodeNode & ASTClassdefCodeNode & ASTBlockNode & ASTClassBlockNode):
	level = 3

	comment: list[ASTCommentNode] | None

	def optimize(self, ns):
		if (self.comment): self.comment.clear()
