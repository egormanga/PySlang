#!/usr/bin/env python3
# PySlang AST

from __future__ import annotations

from .lexer import Expr, Lexer, Token
from typing import Literal
from utils.nolog import *

class ASTNode(ABCSlotsInit):
	lineno: int
	offset: int
	length: int

	def __repr__(self):
		return f"<{self.__typename__} '{self.__str__()}' on line {self.lineno}, offset {self.offset}>"

	@abc.abstractmethod
	def __str__(self):
		return ''

	@property
	def __typename__(self):
		return self.__class__.__name__.removeprefix('AST').removesuffix('Node')

	@classmethod
	def build(cls, t):
		#print(f"> {cls.__name__}\n")
		res = dict()

		annotations = inspect.get_annotations(cls, eval_str=True)

		for i in t.tokens:
			#print(i, end='\n\n')

			name = i.name
			pattern = None

			try: a = annotations[name]
			except KeyError:
				try:
					if (i.typename == 'pattern'): pattern = name = repr(name.removeprefix('/').removesuffix('/'))
				except AssertionError: pass
				try: name, a = first((k, v) for k, v in annotations.items()
				                            if (a := first(typing.get_args(v), default=v) if (typing_inspect.is_optional_type(v) or isinstance(v, types.UnionType)) else v)
				                            for j in (a, *typing.get_args(a), *Stuple(map(typing.get_args, typing.get_args(a))).flatten())
				                            if name == repr(j))
				except StopIteration: raise WTFException(cls, name)

			if (typing_inspect.is_optional_type(a)): a = first(typing.get_args(a), default=a)

			bt = a
			while (not typing_inspect.is_generic_type(bt)):
				try: bt = first(typing.get_args(bt))
				except StopIteration: break
			bt = (typing_inspect.get_origin(bt) or bt)

			while (True):
				aa = typing.get_args(a)
				if (not aa): break
				if (len(aa) > 1): a = aa; break
				else: a = aa[0]

			for aa in (a if (isiterablenostr(a)) else typing.get_args(a) or (a,)):
				if (isinstance(aa, type) and not isinstance(aa, types.GenericAlias) and issubclass(aa, ASTNode)):
					v = aa.build(i)
					break
			else:
				for aa in a if (isiterablenostr(a)) else (a,):
					if (isinstance(aa, type)): v = aa(i.token)
					elif (i.token == aa or (pattern is not None and pattern == repr(aa))): v = i.token
					else: continue
					break
				else: raise WTFException(cls, i.token)

			if (isinstance(bt, type) and issubclass(bt, list)):
				try: l = res[name]
				except KeyError: l = res[name] = list()
				if (isinstance(v, list)): l += v
				else: l.append(v)
			elif (name in res): raise WTFException(cls, name, v)
			else: res[name] = v

		#e = None
		#try:
		return cls(**res, lineno=t.lineno, offset=t.offset, length=t.length)
		#except Exception as ex: e = ex; raise
		#finally:
		#	if (e is None): print(f"< {cls.__name__}\n")

class ASTChoiceNode(ASTNode):
	@property
	def value(self):
		values = tuple(filter(None, (getattr(self, i) for i in self.__annotations__)))
		if (len(values) != 1): raise WTFException(self, values)
		return first(values)

	def __str__(self):
		return f"{self.value}"

	@classmethod
	def build(cls, t):
		res = super().build(t)
		(res.value)
		return res


## Abstract

class ASTCodeNode(ASTNode):
	_codesep: list[Literal['\n', ';']]
	statement: list[ASTStatementNode]

	def __str__(self):
		return S('\n').join(self.statement)

class ASTBlockNode(ASTNode):
	lbrace: Literal['{'] | None
	_codesep: list[Literal['\n', ';']] | None # TODO FIXME: empty list instead
	code: ASTCodeNode | None
	rbrace: Literal['}'] | None
	colon: Literal[':'] | None
	statement: list[ASTStatementNode] | None # TODO FIXME: empty list instead

	def __str__(self):
		#return (S('\n').join(map(lambda x: x.join('\n\n') if ('\n' in x) else x, map(str, self.nodes))).indent().replace('\n\n\n', '\n\n').strip('\n').join('\n\n') if (self.nodes) else '').join('{}')
		return (Sstr(self.code).indent().join((f"{self.lbrace}\n", f"\n{self.rbrace}")) if (self.colon is None) else f"{self.colon} {S('; ').join(self.statement or ())}")


## Primitive

class ASTBinOpExprNode(ASTNode):
	expr: list[ASTValueNode]
	binop: ASTBinOpNode

	def __str__(self):
		return f"{self.lhs} {self.binop} {self.rhs}"

	@property
	def lhs(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[0]

	@property
	def rhs(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[1]

class ASTValueNode(ASTChoiceNode):
	#unopexpr: ASTUnOpExprNode | None
	binopexpr: ASTBinOpExprNode | None
	funccall: ASTFunccallNode | None
	#itemget: ASTItemgetNode | None
	varname: ASTVarnameNode | None
	#lambda: ASTLambdaNode | None
	literal: ASTLiteralNode | None

class ASTExprNode(ASTNode):
	_lparen: Literal['('] | None
	expr: ASTExprNode | None
	_rparen: Literal[')'] | None
	value: ASTValueNode | None

	def __str__(self):
		return f"{f'({self.expr})' if (self.expr is not None) else self.value}"


## Non-final

class ASTTypenameNode(ASTNode):
	modifier: list[ASTIdentifierNode] | None
	type: ASTTypeNode

	def __str__(self):
		return f"{S(', ').join(self.modifier)+' ' if (self.modifier) else ''}{self.type}"

class ASTVardefAssignmentNode(ASTNode):
	identifier: ASTIdentifierNode
	op: Literal['=']
	expr: ASTValueNode

	def __str__(self):
		return f"{self.identifier}{f' = {self.expr}' if (self.expr) else ''}"

class ASTCallArgNode(ASTNode):
	star: Literal['*'] | None
	expr: ASTValueNode

	def __str__(self):
		return f"{self.star or ''}{self.expr}"

class ASTCallArgsNode(ASTNode):
	callarg: list[ASTCallArgNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{S(', ').join(self.callarg)}"

class ASTCallKwargNode(ASTNode):
	identifier: ASTIdentifierNode | None
	eq: Literal['=', ':'] | None
	expr: ASTValueNode
	star: Literal['**'] | None

	def __str__(self):
		return f"{self.identifier or ''}{self.eq or ''}{' '*(self.eq == ':')}{self.star or ''}{self.expr}"

class ASTCallKwargsNode(ASTNode):
	callkwarg: list[ASTCallKwargNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{S(', ').join(self.callkwarg)}"

class ASTAttrSelfOpNode(ASTNode):
	op: Literal['@.', '@', '.', ':']

	def __str__(self):
		return f"{self.op}"

class ASTAttrOpNode(ASTChoiceNode):
	op: Literal['->'] | None
	attrselfop: ASTAttrSelfOpNode | None

class ASTAttrgetNode(ASTNode):
	attrselfop: ASTAttrSelfOpNode | None
	expr: ASTValueNode | None
	attrop: ASTAttrOpNode | None
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.attrselfop or ''}{self.expr or ''}{self.attrop or ''}{self.identifier}"

## Final

class ASTStatementNode(ASTChoiceNode):
	comment: ASTCommentNode | None
	#funcdef: ASTFuncdefNode | None
	#keywordexpr: ASTKeywordExprNode | None
	vardef: ASTVardefNode | None
	#assignment: ASTAssignmentNode | None
	#inplaceopassignment: ASTInplaceOpAssignmentNode | None
	#unpackassignment: ASTUnpackAssignmentNode | None
	#inplaceopunpackassignment: ASTInplaceOpUnpackAssignmentNode | None
	funccall: ASTFunccallNode | None
	#conditional: ASTConditionalNode | None
	#forloop: ASTForLoopNode | None
	#whileloop: ASTWhileLoopNode | None
	#elseclause: ASTElseClauseNode | None
	keyworddef: ASTKeyworddefNode | None
	#classdef: ASTClassdefNode | None

class ASTVardefNode(ASTNode):
	typename: ASTTypenameNode
	vardefassignment: list[ASTVardefAssignmentNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{self.typename} {S(', ').join(self.vardefassignment)}"

class ASTFunccallNode(ASTNode):
	expr: ASTValueNode
	_lparen: '('
	callargs: list[ASTCallArgsNode] | None
	callkwargs: list[ASTCallKwargsNode] | None
	_comma: Literal[','] | None
	_rparen: ')'

	def __str__(self):
		return f"{self.expr}({S(', ').join((*(self.callargs or ()), *(self.callkwargs or ())))})"

class ASTKeyworddefNode(ASTNode):
	defkeyword: ASTDefkeywordNode
	block: ASTBlockNode

	def __str__(self):
		return f"{self.defkeyword} {self.block}"


## Keywords

class ASTDefkeywordNode(ASTNode):
	keyword: Literal['main', 'exit']

	def __str__(self):
		return f"{self.keyword}"


## Identifiers

class ASTIdentifierNode(ASTNode):
	identifier: r'[^\W\d][\w]*'

	def __str__(self):
		return f"{self.identifier}"

class ASTVarnameNode(ASTChoiceNode):
	attrget: ASTAttrgetNode | None
	identifier: ASTIdentifierNode | None


## Data types

class ASTTypeNode(ASTNode):
	type: str

	def __str__(self):
		return f"{self.type}"


## Literals

class ASTLiteralNode(ASTChoiceNode):
	type: str | None
	number: int | None


## Operators

class ASTBinOpNode(ASTNode):
	binchop: Literal[Lexer.sldef.definitions.binchop.format.literals]

	def __str__(self):
		return f"{self.binchop}"


## Comments

class ASTLineCommentNode(ASTNode):
	lc: '#'
	comment: r'.*'

	def __str__(self):
		return f"{self.lc} {self.comment}"

class ASTBlockCommentNode(ASTNode):
	lbc: '#|'
	comment: r'(.+?)\s*\|\#'
	rbc: '|#'

	def __str__(self):
		return f"{self.lbc} {self.comment} {self.rbc}"

class ASTCommentNode(ASTChoiceNode):
	linecomment: ASTLineCommentNode | None
	blockcomment: ASTBlockCommentNode | None


class AST(SlotsInit):
	code: ASTCodeNode
	scope: str

	def __repr__(self):
		return f"<{self.__typename__} '{self.scope}': {self.code}>"

	def __str__(self):
		return f"{self.code}"

	@classmethod
	def build(cls, st, *, scope=''):
		code = ASTCodeNode.build(st)
		return cls(code=code, scope=scope)

# by Sdore, 2021-24
#  slang.sdore.me
