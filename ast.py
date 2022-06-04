#!/usr/bin/env python3
# PySlang AST

from __future__ import annotations

from .lexer import Expr, Lexer, Token
from typing import Literal
from utils.nolog import *

def _commasep(x): return (S(', ').join(x) if (isinstance(x, (tuple, list))) else str(x))

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
			#if (not isinstance(i, Expr) and i.typename == 'literal'): continue

			name = i.name
			#while (isinstance(i, Expr) and i.name == 'expr'):
			#	assert (len(i.tokens) == 1)
			#	i = i.tokens[0]

			#print(i, end='\n\n')

			try: a = annotations[name]
			except KeyError:
				try: name, a = first((k, type(first(typing_inspect.get_args(v), default=v))) for k, v in annotations.items() if any(name == repr(j) for j in (typing_inspect.get_args(v) or (v,))))
				except StopIteration: raise WTFException(cls, name)

			for aa in (typing.get_args(a) or (a,)):
				aa = first(typing_inspect.get_args(aa), default=aa)
				if (isinstance(aa, type) and issubclass(aa, ASTNode)):
					v = aa.build(i)
					break
			else:
				aa = typing.get_args(a)
				if (typing_inspect.is_literal_type(a) or typing_inspect.is_union_type(a) or isinstance(aa, types.UnionType)):
					if (i.token not in aa): raise WTFException(cls, i.token) # XXX
					else: aa = i.token
				else: aa = first(aa, default=a)
				if (isinstance(aa, type)): v = aa(i.token)
				elif (i.token != aa): raise WTFException(cls, i.token) # FIXME: redundant to ^XXX
				else: v = i.token

			bt = a
			while (not typing_inspect.is_generic_type(bt)):
				try: bt = first(typing.get_args(bt))
				except StopIteration: break
			bt = (typing_inspect.get_origin(bt) or bt)

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

class ASTPatternNode(ASTNode):
	@classmethod
	def build(cls, t):
		res = re.fullmatch(cls.PATTERN, t.token)[0]
		return cls(**{first(cls.__annotations__): res}, lineno=t.lineno, offset=t.offset, length=t.length)

	def __str__(self):
		return f"{getattr(self, first(self.__annotations__))}"

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

class ASTCodeNode(ASTNode):
	name: str
	nodes: list

	def __init__(self, name='<code>', nodes=None):
		if (nodes is None): nodes = []
		super().__init__(lineno=1, offset=0, length=0)
		self.name, self.nodes = name, nodes

	def __repr__(self):
		return f"""<{self.__typename__}{f" '{self.name}'" if (self.name and self.name != '<code>') else ''}>"""

	def __str__(self):
		return (S('\n').join(map(lambda x: x.join('\n\n') if ('\n' in x) else x, map(str, self.nodes))).indent().replace('\n\n\n', '\n\n').strip('\n').join('\n\n') if (self.nodes) else '').join('{}')

class ASTIdentifierNode(ASTPatternNode):
	PATTERN = r'[^\W\d][\w]*'

	identifier: str

class ASTVarnameNode(ASTChoiceNode):
	attrget: ASTAttrgetNode | None
	identifier: ASTIdentifierNode | None

class ASTBinOpExprNode(ASTNode):
	expr: list[ASTValueNode]
	binop: ASTBinOp

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
	binopexpr: ASTBinOpExprNode | None
	varname: ASTVarnameNode | None
	literal: ASTLiteralNode | None

class ASTTypenameNode(ASTNode):
	modifier: list[ASTIdentifierNode] | None
	type: ASTIdentifierNode

	def __str__(self):
		return f"{_commasep(self.modifier)+' ' if (self.modifier) else ''}{self.type}"

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
		return f"{_commasep(self.callarg)}"

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
		return f"{_commasep(self.callkwarg)}"

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
		return f"{self.expr or ''}{self.attrop or ''}{self.attrselfop or ''}{self.identifier}"

class ASTVardefNode(ASTNode):
	typename: ASTTypenameNode
	vardefassignment: list[ASTVardefAssignmentNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{self.typename} {_commasep(self.vardefassignment)}"

class ASTFunccallNode(ASTNode):
	expr: ASTValueNode
	_lparen: '('
	callargs: list[ASTCallArgsNode] | None
	callkwargs: list[ASTCallKwargsNode] | None
	_comma: Literal[','] | None
	_rparen: ')'

	def __str__(self):
		return f"{self.expr}({_commasep((*(self.callargs or ()), *(self.callkwargs or ())))})"

class ASTLiteralNode(ASTChoiceNode):
	type: str | None
	number: int | None

class ASTBinOp(ASTNode):
	binchop: Literal[Lexer.sldef.definitions.binchop.format.literals]

	def __str__(self):
		return f"{self.binchop}"

class AST(Slots):
	code: ASTCodeNode
	scope: str

	def __init__(self, scope):
		self.scope = scope

	def __repr__(self):
		return f"<{self.__typename__} '{self.scope}': {self.code}>"

	def __str__(self):
		return f"{self.code}"

	@dispatch
	def add(self, t: Expr):
		match t.name:
			case 'expr': self.add(t.token)
			case 'vardef': self.add(ASTVardefNode.build(t))
			case 'funccall': self.add(ASTFunccallNode.build(t))
			case 'blockcomment': pass
			case _: raise WTFException(t)

	@dispatch
	def add(self, t: ASTNode):
		self.code.nodes.append(t)

	@classmethod
	def build(cls, st, *, scope=''):
		ast = cls(scope=scope)

		for t in st:
			ast.add(t)

		return ast

# by Sdore, 2021-22
#  slang.sdore.me
