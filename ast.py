#!/usr/bin/env python3
# PySlang AST

from __future__ import annotations

from .exceptions import SlValidationError
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

	@abc.abstractmethod
	def validate(self, ns: Namespace) -> None:
		pass

	def optimize(self, ns: Namespace, level: int = 0) -> ASTNode | None:
		return self

	@property
	def __typename__(self):
		return self.__class__.__name__.removeprefix('AST').removesuffix('Node')

	@classmethod
	def build(cls, t: Token):
		#print(f"> {cls.__name__}\n")
		res = dict()

		annotations = inspect.get_annotations(cls, eval_str=True)

		for i in t.tokens:
			#print(i, end='\n\n')

			name = i.name
			pattern = None

			key = name
			if (keyword.iskeyword(key) or keyword.issoftkeyword(key)): key += '_'

			try: a = annotations[key]
			except KeyError:
				try:
					if (i.typename == 'pattern'): pattern = name = repr(name.removeprefix('/').removesuffix('/'))
				except AssertionError: pass
				try: key, a = first((k, v) for k, v in annotations.items()
				                           for a in ((i for i in typing.get_args(v) if i is not NoneType) if (typing_inspect.is_union_type(v)) else (v,))
				                           for j in (a, *typing.get_args(a), *Stuple(map(typing.get_args, typing.get_args(a))).flatten())
				                           if name == repr(j))
				except StopIteration: raise WTFException(cls, name)

			#v = cls.parse(i, a)
			#if (isinstance(v, list)): res.setdefault(key, []).extend(v)
			#elif (key in res): raise WTFException(cls, key, v)
			#else: res[key] = v
			#continue

			###

			if (typing_inspect.is_optional_type(a)):
				assert (typing_inspect.is_union_type(a))
				a = typing.Union[tuple(i for i in typing.get_args(a) if i is not NoneType)]

			bt = a
			while (not typing_inspect.is_generic_type(bt)):
				try: bt = first(typing.get_args(bt))
				except StopIteration: break
			bt = (typing_inspect.get_origin(bt) or bt)

			#dlog('>', a)
			#a_ = a
			#a = (typing.get_args(a) or a)
			#if (isinstance(a, tuple) and len(a) == 1): a = only(a)
			while (True):
				aa = typing.get_args(a)
				if (not aa): break
				if (len(aa) > 1): a = aa; break
				else: a = only(aa)
			#dlog('<', bt, a)
			#if (isiterablenostr(a) and bt in a): a = tuple(i for i in a if i is not bt)
			#else: assert (a == a_ or bt[a] == a_)

			ex = None
			for aa in typing.get_args(a) or (a if (isiterablenostr(a)) else (a,)):
				if (isinstance(aa, type) and not isinstance(aa, types.GenericAlias) and issubclass(aa, ASTNode)):
					v = aa.build(i)
				elif (isinstance(aa, type)):
					try: v = aa(i.token, *((0,) if (aa is int) else ()))
					except Exception as e: ex = e; continue
				elif (i.token == aa or (pattern is not None and pattern == repr(aa))):
					v = i.token
				else: continue
				break
			else: raise WTFException(cls, i) from ex

			if (isinstance(bt, type) and issubclass(bt, list)):
				try: l = res[key]
				except KeyError: l = res[key] = list()
				if (isinstance(v, list)): l += v
				else: l.append(v)
			elif (key in res): raise WTFException(cls, key, v)
			else: res[key] = v

		#e = None
		#try:
		return cls(**res, lineno=t.lineno, offset=t.offset, length=t.length)
		#except Exception as ex: e = ex; raise
		#finally:
		#	if (e is None): print(f"< {cls.__name__}\n")

	@classmethod
	def mimic(cls, node: ASTNode, **kwargs):
		return cls(**kwargs, lineno=node.lineno, offset=node.offset, length=node.length)

	#@classmethod
	#@dispatch
	#def parse(cls, t: Token, a: typing_inspect.is_union_type):
	#	for aa in typing.get_args(a):
	#		return cls.parse(t, aa)
	#
	#@classmethod
	#@dispatch
	#def parse(cls, t: Token, a: lambda: type[ASTNode]):
	#	return a.build(t)
	#
	#@classmethod
	#@dispatch
	#def parse(cls, t: Token, a: type[list]):
	#	return [cls.parse(t, only(typing.get_args(a)))]
	#
	#@classmethod
	#@dispatch
	#def parse(cls, t: Token, a: type[Literal]):
	#	if (t.token in typing.get_args(a)): return t.token
	#
	#@classmethod
	#@dispatch
	#def parse(cls, t: Token, a: str):
	#	if (t.token == a): return t.token
	#
	#@classmethod
	#@dispatch
	#def parse(cls, t: Token, a: type):
	#	return a(t.token)

class ASTSimpleNode(ASTNode):
	def validate(self, ns):
		pass

class ASTChoiceNode(ASTNode):
	def __str__(self):
		try: return f"{self.value}"
		except StopIteration: return super().__str__()

	def validate(self, ns):
		self.value.validate(ns)

	def optimize(self, ns, level=0):
		value = self.value.optimize(ns, level)
		if (not value): return None
		try: return self.__class__(**{first(k for k, v in get_annotations(self.__class__).items() if dispatch._dispatch__typecheck(value, v)): value}, lineno=self.lineno, offset=self.offset, length=self.length)
		except StopIteration: return value

	@property
	def value(self):
		return only(v for i in self.__annotations__ if (v := getattr(self, i)) is not None)

	@value.setter
	def value(self, x):
		setattr(self, only(i for i in self.__annotations__ if getattr(self, i) is not None), x)

	@classmethod
	def build(cls, t):
		res = super().build(t)
		(res.value)
		return res


## Abstract

class ASTCodeNode(ASTNode):
	statement: list[ASTStatementNode] | None
	comment: list[ASTCommentNode] | None
	_codesep: list[Literal['\n', ';']] | None

	def __str__(self):
		return S('\n').join(self.statement or ())

	def validate(self, ns):
		for i in (self.statement or ()):
			i.validate(ns)

	def optimize(self, ns, level=0):
		if (self.statement): self.statement = list(filter(None, (i.optimize(ns, level) for i in self.statement)))
		if (level >= 3 and self.comment): self.comment.clear()
		if (not self.statement and not self.comment): return None
		return super().optimize(ns, level)

class ASTClassCodeNode(ASTNode):
	classstatement: list[ASTClassStatementNode] | None
	comment: list[ASTCommentNode] | None
	_codesep: list[Literal['\n', ';']] | None

	def __str__(self):
		return S('\n').join(self.classstatement or ())

	def validate(self, ns):
		for i in (self.classstatement or ()):
			i.validate(ns)

	def optimize(self, ns, level=0):
		if (self.classstatement): self.classstatement = list(filter(None, (i.optimize(ns, level) for i in self.classstatement)))
		if (level >= 3 and self.comment): self.comment.clear()
		if (not self.classstatement and not self.comment): return None
		return super().optimize(ns, level)

class ASTClassdefCodeNode(ASTNode):
	classdefstatement: list[ASTClassdefStatementNode] | None
	comment: list[ASTCommentNode] | None
	_codesep: list['\n'] | None

	def __str__(self):
		return S('\n').join(self.classdefstatement or ())

	def validate(self, ns):
		for i in (self.classdefstatement or ()):
			i.validate(ns)

	def optimize(self, ns, level=0):
		if (self.classdefstatement): self.classdefstatement = list(filter(None, (i.optimize(ns, level) for i in self.classdefstatement)))
		if (level >= 3 and self.comment): self.comment.clear()
		if (not self.classdefstatement and not self.comment): return None
		return super().optimize(ns, level)

class ASTBlockNode(ASTNode):
	lbrace: Literal['{'] | None
	_nl: list['\n'] | None
	code: ASTCodeNode | None
	rbrace: Literal['}'] | None
	colon: Literal[':'] | None
	statement: list[ASTStatementNode] | None
	_semicolon: list[';'] | None
	comment: list[ASTCommentNode] | None

	def __str__(self):
		#return (S('\n').join(map(lambda x: x.join('\n\n') if ('\n' in x) else x, map(str, self.nodes))).indent().replace('\n\n\n', '\n\n').strip('\n').join('\n\n') if (self.nodes) else '').join('{}')
		return (Sstr(self.code).indent().join((f" {self.lbrace}\n", f"\n{self.rbrace}"))
		        if (self.colon is None)
		        else f"{self.colon} {S('; ').join(self.statement or ())}{self.comment or ''}\n")

	def validate(self, ns):
		if (self.code): self.code.validate(ns)
		for i in (self.statement or ()):
			i.validate(ns)
		if (unused := ns.unused()): raise SlValidationError.from_node(self, "Unused variables", unused)

	def optimize(self, ns, level=0):
		if (self.code): self.code = self.code.optimize(ns, level)
		if (self.statement): self.statement = list(filter(None, (i.optimize(ns, level) for i in self.statement)))
		if (level >= 3 and self.comment): self.comment.clear()
		if (not self.code and not self.statement and not self.comment): return None
		return super().optimize(ns, level)

class ASTClassBlockNode(ASTNode):
	lbrace: Literal['{'] | None
	_nl: list['\n'] | None
	classcode: ASTClassCodeNode | None
	rbrace: Literal['}'] | None
	colon: Literal[':'] | None
	classstatement: list[ASTClassStatementNode] | None
	_semicolon: list[';'] | None
	comment: list[ASTCommentNode] | None

	def __str__(self):
		return (Sstr(self.classcode).indent().join((f" {self.lbrace}\n", f"\n{self.rbrace}"))
		        if (self.colon is None)
		        else f"{self.colon} {S('; ').join(self.classstatement or ())}{self.comment or ''}\n")

	def validate(self, ns):
		if (self.classcode): self.classcode.validate(ns)
		for i in (self.classstatement or ()):
			i.validate(ns)
		if (unused := ns.unused()): raise SlValidationError.from_node(self, "Unused variables", unused)

	def optimize(self, ns, level=0):
		if (self.classcode): self.classcode = self.classcode.optimize(ns, level)
		if (self.classstatement): self.classstatement = list(filter(None, (i.optimize(ns, level) for i in self.classstatement)))
		if (level >= 3 and self.comment): self.comment.clear()
		if (not self.classcode and not self.classstatement and not self.comment): return None
		return super().optimize(ns, level)

class ASTClassdefBlockNode(ASTNode):
	lbrace: '{'
	_codesep: list['\n'] | None
	classdefcode: ASTClassdefCodeNode
	rbrace: '}'

	def __str__(self):
		return Sstr(self.classdefcode).indent().join((f" {self.lbrace}\n", f"\n{self.rbrace}"))

	def validate(self, ns):
		if (self.classdefcode): self.classdefcode.validate(ns)
		for i in (self.classdefstatement or ()):
			i.validate(ns)

	def optimize(self, ns, level=0):
		if (self.classdefcode): self.classdefcode = self.classdefcode.optimize(ns, level)
		if (not self.classdefcode): return None
		return super().optimize(ns, level)


## Primitive

class ASTBinOpExprNode(ASTNode):
	expr: list[ASTExprNode]
	binop: ASTBinOpNode

	def __str__(self):
		return f"{self.lhs} {self.binop} {self.rhs}"

	def validate(self, ns):
		self.lhs.validate(ns)
		self.rhs.validate(ns)

	def optimize(self, ns, level=0):
		lhs = self.lhs = self.lhs.optimize(ns, level)
		rhs = self.rhs = self.rhs.optimize(ns, level)
		if (not lhs or not rhs): return None
		if (lhs in ns and rhs in ns):
			try: res = ASTLiteralNode.fold(f"{ns.value(lhs)} {self.binop} {ns.value(rhs)}", self)
			except SyntaxError: pass
			else:
				ns.unref(lhs)
				ns.unref(rhs)
				return res
		return super().optimize(ns, level)

	@property
	def lhs(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[0]

	@lhs.setter
	def lhs(self, x):
		self.expr[0] = x

	@property
	def rhs(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[-1]

	@rhs.setter
	def rhs(self, x):
		self.expr[-1] = x

class ASTUnPreOpExprNode(ASTNode):
	unop: ASTUnOpNode
	expr: ASTExprNode

	def __str__(self):
		return f"{self.unop}{' '*(str(self.unop)[-1].isalnum())}{self.expr}"

	def validate(self, ns):
		self.expr.validate(ns)

	def optimize(self, ns, level=0):
		expr = self.expr = self.expr.optimize(ns, level)
		if (not expr): return None
		if (self.expr in ns):
			try: res = ASTLiteralNode.fold(f"{self.unop} {ns.value(self.expr)}", self)
			except SyntaxError: pass
			else:
				ns.unref(self.expr)
				return res
		return super().optimize(ns, level)

class ASTUnPostOpExprNode(ASTNode):
	expr: ASTExprNode
	unpostop: ASTUnPostOpNode

	def __str__(self):
		return f"{self.expr}{self.unpostop}"

	def validate(self, ns):
		self.expr.validate(ns)

	def optimize(self, ns, level=0):
		expr = self.expr = self.expr.optimize(ns, level)
		if (not expr): return None
		return super().optimize(ns, level)

class ASTUnOpExprNode(ASTChoiceNode):
	unpreopexpr: ASTUnPreOpExprNode | None
	unpostopexpr: ASTUnPostOpExprNode | None

	def validate(self, ns):
		self.value.validate(ns)

	def optimize(self, ns, level=0):
		value = self.value = self.value.optimize(ns, level)
		if (not value): return None
		return super().optimize(ns, level)

class ASTItemgetNode(ASTNode):
	expr: list[ASTExprNode]
	lbrk: '['
	rbrk: ']'

	def __str__(self):
		return f"{self.value}{self.lbrk}{self.index}{self.rbrk}"

	def validate(self, ns):
		for i in self.expr:
			i.validate(ns)

	def optimize(self, ns, level=0):
		value = self.value = self.value.optimize(ns, level)
		index = self.index = self.index.optimize(ns, level)
		if (not value or not index): return None
		return super().optimize(ns, level)

	@property
	def value(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[0]

	@value.setter
	def value(self, x):
		self.expr[0] = x

	@property
	def index(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[-1]

	@index.setter
	def index(self, x):
		self.expr[-1] = x

class ASTValueNode(ASTChoiceNode):
	binopexpr: ASTBinOpExprNode | None
	unopexpr: ASTUnOpExprNode | None
	itemget: ASTItemgetNode | None
	funccall: ASTFunccallNode | None
	varname: ASTVarnameNode | None
	#lambda: ASTLambdaNode | None
	literal: ASTLiteralNode | None

	def validate(self, ns):
		self.value.validate(ns)
		if (self.varname and self.varname.identifier): ns.ref(self.varname)

class ASTExprNode(ASTNode):
	_lparen: Literal['('] | None
	expr: ASTExprNode | None
	_rparen: Literal[')'] | None
	value: ASTValueNode | None

	def __str__(self):
		return f"{f'({self.expr})' if (self.expr is not None) else self.value}"

	def validate(self, ns):
		if (self.value): self.value.validate(ns)
		else: self.expr.validate(ns)

	def optimize(self, ns, level=0):
		if (self.expr): return self.expr.optimize(ns, level)
		return self.value.optimize(ns, level)


## Non-final

class ASTTypeNode(ASTNode):
	modifier: list[Literal[Lexer.sldef.definitions.modifier.format.literals]] | None
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{S(', ').join(self.modifier)+' ' if (self.modifier) else ''}{self.identifier}"

	def __eq__(self, other):
		return (isinstance(other, self.__class__) and self.identifier == other.identifier and self.modifier == other.modifier)

	def validate(self, ns):
		self.identifier.validate(ns)
		ns.ref(self.identifier)

	def optimize(self, ns, level=0):
		identifier = self.identifier = self.identifier.optimize(ns, level)
		if (not identifier): return None
		return super().optimize(ns, level)

class ASTCatchClauseNode(ASTNode):
	catch: 'catch'
	type_: ASTTypeNode | None
	identifier: ASTIdentifierNode | None
	block: ASTBlockNode

	def __str__(self):
		return f"{self.catch}{f' {self.type_}' if (self.type_ is not None) else ''}{f' {self.identifier}' if (self.identifier is not None) else ''}{self.block}"

	def validate(self, ns):
		self.type_.validate(ns)
		self.identifier.validate(ns)

		blockns = ns.derive(self)
		blockns.define(self.identifier, self.type_)
		self.block.validate(blockns)

	def optimize(self, ns, level=0):
		self.type_ = self.type_.optimize(ns, level)
		self.identifier = self.identifier.optimize(ns, level)

		blockns = ns.derive(self)
		self.block = self.block.optimize(blockns)
		return super().optimize(ns, level)

class ASTFinallyClauseNode(ASTNode):
	finally_: 'finally'
	block: ASTBlockNode

	def __str__(self):
		return f"{self.finally_}{self.block}"

	def validate(self, ns):
		blockns = ns.derive(self)
		self.block.validate(blockns)

	def optimize(self, ns, level=0):
		blockns = ns.derive(self)
		self.block = self.block.optimize(blockns)
		return super().optimize(ns, level)

class ASTVardefAssignmentNode(ASTNode):
	identifier: ASTIdentifierNode
	op: Literal['='] | None
	expr: ASTExprNode | None
	funccallargs: ASTFunccallArgsNode | None

	def __str__(self):
		return f"{self.identifier}{f' {self.op} {self.expr}' if (self.expr) else ''}{self.funccallargs or ''}"

	def validate(self, ns, *, type_):
		self.identifier.validate(ns)
		if (self.expr): self.expr.validate(ns)
		if (self.funccallargs): self.funccallargs.validate(ns)

		ns.define(self.identifier, type_, value=self.expr)

	def optimize(self, ns, level=0):
		if (level == 0 and not ns.refs(self.identifier)): return None

		identifier = self.identifier = self.identifier.optimize(ns, level)
		if (self.expr): self.expr = self.expr.optimize(ns, level)
		if (self.funccallargs): self.funccallargs = self.funccallargs.optimize(ns, level)
		if (not identifier): return None
		return super().optimize(ns, level)

class ASTArgdefNode(ASTNode):
	special: Literal['/', '*'] | None
	type_: ASTTypeNode
	identifier: ASTIdentifierNode
	lbrk: Literal['['] | None
	integer: int | None
	rbrk: Literal[']'] | None
	mode: Literal['?', '+', '*', '**', '='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.special or ''}{self.type_} {self.identifier}{f'{self.lbrk}{self.integer}{self.rbrk}' if (self.integer is not None) else ''}{self.mode or ''}{self.expr or ''}"

	def validate(self, ns):
		self.type_.validate(ns)
		self.identifier.validate(ns)
		if (self.expr): self.expr.validate(ns)

		ns.define(self.identifier, self.type_, value=self.expr)

class ASTClassArgdefNode(ASTNode):
	type_: ASTTypeNode
	classvarname: ASTClassVarnameNode
	mode: Literal['?', '+', '*', '**', '='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.type_} {self.classvarname}{self.mode or ''}{self.expr or ''}"

	def validate(self, ns):
		self.type_.validate(ns)
		self.classvarname.validate(ns)
		if (self.expr): self.expr.validate(ns)

		ns.define(self.classvarname, self.type_, value=self.expr)

class ASTCallArgNode(ASTNode):
	star: Literal['*'] | None
	expr: ASTExprNode

	def __str__(self):
		return f"{self.star or ''}{self.expr}"

	def validate(self, ns):
		self.expr.validate(ns)

	def optimize(self, ns, level=0):
		expr = self.expr = self.expr.optimize(ns, level)
		if (not expr): return None
		return super().optimize(ns, level)

class ASTCallArgsNode(ASTNode):
	callarg: list[ASTCallArgNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{S(', ').join(self.callarg)}"

	def validate(self, ns):
		for i in self.callarg:
			i.validate(ns)

	def optimize(self, ns, level=0):
		callarg = self.callarg = list(filter(None, (i.optimize(ns, level) for i in self.callarg)))
		if (not callarg): return None
		return super().optimize(ns, level)

class ASTCallKwargNode(ASTNode):
	identifier: ASTIdentifierNode
	eq: Literal['=', ':'] | None
	expr: ASTExprNode | None
	star: Literal['**'] | None

	def __str__(self):
		return f"{not self.expr and self.eq or ''}{self.identifier}{self.eq or ''}{' '*(self.eq == ':')}{self.star or ''}{self.expr or ''}"

	def validate(self, ns):
		self.identifier.validate(ns)
		if (self.expr):
			self.expr.validate(ns)

	def optimize(self, ns, level=0):
		identifier = self.identifier = self.identifier.optimize(ns, level)
		if (self.expr): self.expr = self.expr.optimize(ns, level)
		if (not identifier and not self.expr): return None
		return super().optimize(ns, level)

class ASTCallKwargsNode(ASTNode):
	callkwarg: list[ASTCallKwargNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{S(', ').join(self.callkwarg)}"

	def validate(self, ns):
		for i in self.callkwarg:
			i.validate(ns)

	def optimize(self, ns, level=0):
		callkwarg = self.callkwarg = list(filter(None, (i.optimize(ns, level) for i in self.callkwarg)))
		if (not callkwarg): return None
		return super().optimize(ns, level)

class ASTFunccallArgsNode(ASTNode):
	lparen: '('
	callargs: list[ASTCallArgsNode] | None
	_comma: list[','] | None
	callkwargs: list[ASTCallKwargsNode] | None
	rparen: ')'

	def __str__(self):
		return f"{self.lparen}{S(', ').join((*(self.callargs or ()), *(self.callkwargs or ())))}{self.rparen}"

	def validate(self, ns):
		for i in (self.callargs or ()):
			i.validate(ns)

		for i in (self.callkwargs or ()):
			i.validate(ns)

	def optimize(self, ns, level=0):
		if (self.callargs): self.callargs = list(filter(None, (i.optimize(ns, level) for i in self.callargs)))
		if (self.callkwargs): self.callkwargs = list(filter(None, (i.optimize(ns, level) for i in self.callkwargs)))
		return super().optimize(ns, level)

class ASTAttrSelfOpNode(ASTSimpleNode):
	op: Literal['@.', '@', '.', ':']

	def __str__(self):
		return f"{self.op}"

class ASTAttrOpNode(ASTChoiceNode):
	op: Literal['->'] | None
	attrselfop: ASTAttrSelfOpNode | None

class ASTAttrgetNode(ASTNode):
	expr: ASTExprNode
	attrop: ASTAttrOpNode
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.expr}{self.attrop}{self.identifier}"

	def validate(self, ns):
		self.expr.validate(ns)
		self.identifier.validate(ns)

class ASTClassAttrgetNode(ASTNode):
	attrselfop: ASTAttrSelfOpNode | None
	expr: ASTExprNode | None
	attrop: ASTAttrOpNode | None
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.attrselfop or ''}{self.expr or ''}{self.attrop or ''}{self.identifier}"

	def validate(self, ns):
		self.expr.validate(ns)
		self.identifier.validate(ns)


## Final

class ASTStatementNode(ASTChoiceNode):
	comment: ASTCommentNode | None
	reserved: ASTReservedNode | None
	funcdef: ASTFuncdefNode | None
	keywordexpr: ASTKeywordExprNode | None
	conditional: ASTConditionalNode | None
	forloop: ASTForLoopNode | None
	whileloop: ASTWhileLoopNode | None
	doblock: ASTDoBlockNode | None
	classdef: ASTClassdefNode | None
	vardef: ASTVardefNode | None
	assignment: ASTAssignmentNode | None
	funccall: ASTFunccallNode | None
	keyworddef: ASTKeyworddefNode | None

class ASTClassStatementNode(ASTChoiceNode):
	comment: ASTCommentNode | None
	reserved: ASTReservedNode | None
	funcdef: ASTFuncdefNode | None
	keywordexpr: ASTKeywordExprNode | None
	conditional: ASTConditionalNode | None
	forloop: ASTForLoopNode | None
	whileloop: ASTWhileLoopNode | None
	doblock: ASTDoBlockNode | None
	vardef: ASTVardefNode | None
	classassignment: ASTClassAssignmentNode | None
	funccall: ASTFunccallNode | None

class ASTClassdefStatementNode(ASTChoiceNode):
	comment: ASTCommentNode | None
	reserved: ASTReservedNode | None
	classfuncdef: ASTClassFuncdefNode | None
	classdef: ASTClassdefNode | None
	vardef: ASTVardefNode | None
	classkeyworddef: ASTClassKeyworddefNode | None

class ASTFuncdefNode(ASTNode):
	type_: ASTTypeNode | None
	def_: Literal['def'] | None
	identifier: ASTIdentifierNode
	lparen: '('
	argdef: list[ASTArgdefNode] | None
	_comma: list[','] | None
	rparen: ')'
	block: ASTBlockNode | None
	eq: Literal['='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.type_ or self.def_} {self.identifier}{self.lparen}{S(', ').join(self.argdef or ())}{self.rparen}{self.block or ''}{f' {self.eq} {self.expr}' if (self.expr) else ''}"

	def validate(self, ns):
		self.type_.validate(ns)
		self.identifier.validate(ns)

		funcns = ns.derive(self)
		for i in (self.argdef or ()):
			i.validate(funcns)

		ns.define(self.identifier, self.type_, value=self.expr)

		if (self.block): self.block.validate(funcns)
		if (self.expr): self.expr.validate(funcns)

		ns.assign(self.identifier, self.expr)

	def optimize(self, ns, level=0):
		if (self.type_): self.type_ = self.type_.optimize(ns, level)
		identifier = self.identifier = self.identifier.optimize(ns, level)

		funcns = ns.derive(self)
		if (self.argdef): self.argdef = list(filter(None, (i.optimize(funcns, level) for i in self.argdef)))
		if (self.block): self.block = self.block.optimize(funcns, level)
		if (self.expr): self.expr = self.expr.optimize(funcns, level)

		if (not identifier): return None
		return super().optimize(ns, level)

class ASTClassFuncdefNode(ASTNode):
	type_: ASTTypeNode | None
	method: Literal['method'] | None
	identifier: ASTIdentifierNode
	lparen: '('
	argdef: list[ASTArgdefNode] | None
	_comma: list[','] | None
	rparen: ')'
	classblock: ASTClassBlockNode | None
	eq: Literal['='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.type_ or self.method} {self.identifier}{self.lparen}{S(', ').join(self.argdef or ())}{self.rparen}{self.classblock or ''}{f' {self.eq} {self.expr}' if (self.expr) else ''}"

	def validate(self, ns):
		self.type_.validate(ns)
		self.identifier.validate(ns)

		funcns = ns.derive(self)
		for i in (self.argdef or ()):
			i.validate(funcns)

		ns.define(self.identifier, self.type_, value=self.expr)

		if (self.classblock): self.classblock.validate(funcns)
		if (self.expr): self.expr.validate(funcns)

		ns.assign(self.identifier, self.expr)

	def optimize(self, ns, level=0):
		if (self.type_): self.type_ = self.type_.optimize(ns, level)
		identifier = self.identifier = self.identifier.optimize(ns, level)

		funcns = ns.derive(self)
		if (self.argdef): self.argdef = list(filter(None, (i.optimize(funcns, level) for i in self.argdef)))
		if (self.classblock): self.block = self.classblock.optimize(funcns, level)
		if (self.expr): self.expr = self.expr.optimize(funcns, level)

		if (not identifier): return None
		return super().optimize(ns, level)

class ASTKeywordExprNode(ASTChoiceNode):
	return_: ASTReturnNode | None
	raise_: ASTRaiseNode | None
	throw: ASTThrowNode | None
	resume: ASTResumeNode | None
	break_: ASTBreakNode | None
	continue_: ASTContinueNode | None
	fallthrough: ASTFallthroughNode | None
	#import_: ASTImportNode | None
	delete: ASTDeleteNode | None
	#assert_: ASTAssertNode | None
	#super: ASTSuperNode | None
	#breakpoint: ASTBreakpointNode | None

class ASTConditionalNode(ASTNode):
	if_: 'if'
	expr: list[ASTExprNode]
	block: list[ASTBlockNode]
	_elif: list['elif'] | None
	_nl: list['\n'] | None
	else_: Literal['else'] | None

	def __str__(self):
		return f"{self.if_} {S(' elif ').join(f'{expr}{block}' for expr, block in zip(self.expr, self.block))}{f' {self.else_}{self.elseblock}' if (self.else_) else ''}"

	def validate(self, ns):
		for i in self.expr:
			i.validate(ns)

		for ii, i in enumerate(self.block):
			blockns = ns.derive(self, ii)
			i.validate(blockns)

	def optimize(self, ns, level=0):
		expr = self.expr = list(filter(None, (i.optimize(ns, level) for i in self.expr)))

		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			self.block[ii] = i.optimize(loopns)

		if (not expr): return None
		return super().optimize(ns, level)

	@property
	def elseblock(self):
		return self.block[-1]

class ASTForLoopNode(ASTNode):
	for_: 'for'
	type_: ASTTypeNode
	identifier: ASTIdentifierNode
	in_: 'in'
	expr: ASTExprNode
	block: list[ASTBlockNode]
	_nl: list['\n'] | None
	else_: Literal['else'] | None

	def __str__(self):
		return f"{self.for_} {self.type_} {self.identifier} {self.in_} {self.expr}{self.forblock}{f' {self.else_}{self.elseblock}' if (self.else_) else ''}"

	def validate(self, ns):
		self.type_.validate(ns)
		self.identifier.validate(ns)
		self.expr.validate(ns)

		ns.define(self.identifier, self.type_)

		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			i.validate(loopns)

	def optimize(self, ns, level=0):
		self.type_ = self.type_.optimize(ns, level)
		self.identifier = self.identifier.optimize(ns, level)

		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			self.block[ii] = i.optimize(loopns)

		return super().optimize(ns, level)

	@property
	def forblock(self):
		return self.block[0]

	@property
	def elseblock(self):
		return self.block[-1]

class ASTWhileLoopNode(ASTNode):
	while_: 'while'
	expr: ASTExprNode
	block: list[ASTBlockNode]
	_nl: list['\n'] | None
	else_: Literal['else'] | None

	def __str__(self):
		return f"{self.while_} {self.expr}{self.whileblock}{f' {self.else_}{self.elseblock}' if (self.else_) else ''}"

	def validate(self, ns):
		self.expr.validate(ns)

		for ii, i in enumerate(self.block):
			blockns = ns.derive(self, ii)
			i.validate(blockns)

	def optimize(self, ns, level=0):
		expr = self.expr = self.expr.optimize(ns, level)

		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			self.block[ii] = i.optimize(loopns)

		if (not expr): return None
		return super().optimize(ns, level)

	@property
	def whileblock(self):
		return self.block[0]

	@property
	def elseblock(self):
		return self.block[-1]

class ASTDoBlockNode(ASTNode):
	do_: 'do'
	block: list[ASTBlockNode]
	_nl: list['\n'] | None
	catchclause: list[ASTCatchClauseNode] | None
	else_: Literal['else'] | None
	finallyclause: ASTFinallyClauseNode | None

	def __str__(self):
		return f"""{self.do_}{self.block[0]}{f" {S(' ').join(self.catchclause)}" if (self.catchclause) else ''}{f' {self.else_}{self.elseblock}' if (self.else_) else ''}{f" {self.finallyclause}" if (self.finallyclause) else ''}"""

	def validate(self, ns):
		for ii, i in enumerate(self.block):
			blockns = ns.derive(self, ii)
			i.validate(blockns)

		for i in self.catchclause:
			i.validate(ns)

		self.finallyclause.validate(ns)

	def optimize(self, ns, level=0):
		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			self.block[ii] = i.optimize(loopns)

		self.catchclause = list(filter(None, (i.optimize(ns, level) for i in self.catchclause)))
		self.finalyclause = list(filter(None, (i.optimize(ns, level) for i in self.finallyclause)))

		return super().optimize(ns, level)

	@property
	def doblock(self):
		return self.block[0]

	@property
	def elseblock(self):
		return self.block[-1]

class ASTClassdefNode(ASTNode):
	class_: 'class'
	identifier: list[ASTIdentifierNode]
	_lt: list['<'] | None
	classdefblock: ASTClassdefBlockNode

	def __str__(self):
		return f"""{self.class_} {S(' < ').join(self.identifier)}{self.classdefblock}"""

	def validate(self, ns):
		self.identifier.validate(ns)

		classns = ns.derive(self)
		self.classdefblock.validate(classns)

	@property
	def classname(self):
		return self.identifier[0]

	@property
	def bases(self):
		return self.identifier[1:]

class ASTVardefNode(ASTNode):
	type_: ASTTypeNode
	vardefassignment: list[ASTVardefAssignmentNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{self.type_} {S(', ').join(self.vardefassignment)}"

	def validate(self, ns):
		self.type_.validate(ns)
		for i in self.vardefassignment:
			i.validate(ns, type_=self.type_)

	def optimize(self, ns, level=0):
		self.type_ = self.type_.optimize(ns, level)
		vardefassignment = self.vardefassignment = list(filter(None, (i.optimize(ns, level) for i in self.vardefassignment)))
		if (not vardefassignment): return None
		return super().optimize(ns, level)

class ASTAssignmentNode(ASTNode):
	varname: list[ASTVarnameNode]
	_comma: list[','] | None
	assignment: Literal['='] | str
	expr: ASTExprNode

	def __str__(self):
		return f"{S(', ').join(self.varname)} {self.assignment} {self.expr}"

	def validate(self, ns):
		for i in self.varname:
			i.validate(ns)
		self.expr.validate(ns)

		if (len(self.varname) == 1): ns.assign(only(self.varname), self.expr) # TODO: unpack

	def optimize(self, ns, level=0):
		varname = self.varname = list(filter(None, (i.optimize(ns, level) for i in self.varname)))
		expr = self.expr = self.expr.optimize(ns, level)
		if (not varname or not expr): return None
		return super().optimize(ns, level)

class ASTClassAssignmentNode(ASTNode):
	classvarname: list[ASTClassVarnameNode]
	_comma: list[','] | None
	assignment: Literal['='] | str
	expr: ASTExprNode

	def __str__(self):
		return f"{S(', ').join(self.classvarname)} {self.assignment} {self.expr}"

	def validate(self, ns):
		for i in self.classvarname:
			i.validate(ns)
		self.expr.validate(ns)

		if (len(self.classvarname) == 1): ns.assign(only(self.classvarname), self.expr) # TODO: unpack

	def optimize(self, ns, level=0):
		classvarname = self.classvarname = list(filter(None, (i.optimize(ns, level) for i in self.classvarname)))
		expr = self.expr = self.expr.optimize(ns, level)
		if (not classvarname or not expr): return None
		return super().optimize(ns, level)

class ASTFunccallNode(ASTNode):
	expr: ASTExprNode
	funccallargs: ASTFunccallArgsNode

	def __str__(self):
		return f"{self.expr}{self.funccallargs}"

	def validate(self, ns):
		self.expr.validate(ns)
		self.funccallargs.validate(ns)

	def optimize(self, ns, level=0):
		expr = self.expr = self.expr.optimize(ns, level)
		funccallargs = self.funccallargs = self.funccallargs.optimize(ns, level)
		if (not expr or not funccallargs): return None
		return super().optimize(ns, level)

class ASTKeyworddefNode(ASTNode):
	defkeyword: ASTDefkeywordNode
	block: ASTBlockNode

	def __str__(self):
		return f"{self.defkeyword}{self.block}"

	def validate(self, ns):
		blockns = ns.derive(self)
		self.block.validate(blockns)

	def optimize(self, ns, level=0):
		blockns = ns.derive(self)
		block = self.block = self.block.optimize(blockns, level)
		if (not block): return None
		return super().optimize(ns, level)

class ASTClassKeyworddefNode(ASTNode):
	classdefkeyword: ASTClassDefkeywordNode
	classargdef: list[ASTClassArgdefNode] | None
	lparen: Literal['('] | None
	_comma: list[','] | None
	_rparen: Literal[')'] | None
	classblock: ASTClassBlockNode | None
	semicolon: Literal[';'] | None

	def __str__(self):
		return f"""{self.classdefkeyword}{f" ({S(', ').join(self.classargdef or ())})" if (self.classargdef or self.lparen) else ''}{self.classblock or self.semicolon}"""

	def validate(self, ns):
		blockns = ns.derive(self)
		self.classdefblock.validate(blockns)

	def optimize(self, ns, level=0):
		blockns = ns.derive(self)
		classdefblock = self.classdefblock = self.classdefblock.optimize(blockns, level)
		if (not classdefblock): return None
		return super().optimize(ns, level)


## Keywords

class ASTReturnNode(ASTNode):
	return_: 'return'
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.return_}{f' {self.expr}' if (self.expr is not None) else ''}"

	def validate(self, ns):
		self.expr.validate(ns)

	def optimize(self, ns, level=0):
		expr = self.expr = self.expr.optimize(ns, level)
		if (not expr): return None
		return super().optimize(ns, level)

class ASTRaiseNode(ASTSimpleNode):
	raise_: 'raise'

	def __str__(self):
		return f"{self.raise_}"

class ASTThrowNode(ASTNode):
	throw: 'throw'
	expr: ASTExprNode

	def __str__(self):
		return f"{self.throw} {self.expr}"

	def validate(self, ns):
		self.expr.validate(ns)

	def optimize(self, ns, level=0):
		expr = self.expr = self.expr.optimize(ns, level)
		if (not expr): return None
		return super().optimize(ns, level)

class ASTResumeNode(ASTNode):
	resume: 'resume'
	integer: int | None

	def __str__(self):
		return f"{self.resume}{f' {self.integer}' if (self.integer) else ''}"

	def validate(self, ns):
		assert (self.integer > 0)

class ASTBreakNode(ASTNode):
	break_: 'break'
	integer: int | None

	def __str__(self):
		return f"{self.break_}{f' {self.integer}' if (self.integer) else ''}"

	def validate(self, ns):
		assert (self.integer > 0)

class ASTContinueNode(ASTNode):
	continue_: 'continue'
	integer: int | None

	def __str__(self):
		return f"{self.continue_}{f' {self.integer}' if (self.integer) else ''}"

	def validate(self, ns):
		assert (self.integer > 0)

class ASTFallthroughNode(ASTSimpleNode):
	fallthrough: 'fallthrough'

	def __str__(self):
		return f"{self.fallthrough}"

class ASTDeleteNode(ASTNode):
	delete: 'delete'
	varname: ASTVarnameNode

	def __str__(self):
		return f"{self.delete} {self.varname}"

	def validate(self, ns):
		self.ns.delete(self.varname)

	def optimize(self, ns, level=0):
		varname = self.varname = self.varname.optimize(ns, level)
		if (not varname): return None
		return super().optimize(ns, level)

class ASTDefkeywordNode(ASTSimpleNode):
	defkeyword: Literal['main', 'exit']

	def __str__(self):
		return f"{self.defkeyword}"

class ASTClassDefkeywordNode(ASTSimpleNode):
	classdefkeyword: Literal['init', 'destroy', 'constr', 'property', 'repr', 'eq', 'ne', *(i+j for i in 'lg' for j in 'te')]

	def __str__(self):
		return f"{self.classdefkeyword}"

class ASTReservedNode(ASTSimpleNode):
	reserved: Literal[Lexer.sldef.definitions.reserved.format.literals]

	def __str__(self):
		return f"{self.reserved}"

	def validate(self, ns):
		raise SlValidationError("Reserved keyword", self.reserved)


## Identifiers

class ASTIdentifierNode(ASTSimpleNode):
	_minified = int()

	identifier: r'[^\W\d][\w]*'

	def __str__(self):
		return f"{self.identifier}"

	def __eq__(self, other):
		return (isinstance(other, self.__class__) and self.identifier == other.identifier)

	def optimize(self, ns, level=0):
		if (level >= 5):
			cached = self._minify.is_cached(self.identifier)
			identifier = self._minify(self.identifier)
			if (not cached): ns.rename(self.identifier, identifier)
			self.identifier = identifier
		return super().optimize(ns, level)

	@classmethod
	@cachedfunction
	def _minify(cls, identifier):
		r = base64.b32encode(cls._minified.to_bytes(((cls._minified.bit_length() + 7) // 8) or 1)).rstrip(b'=').decode()
		if (not len(r) < len(identifier)): return identifier
		cls._minified += 1
		return r


class ASTVarnameNode(ASTChoiceNode):
	attrget: ASTAttrgetNode | None
	identifier: ASTIdentifierNode | None

class ASTClassVarnameNode(ASTChoiceNode):
	classattrget: ASTClassAttrgetNode | None
	identifier: ASTIdentifierNode | None


## Literals

class ASTListNode(ASTNode):
	lbrk: '['
	type_: ASTTypeNode
	colon: Literal[':'] | None
	expr: list[ASTExprNode] | None
	_comma: list[','] | None
	rbrk: ']'

	def __str__(self):
		return f"""{self.lbrk}{self.type_}{f"{self.colon} {S(', ').join(self.expr)}" if (self.expr) else ''}{self.rbrk}"""

class ASTLiteralNode(ASTChoiceNode):
	boolean: Literal['true', 'false'] | None
	number: int | float | complex | str | None
	character: str | None
	string: str | None
	#tuple: ASTTupleNode | None
	list: ASTListNode | None

	def validate(self, ns):
		return super(ASTChoiceNode, self).validate(ns)

	def optimize(self, ns, level=0):
		return super(ASTChoiceNode, self).optimize(ns, level)

	@classmethod
	def fold(cls, expr: str, node: ASTNode):
		match eval(expr, {}):
			case bool(r): return cls.mimic(node, boolean=('true', 'false')[r])
			case int(r) | float(r) | complex(r): return cls.mimic(node, number=r)
			case str(r):
				match len(r):
					case 1: return cls.mimic(code, character=f"'{'\\'*(r == '\'')}{r}'")
					case _: return cls.mimic(node, string=repr(r))
			case _: raise WTFException(_)


## Operators

class ASTUnOpNode(ASTChoiceNode):
	unchop: Literal[Lexer.sldef.definitions.unchop.format.literals] | None
	undchop: Literal[Lexer.sldef.definitions.undchop.format.literals] | None
	unkwop: Literal[Lexer.sldef.definitions.unkwop.format.literals] | None
	unmathop: str | None

class ASTUnPostOpNode(ASTChoiceNode):
	unchpostop: Literal[Lexer.sldef.definitions.unchpostop.format.literals] | None
	undchpostop: Literal[Lexer.sldef.definitions.undchpostop.format.literals] | None

class ASTBinOpNode(ASTChoiceNode):
	binchop: Literal[Lexer.sldef.definitions.binchop.format.literals] | None
	bindchop: Literal[Lexer.sldef.definitions.bindchop.format.literals] | None
	binkwop: Literal[Lexer.sldef.definitions.binkwop.format.literals] | None
	binmathop: Literal[Lexer.sldef.definitions.binmathop.format.literals] | None


## Comments

class ASTBlockCommentNode(ASTSimpleNode):
	blockcomment: str

	def __str__(self):
		return f"{self.blockcomment}"

class ASTLineCommentNode(ASTSimpleNode):
	linecomment: str

	def __str__(self):
		return f"{self.linecomment}"

class ASTCommentNode(ASTChoiceNode):
	blockcomment: ASTBlockCommentNode | None
	linecomment: ASTLineCommentNode | None

	def validate(self, ns):
		pass

	def optimize(self, ns, level=0):
		if (level >= 3): return None
		return super().optimize(ns, level)


##

class Namespace(Slots):
	parent: ...
	types: dict
	values: dict
	refcnt: dict

	def __init__(self, parent=None):
		self.parent = parent

	def __repr__(self):
		return '\n\t '.join(f"<{self.refcnt.get(i, '?')}> {self.types.get(i, '*')} {i} = {self.values.get(i, '…')}" for i in S((*self.types, *self.values, *self.refcnt)).uniquize()).join('{}')

	def __contains__(self, x):
		try: self.values[x]
		except KeyError: return False
		else: return True

	@cachedfunction
	def derive(self, node: ASTNode, index: int = 0):
		return self.__class__(parent=self)

	@dispatch
	def value(self, name: ASTValueNode | ASTVarnameNode):
		return self.value(name.value)

	@dispatch
	def value(self, name: ASTIdentifierNode):
		return self.value(name.identifier)

	@dispatch
	def value(self, name: str):
		return self.values[name]

	@dispatch
	def value(self, value: ASTLiteralNode):
		return value.value

	@dispatch
	def define(self, name: ASTVarnameNode, type_: ASTTypeNode, value=None):
		self.define(name.value, type_, value=value)

	@dispatch
	def define(self, name: ASTIdentifierNode, type_: ASTTypeNode, value=None):
		self.define(name.identifier, type_, value=value)

	@dispatch
	def define(self, name: str, type_: ASTTypeNode):
		if ((t := self.types.get(name, type_)) != type_): raise ValueError(name, "redefined", t, type_)
		self.types[name] = type_
		self.refcnt[name] = 0

	@dispatch
	def define(self, name: str, type_: ASTTypeNode, value):
		self.define(name, type_)
		self.values[name] = value
		self.refcnt[name] = 0

	@dispatch
	def assign(self, name: ASTVarnameNode, value):
		self.assign(name.value, value)

	@dispatch
	def assign(self, name: ASTIdentifierNode, value):
		self.assign(name.identifier, value)

	@dispatch
	def assign(self, name: str, value, *, _set=True):
		if (name not in self.types and self.parent is not None):
			try: self.parent.assign(name, value, _set=False)
			except KeyError: pass
			else: return

		if (_set): self.values[name] = value
		else: raise KeyError(name)

	@dispatch
	def rename(self, old: str, new: str):
		try: self.types[new] = self.types.pop(old)
		except KeyError: pass
		try: self.values[new] = self.values.pop(old)
		except KeyError: pass
		try: self.refcnt[new] = self.refcnt.pop(old)
		except KeyError: pass

	@dispatch
	def delete(self, name: ASTVarnameNode):
		self.delete(name.value)

	@dispatch
	def delete(self, name: ASTIdentifierNode):
		self.delete(name.identifier)

	@dispatch
	def delete(self, name: str):
		del self.types[name]
		del self.values[name]
		del self.refcnt[name]

	@dispatch
	def ref(self, name: ASTVarnameNode, *, _set=True):
		self.ref(name.value)

	@dispatch
	def ref(self, name: ASTIdentifierNode, *, _set=True):
		try: self.refcnt[name.identifier] += 1
		except KeyError:
			if (self.parent is not None):
				try: self.parent.ref(name, _set=False)
				except KeyError: pass
				else: return

			if (_set): self.refcnt[name.identifier] = 1
			else: raise

	@dispatch
	def unref(self, name: ASTValueNode | ASTVarnameNode):
		self.unref(name.value)

	@dispatch
	def unref(self, name: ASTIdentifierNode):
		try: self.refcnt[name.identifier] -= 1
		except KeyError:
			if (self.parent is not None): self.parent.unref(name)
			else: raise
		else: assert (self.refcnt[name.identifier] >= 0)

	@dispatch
	def unref(self, value: ASTLiteralNode):
		pass

	@dispatch
	def refs(self, name: ASTVarnameNode):
		return self.refs(name.value)

	@dispatch
	def refs(self, name: ASTIdentifierNode):
		try: return self.refcnt[name.identifier]
		except KeyError:
			if (self.parent is not None): self.parent.refs(name)
			else: raise

	def unused(self) -> set[str]:
		return {i for i in self.types if not self.refcnt.get(i)}

class AST(SlotsInit):
	code: ASTCodeNode
	scope: str

	def __repr__(self):
		return f"<{self.__typename__} '{self.scope}': {self.code}>"

	def __str__(self):
		return f"{self.code}"

	def validate(self) -> Namespace:
		ns = Namespace()
		self.code.validate(ns)
		return ns

	def optimize(self, level=1):
		ns = self.validate()
		self.code.optimize(ns, level)
		self.code.optimize(ns)

	@classmethod
	def build(cls, st, *, scope='') -> AST:
		code = ASTCodeNode.build(st)
		return cls(code=code, scope=scope)

# by Sdore, 2021-24
#  slang.sdore.me
