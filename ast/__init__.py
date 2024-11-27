#!/usr/bin/env python3
# PySlang Abstract Syntax Tree

from __future__ import annotations

from .. import sldef
from ..exceptions import SlValidationError
from ..lexer import Expr, Token
from typing import Literal
from utils.nolog import *

sldef = sldef.load()

class ASTNodeMeta(ABCSlotsInitMeta):
	class OptimizerMeta(SlotsTypecheckMeta):
		def __new__(metacls, name, bases, classdict, *, optimizers: tuple[list] = ()):
			classdict.setdefault('level', 1)

			for i in bases:
				optimizers += i.__optimizers__

			cls = super().__new__(metacls, name, bases, (classdict | {'__optimizers__': (classdict.get('__optimizers__', ()) + optimizers)}))

			if (bases):
				for i in cls.__optimizers__:
					i.append(cls)

			return cls

		def __and__(self, other):
			return self.__class__(f"({self.__name__.removesuffix('Optimizer').strip('()')} & {other.__class__.__name__})Optimizer", (), {
				'__annotations__': dict(set(get_annotations(self).items()) & set(get_annotations(other).items())),
			}, optimizers=(self.__optimizers__ + (other.optimizers,)))

	def __new__(metacls, name, bases, classdict):
		return super().__new__(metacls, name, bases, (classdict | {'optimizers': []}))

	def __and__(self, other):
		return self.OptimizerMeta(f"({self.__class__.__name__} & {other.__class__.__name__})Optimizer", (), {
			'__annotations__': dict(set(get_annotations(self).items()) & set(get_annotations(other).items())),
		}, optimizers=(self.optimizers, other.optimizers))

class ASTNode(metaclass=ASTNodeMeta):
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

	@abc.abstractmethod
	def analyze(self, ns: Namespace) -> None:
		pass

	def validate(self, ns: Namespace) -> None:
		for i in self.__annotations__:
			try: v = getattr(self, i)
			except AttributeError: continue
			else:
				for j in (v if (isiterablenostr(v)) else (v,)):
					try: j.validate
					except AttributeError: pass
					else: j.validate(ns)

	def optimize(self, ns: Namespace, level: int = 0) -> ASTNode | None:
		cls = self.__class__

		for i in (cls.optimizers + cls.__subclasses__()):
			if (i.level > level): continue
			if (not isinstance((self := (i.optimize(self, ns) or self)), cls)): return self

		return self

	@classmethod
	def build(cls, t: Token):
		#print(f"> {cls.__name__}\n")
		res = dict()

		annotations = inspect.get_annotations(cls, eval_str=True)

		for i in t.tokens:
			#print(i, end='\n\n')

			name = (last(i.name.rpartition('.')) if (i.name.replace('.', '').isidentifier()) else i.name)
			pattern = None

			key = name
			if (keyword.iskeyword(key) or keyword.issoftkeyword(key)): key += '_'
			if (not isinstance(i, Expr) and i.typename == 'joint' and key in annotations): key += '_'

			try: a = annotations[key]
			except KeyError:
				if (not isinstance(i, Expr) and i.typename == 'pattern'): pattern = name = repr(name.removeprefix('/').removesuffix('/'))
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
	def analyze(self, ns):
		pass

class ASTChoiceNode(ASTNode):
	def __str__(self):
		try: return f"{self.value}"
		except StopIteration: return super().__str__()

	def analyze(self, ns):
		self.value.analyze(ns)

	def validate(self, ns: Namespace) -> None:
		v = self.value
		for i in (v if (isiterablenostr(v)) else (v,)):
			try: i.validate
			except AttributeError: pass
			else: i.validate(ns)

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

	def analyze(self, ns):
		for i in (self.statement or ()):
			i.analyze(ns)

	def optimize(self, ns, level=0):
		if (self.statement): self.statement = list(filter(None, (i.optimize(ns, level) for i in self.statement)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.statement and not self.comment): return None
		else: return self

class ASTClassCodeNode(ASTNode):
	classstatement: list[ASTClassStatementNode] | None
	comment: list[ASTCommentNode] | None
	_codesep: list[Literal['\n', ';']] | None

	def __str__(self):
		return S('\n').join(self.classstatement or ())

	def analyze(self, ns):
		for i in (self.classstatement or ()):
			i.analyze(ns)

	def optimize(self, ns, level=0):
		if (self.classstatement): self.classstatement = list(filter(None, (i.optimize(ns, level) for i in self.classstatement)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.classstatement and not self.comment): return None
		else: return self

class ASTClassdefCodeNode(ASTNode):
	classdefstatement: list[ASTClassdefStatementNode] | None
	comment: list[ASTCommentNode] | None
	_codesep: list['\n'] | None

	def __str__(self):
		return S('\n').join(self.classdefstatement or ())

	def analyze(self, ns):
		for i in (self.classdefstatement or ()):
			i.analyze(ns)

	def optimize(self, ns, level=0):
		if (self.classdefstatement): self.classdefstatement = list(filter(None, (i.optimize(ns, level) for i in self.classdefstatement)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.classdefstatement and not self.comment): return None
		else: return self

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

	def analyze(self, ns):
		if (self.code): self.code.analyze(ns)
		for i in (self.statement or ()):
			i.analyze(ns)

	def validate(self, ns):
		if (unused := ns.unused()): raise SlValidationError.from_node(self, "Unused variables", unused)

	def optimize(self, ns, level=0):
		if (self.code): self.code = self.code.optimize(ns, level)
		if (self.statement): self.statement = list(filter(None, (i.optimize(ns, level) for i in self.statement)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.code and not self.statement and not self.comment): return None
		else: return self

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

	def analyze(self, ns):
		if (self.classcode): self.classcode.analyze(ns)
		for i in (self.classstatement or ()):
			i.analyze(ns)

	def validate(self, ns):
		if (unused := ns.unused()): raise SlValidationError.from_node(self, "Unused variables", unused)

	def optimize(self, ns, level=0):
		if (self.classcode): self.classcode = self.classcode.optimize(ns, level)
		if (self.classstatement): self.classstatement = list(filter(None, (i.optimize(ns, level) for i in self.classstatement)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.classcode and not self.classstatement and not self.comment): return None
		else: return self

class ASTClassdefBlockNode(ASTNode):
	lbrace: '{'
	_codesep: list['\n'] | None
	classdefcode: ASTClassdefCodeNode
	rbrace: '}'

	def __str__(self):
		return Sstr(self.classdefcode).indent().join((f" {self.lbrace}\n", f"\n{self.rbrace}"))

	def analyze(self, ns):
		if (self.classdefcode): self.classdefcode.analyze(ns)
		for i in (self.classdefstatement or ()):
			i.analyze(ns)

	def optimize(self, ns, level=0):
		if (self.classdefcode): self.classdefcode = self.classdefcode.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.classdefcode): return None
		else: return self


## Primitive

class ASTBinOpExprNode(ASTNode):
	expr: list[ASTExprNode]
	binop: ASTBinOpNode

	def __str__(self):
		return f"{self.lhs} {self.binop} {self.rhs}"

	def analyze(self, ns):
		self.lhs.analyze(ns)
		self.rhs.analyze(ns)

	def optimize(self, ns, level=0):
		self.lhs = self.lhs.optimize(ns, level)
		self.rhs = self.rhs.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and (not self.lhs or not self.rhs)): return None
		else: return self

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

	@property
	def op(self):
		return self.binop

class ASTUnOpExprNode(ASTNode):
	lhs = rhs = None

	expr: ASTExprNode

class ASTUnPreOpExprNode(ASTUnOpExprNode):
	unop: ASTUnOpNode
	expr: ASTExprNode

	def __str__(self):
		return f"{self.unop}{' '*(str(self.unop)[-1].isalnum())}{self.expr}"

	def analyze(self, ns):
		self.expr.analyze(ns)

	def optimize(self, ns, level=0):
		self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.expr): return None
		else: return self

	@property
	def rhs(self):
		return self.expr

	@rhs.setter
	def rhs(self, x):
		self.expr = x

	@property
	def op(self):
		return self.unop

class ASTUnPostOpExprNode(ASTUnOpExprNode):
	expr: ASTExprNode
	unpostop: ASTUnPostOpNode

	def __str__(self):
		return f"{self.expr}{' '*(str(self.unpostop)[0].isalnum())}{self.unpostop}"

	def analyze(self, ns):
		self.expr.analyze(ns)

	def optimize(self, ns, level=0):
		self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.expr): return None
		else: return self

	@property
	def lhs(self):
		return self.expr

	@lhs.setter
	def lhs(self, x):
		self.expr = x

	@property
	def op(self):
		return self.unpostop

class ASTUnOpExprNode(ASTChoiceNode):
	unpreopexpr: ASTUnPreOpExprNode | None
	unpostopexpr: ASTUnPostOpExprNode | None

	def analyze(self, ns):
		self.value.analyze(ns)

	def optimize(self, ns, level=0):
		self.value = self.value.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.value): return None
		else: return self

class ASTItemgetNode(ASTNode):
	expr: list[ASTExprNode]
	lbrk: '['
	rbrk: ']'

	def __str__(self):
		return f"{self.value}{self.lbrk}{self.index}{self.rbrk}"

	def analyze(self, ns):
		for i in self.expr:
			i.analyze(ns)

	def optimize(self, ns, level=0):
		self.value = self.value.optimize(ns, level)
		self.index = self.index.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and (not self.value or not self.index)): return None
		else: return self

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

	def analyze(self, ns):
		self.value.analyze(ns)
		if (self.varname and self.varname.identifier): ns.ref(self.varname)

class ASTExprNode(ASTNode):
	_lparen: Literal['('] | None
	expr: ASTExprNode | None
	_rparen: Literal[')'] | None
	value: ASTValueNode | None

	def __str__(self):
		return f"{f'({self.expr})' if (self.expr is not None) else self.value}"

	def analyze(self, ns):
		if (self.value): self.value.analyze(ns)
		else: self.expr.analyze(ns)

	def optimize(self, ns, level=0):
		return (self.expr or self.value).optimize(ns, level)


## Non-final

class ASTTypeNode(ASTNode):
	modifier: list[Literal[sldef.definitions.modifier.format.literals]] | None
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{S(', ').join(self.modifier)+' ' if (self.modifier) else ''}{self.identifier}"

	def __eq__(self, other):
		return (isinstance(other, self.__class__) and self.identifier == other.identifier and self.modifier == other.modifier)

	def analyze(self, ns):
		self.identifier.analyze(ns)
		ns.ref(self.identifier)

	def optimize(self, ns, level=0):
		self.identifier = self.identifier.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.identifier): return None
		else: return self

class ASTCatchClauseNode(ASTNode):
	catch: 'catch'
	type_: ASTTypeNode | None
	identifier: ASTIdentifierNode | None
	block: ASTBlockNode

	def __str__(self):
		return f"{self.catch}{f' {self.type_}' if (self.type_ is not None) else ''}{f' {self.identifier}' if (self.identifier is not None) else ''}{self.block}"

	def analyze(self, ns):
		self.type_.analyze(ns)
		self.identifier.analyze(ns)

		blockns = ns.derive(self)
		blockns.define(self.identifier, self.type_)
		self.block.analyze(blockns)

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

	def analyze(self, ns):
		blockns = ns.derive(self)
		self.block.analyze(blockns)

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

	def analyze(self, ns, *, type_):
		self.identifier.analyze(ns)
		if (self.expr): self.expr.analyze(ns)
		if (self.funccallargs): self.funccallargs.analyze(ns)

		ns.define(self.identifier, type_, value=self.expr)

	def optimize(self, ns, level=0):
		if (level == 0 and not ns.refs(self.identifier)): return None

		self.identifier = self.identifier.optimize(ns, level)
		if (self.expr): self.expr = self.expr.optimize(ns, level)
		if (self.funccallargs): self.funccallargs = self.funccallargs.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.identifier): return None
		else: return self

class ASTArgdefNode(ASTNode):
	special: Literal['/', '*'] | None
	type_: ASTTypeNode
	identifier: ASTIdentifierNode
	lbrk: Literal['['] | None
	integer: int | None
	rbrk: Literal[']'] | None
	mode: Literal['**', '?', '+', '*', '='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.special or ''}{self.type_} {self.identifier}{f'{self.lbrk}{self.integer}{self.rbrk}' if (self.integer is not None) else ''}{self.mode or ''}{self.expr or ''}"

	def analyze(self, ns):
		self.type_.analyze(ns)
		self.identifier.analyze(ns)
		if (self.expr): self.expr.analyze(ns)

		ns.define(self.identifier, self.type_, value=self.expr)

class ASTClassArgdefNode(ASTNode):
	type_: ASTTypeNode
	classvarname: ASTClassVarnameNode
	mode: Literal['**', '?', '+', '*', '='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.type_} {self.classvarname}{self.mode or ''}{self.expr or ''}"

	def analyze(self, ns):
		self.type_.analyze(ns)
		self.classvarname.analyze(ns)
		if (self.expr): self.expr.analyze(ns)

		ns.define(self.classvarname, self.type_, value=self.expr)

class ASTCallArgNode(ASTNode):
	star: Literal['*'] | None
	expr: ASTExprNode

	def __str__(self):
		return f"{self.star or ''}{self.expr}"

	def analyze(self, ns):
		self.expr.analyze(ns)

	def optimize(self, ns, level=0):
		self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.expr): return None
		else: return self

class ASTCallArgsNode(ASTNode):
	callarg: list[ASTCallArgNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{S(', ').join(self.callarg)}"

	def analyze(self, ns):
		for i in self.callarg:
			i.analyze(ns)

	def optimize(self, ns, level=0):
		self.callarg = list(filter(None, (i.optimize(ns, level) for i in self.callarg)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.callarg): return None
		else: return self

class ASTCallKwargNode(ASTNode):
	identifier: ASTIdentifierNode
	eq: Literal['=', ':'] | None
	expr: ASTExprNode | None
	star: Literal['**'] | None

	def __str__(self):
		return f"{not self.expr and self.eq or ''}{self.identifier}{self.eq or ''}{' '*(self.eq == ':')}{self.star or ''}{self.expr or ''}"

	def analyze(self, ns):
		self.identifier.analyze(ns)
		if (self.expr):
			self.expr.analyze(ns)

	def optimize(self, ns, level=0):
		self.identifier = self.identifier.optimize(ns, level)
		if (self.expr): self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.identifier and not self.expr): return None
		else: return self

class ASTCallKwargsNode(ASTNode):
	callkwarg: list[ASTCallKwargNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{S(', ').join(self.callkwarg)}"

	def analyze(self, ns):
		for i in self.callkwarg:
			i.analyze(ns)

	def optimize(self, ns, level=0):
		self.callkwarg = list(filter(None, (i.optimize(ns, level) for i in self.callkwarg)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.callkwarg): return None
		else: return self

class ASTFunccallArgsNode(ASTNode):
	lparen: '('
	callargs: list[ASTCallArgsNode] | None
	_comma: list[','] | None
	callkwargs: list[ASTCallKwargsNode] | None
	rparen: ')'

	def __str__(self):
		return f"{self.lparen}{S(', ').join((*(self.callargs or ()), *(self.callkwargs or ())))}{self.rparen}"

	def analyze(self, ns):
		for i in (self.callargs or ()):
			i.analyze(ns)

		for i in (self.callkwargs or ()):
			i.analyze(ns)

	def optimize(self, ns, level=0):
		if (self.callargs): self.callargs = list(filter(None, (i.optimize(ns, level) for i in self.callargs)))
		if (self.callkwargs): self.callkwargs = list(filter(None, (i.optimize(ns, level) for i in self.callkwargs)))

		return super().optimize(ns, level)

class ASTAttrSelfOpNode(ASTSimpleNode):
	op: Literal['@.', '@', '.', ':']

	def __str__(self):
		return f"{self.op}"

class ASTAttrOpNode(ASTSimpleNode, ASTChoiceNode):
	op: Literal['->'] | None
	attrselfop: ASTAttrSelfOpNode | None

class ASTAttrgetNode(ASTNode):
	expr: ASTExprNode
	attrop: ASTAttrOpNode
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.expr}{self.attrop}{self.identifier}"

	def analyze(self, ns):
		self.expr.analyze(ns)
		self.identifier.analyze(ns)

class ASTClassAttrgetNode(ASTNode):
	attrselfop: ASTAttrSelfOpNode | None
	expr: ASTExprNode | None
	attrop: ASTAttrOpNode | None
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.attrselfop or ''}{self.expr or ''}{self.attrop or ''}{self.identifier}"

	def analyze(self, ns):
		self.expr.analyze(ns)
		self.identifier.analyze(ns)


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

	def analyze(self, ns):
		self.type_.analyze(ns)
		self.identifier.analyze(ns)

		funcns = ns.derive(self)
		for i in (self.argdef or ()):
			i.analyze(funcns)

		ns.define(self.identifier, self.type_, value=self.expr)

		if (self.block): self.block.analyze(funcns)
		if (self.expr): self.expr.analyze(funcns)

		ns.assign(self.identifier, self.expr)

	def optimize(self, ns, level=0):
		if (self.type_): self.type_ = self.type_.optimize(ns, level)
		self.identifier = self.identifier.optimize(ns, level)

		funcns = ns.derive(self)
		if (self.argdef): self.argdef = list(filter(None, (i.optimize(funcns, level) for i in self.argdef)))
		if (self.block): self.block = self.block.optimize(funcns, level)
		if (self.expr): self.expr = self.expr.optimize(funcns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.identifier): return None
		else: return self

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

	def analyze(self, ns):
		self.type_.analyze(ns)
		self.identifier.analyze(ns)

		funcns = ns.derive(self)
		for i in (self.argdef or ()):
			i.analyze(funcns)

		ns.define(self.identifier, self.type_, value=self.expr)

		if (self.classblock): self.classblock.analyze(funcns)
		if (self.expr): self.expr.analyze(funcns)

		ns.assign(self.identifier, self.expr)

	def optimize(self, ns, level=0):
		if (self.type_): self.type_ = self.type_.optimize(ns, level)
		self.identifier = self.identifier.optimize(ns, level)

		funcns = ns.derive(self)
		if (self.argdef): self.argdef = list(filter(None, (i.optimize(funcns, level) for i in self.argdef)))
		if (self.classblock): self.block = self.classblock.optimize(funcns, level)
		if (self.expr): self.expr = self.expr.optimize(funcns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.identifier): return None
		else: return self

class ASTKeywordExprNode(ASTChoiceNode):
	return_: ASTReturnNode | None
	raise_: ASTRaiseNode | None
	throw: ASTThrowNode | None
	resume: ASTResumeNode | None
	break_: ASTBreakNode | None
	continue_: ASTContinueNode | None
	fallthrough: ASTFallthroughNode | None
	import_: ASTImportNode | None
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

	def analyze(self, ns):
		for i in self.expr:
			i.analyze(ns)

		for ii, i in enumerate(self.block):
			blockns = ns.derive(self, ii)
			i.analyze(blockns)

	def optimize(self, ns, level=0):
		self.expr = list(filter(None, (i.optimize(ns, level) for i in self.expr)))

		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			self.block[ii] = i.optimize(loopns)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.expr): return None
		else: return self

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

	def analyze(self, ns):
		self.type_.analyze(ns)
		self.identifier.analyze(ns)
		self.expr.analyze(ns)

		ns.define(self.identifier, self.type_)

		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			i.analyze(loopns)

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

	def analyze(self, ns):
		self.expr.analyze(ns)

		for ii, i in enumerate(self.block):
			blockns = ns.derive(self, ii)
			i.analyze(blockns)

	def optimize(self, ns, level=0):
		self.expr = self.expr.optimize(ns, level)

		for ii, i in enumerate(self.block):
			loopns = ns.derive(self, ii)
			self.block[ii] = i.optimize(loopns)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.expr): return None
		else: return self

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

	def analyze(self, ns):
		for ii, i in enumerate(self.block):
			blockns = ns.derive(self, ii)
			i.analyze(blockns)

		for i in self.catchclause:
			i.analyze(ns)

		self.finallyclause.analyze(ns)

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

	def analyze(self, ns):
		self.identifier.analyze(ns)

		classns = ns.derive(self)
		self.classdefblock.analyze(classns)

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

	def analyze(self, ns):
		self.type_.analyze(ns)
		for i in self.vardefassignment:
			i.analyze(ns, type_=self.type_)

	def optimize(self, ns, level=0):
		self.type_ = self.type_.optimize(ns, level)
		self.vardefassignment = list(filter(None, (i.optimize(ns, level) for i in self.vardefassignment)))

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.vardefassignment): return None
		else: return self

class ASTAssignmentNode(ASTNode):
	varname: list[ASTVarnameNode]
	_comma: list[','] | None
	assignment: Literal['='] | str
	expr: ASTExprNode

	def __str__(self):
		return f"{S(', ').join(self.varname)} {self.assignment} {self.expr}"

	def analyze(self, ns):
		for i in self.varname:
			i.analyze(ns)
		self.expr.analyze(ns)

		if (len(self.varname) == 1): ns.assign(only(self.varname), self.expr) # TODO: unpack

	def optimize(self, ns, level=0):
		self.varname = list(filter(None, (i.optimize(ns, level) for i in self.varname)))
		self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and (not self.varname or not self.expr)): return None
		else: return self

class ASTClassAssignmentNode(ASTNode):
	classvarname: list[ASTClassVarnameNode]
	_comma: list[','] | None
	assignment: Literal['='] | str
	expr: ASTExprNode

	def __str__(self):
		return f"{S(', ').join(self.classvarname)} {self.assignment} {self.expr}"

	def analyze(self, ns):
		for i in self.classvarname:
			i.analyze(ns)
		self.expr.analyze(ns)

		if (len(self.classvarname) == 1): ns.assign(only(self.classvarname), self.expr) # TODO: unpack

	def optimize(self, ns, level=0):
		self.classvarname = list(filter(None, (i.optimize(ns, level) for i in self.classvarname)))
		self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and (not self.classvarname or not self.expr)): return None
		else: return self

class ASTFunccallNode(ASTNode):
	expr: ASTExprNode
	funccallargs: ASTFunccallArgsNode

	def __str__(self):
		return f"{self.expr}{self.funccallargs}"

	def analyze(self, ns):
		self.expr.analyze(ns)
		self.funccallargs.analyze(ns)

	def optimize(self, ns, level=0):
		self.expr = self.expr.optimize(ns, level)
		self.funccallargs = self.funccallargs.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and (not self.expr or not self.funccallargs)): return None
		else: return self

class ASTKeyworddefNode(ASTNode):
	defkeyword: ASTDefkeywordNode
	block: ASTBlockNode

	def __str__(self):
		return f"{self.defkeyword}{self.block}"

	def analyze(self, ns):
		blockns = ns.derive(self)
		self.block.analyze(blockns)

	def optimize(self, ns, level=0):
		blockns = ns.derive(self)
		self.block = self.block.optimize(blockns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.block): return None
		else: return self

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

	def analyze(self, ns):
		blockns = ns.derive(self)
		self.classdefblock.analyze(blockns)

	def optimize(self, ns, level=0):
		blockns = ns.derive(self)
		self.classdefblock = self.classdefblock.optimize(blockns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.classdefblock): return None
		else: return self


## Keywords

class ASTReturnNode(ASTNode):
	return_: 'return'
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.return_}{f' {self.expr}' if (self.expr is not None) else ''}"

	def analyze(self, ns):
		self.expr.analyze(ns)

	def optimize(self, ns, level=0):
		self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.expr): return None
		else: return self

class ASTRaiseNode(ASTSimpleNode):
	raise_: 'raise'

	def __str__(self):
		return f"{self.raise_}"

class ASTThrowNode(ASTNode):
	throw: 'throw'
	expr: ASTExprNode

	def __str__(self):
		return f"{self.throw} {self.expr}"

	def analyze(self, ns):
		self.expr.analyze(ns)

	def optimize(self, ns, level=0):
		self.expr = self.expr.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.expr): return None
		else: return self

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

class ASTImportNode(ASTNode):
	import_: 'import'
	import__: list[str]
	identifier: list[ASTIdentifierNode]
	colon: Literal[':'] | None
	_comma: list[','] | None
	star: Literal['*'] | None

	def __str__(self):
		return f"""{self.import_} {str().join(self.import__)}{self.package}{self.colon*bool(self.names or self.star)}{f" {S(', ').join(self.names)}" if (self.names) else ''}{self.star or ''}"""

	def analyze(self, ns):
		for i in self.identifier:
			i.analyze(ns)

	@property
	def package(self):
		return self.identifier[0]

	@property
	def names(self):
		return self.identifier[1:]

class ASTDeleteNode(ASTNode):
	delete: 'delete'
	varname: ASTVarnameNode

	def __str__(self):
		return f"{self.delete} {self.varname}"

	def analyze(self, ns):
		self.ns.delete(self.varname)

	def optimize(self, ns, level=0):
		self.varname = self.varname.optimize(ns, level)

		cls, self = self.__class__, super().optimize(ns, level)
		if (isinstance(self, cls) and not self.varname): return None
		else: return self

class ASTDefkeywordNode(ASTSimpleNode):
	defkeyword: Literal['main', 'exit']

	def __str__(self):
		return f"{self.defkeyword}"

class ASTClassDefkeywordNode(ASTSimpleNode):
	classdefkeyword: Literal['init', 'destroy', 'constr', 'property', 'repr', 'eq', 'ne', *(i+j for i in 'lg' for j in 'te')]

	def __str__(self):
		return f"{self.classdefkeyword}"

class ASTReservedNode(ASTSimpleNode):
	reserved: Literal[sldef.definitions.reserved.format.literals]

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

	def analyze(self, ns):
		return super(ASTChoiceNode, self).analyze(ns)

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

class ASTUnOpNode(ASTSimpleNode, ASTChoiceNode):
	unchop: Literal[sldef.definitions.unchop.format.literals] | None
	undchop: Literal[sldef.definitions.undchop.format.literals] | None
	unkwop: Literal[sldef.definitions.unkwop.format.literals] | None
	unmathop: str | None

class ASTUnPostOpNode(ASTSimpleNode, ASTChoiceNode):
	unchpostop: Literal[sldef.definitions.unchpostop.format.literals] | None
	undchpostop: Literal[sldef.definitions.undchpostop.format.literals] | None

class ASTBinOpNode(ASTSimpleNode, ASTChoiceNode):
	binchop: Literal[sldef.definitions.binchop.format.literals] | None
	bindchop: Literal[sldef.definitions.bindchop.format.literals] | None
	binkwop: Literal[sldef.definitions.binkwop.format.literals] | None
	binmathop: Literal[sldef.definitions.binmathop.format.literals] | None


## Comments

class ASTBlockCommentNode(ASTSimpleNode):
	blockcomment: str

	def __str__(self):
		return f"{self.blockcomment}"

class ASTLineCommentNode(ASTSimpleNode):
	linecomment: str

	def __str__(self):
		return f"{self.linecomment}"

class ASTCommentNode(ASTSimpleNode, ASTChoiceNode):
	blockcomment: ASTBlockCommentNode | None
	linecomment: ASTLineCommentNode | None

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
		return '\n\t '.join(f"<{self.refcnt.get(i, '?')}> {self.types.get(i, '*')} {i}{f' = {v}' if (v := self.values.get(i)) else ''}" for i in S((*self.types, *self.values, *self.refcnt)).uniquize()).join('{}')

	def __contains__(self, x):
		try: self.value(x)
		except KeyError: return False
		else: return True

	@cachedfunction
	def derive(self, node: ASTNode, index: int = 0, /):
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
	def __define(self, name: ASTVarnameNode, type_: ASTTypeNode, value=None):
		self.__define(name.value, type_, value=value)

	@dispatch
	def __define(self, name: ASTIdentifierNode, type_: ASTTypeNode, value=None):
		self.__define(name.identifier, type_, value=value)

	@dispatch
	def __define(self, name: str, type_, value):
		self.__define(name, type_)
		self.values[name] = value

	@dispatch
	def __define(self, name: str, type_):
		if ((t := self.types.get(name, type_)) != type_): raise ValueError(name, "redefined", t, type_)
		self.types[name] = type_
		self.refcnt[name] = 0

	define = __define

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

	def analyze(self) -> Namespace:
		ns = Namespace()
		self.code.analyze(ns)
		return ns

	def validate(self, ns: Namespace = None):
		if (ns is None): ns = self.analyze()
		self.code.validate(ns)

	def optimize(self, ns: Namespace = None, level=1):
		if (ns is None): ns = self.analyze()
		self.code.optimize(ns, level)
		self.code.optimize(ns)

	@classmethod
	def build(cls, st, *, scope='') -> AST:
		code = ASTCodeNode.build(st)
		return cls(code=code, scope=scope)

from . import optimizers

# by Sdore, 2021-24
#  slang.sdore.me
