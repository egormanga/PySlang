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

class ASTChoiceNode(ASTNode):
	@property
	def value(self):
		return only(v for i in self.__annotations__ if (v := getattr(self, i)) is not None)

	def __str__(self):
		try: return f"{self.value}"
		except StopIteration: return super().__str__()

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

class ASTClassCodeNode(ASTNode):
	classstatement: list[ASTClassStatementNode] | None
	comment: list[ASTCommentNode] | None
	_codesep: list[Literal['\n', ';']] | None

	def __str__(self):
		return S('\n').join(self.classstatement or ())

class ASTClassdefCodeNode(ASTNode):
	classdefstatement: list[ASTClassdefStatementNode] | None
	comment: list[ASTCommentNode] | None
	_codesep: list['\n'] | None

	def __str__(self):
		return S('\n').join(self.classdefstatement or ())

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

class ASTClassdefBlockNode(ASTNode):
	lbrace: '{'
	_codesep: list['\n'] | None
	classdefcode: ASTClassdefCodeNode
	rbrace: '}'

	def __str__(self):
		return Sstr(self.classdefcode).indent().join((f" {self.lbrace}\n", f"\n{self.rbrace}"))


## Primitive

class ASTBinOpExprNode(ASTNode):
	expr: list[ASTExprNode]
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

class ASTUnPreOpExprNode(ASTNode):
	unop: ASTUnOpNode
	expr: ASTExprNode

	def __str__(self):
		return f"{self.unop}{' '*(str(self.unop)[-1].isalnum())}{self.expr}"

class ASTUnPostOpExprNode(ASTNode):
	expr: ASTExprNode
	unpostop: ASTUnPostOpNode

	def __str__(self):
		return f"{self.expr}{self.unpostop}"

class ASTUnOpExprNode(ASTChoiceNode):
	unpreopexpr: ASTUnPreOpExprNode | None
	unpostopexpr: ASTUnPostOpExprNode | None

class ASTItemgetNode(ASTNode):
	expr: list[ASTExprNode]
	lbrk: '['
	rbrk: ']'

	def __str__(self):
		return f"{self.value}{self.lbrk}{self.index}{self.rbrk}"

	@property
	def value(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[0]

	@property
	def index(self):
		if (len(self.expr) != 2): raise WTFException(self, self.expr)
		return self.expr[1]

class ASTValueNode(ASTChoiceNode):
	binopexpr: ASTBinOpExprNode | None
	unopexpr: ASTUnOpExprNode | None
	itemget: ASTItemgetNode | None
	funccall: ASTFunccallNode | None
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

class ASTTypeNode(ASTNode):
	modifier: list[Literal[Lexer.sldef.definitions.modifier.format.literals]] | None
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{S(', ').join(self.modifier)+' ' if (self.modifier) else ''}{self.identifier}"

class ASTCatchClauseNode(ASTNode):
	catch: 'catch'
	type_: ASTTypeNode | None
	identifier: ASTIdentifierNode | None
	block: ASTBlockNode

	def __str__(self):
		return f"{self.catch}{f' {self.type_}' if (self.type_ is not None) else ''}{f' {self.identifier}' if (self.identifier is not None) else ''}{self.block}"

class ASTFinallyClauseNode(ASTNode):
	finally_: 'finally'
	block: ASTBlockNode

	def __str__(self):
		return f"{self.finally_}{self.block}"

class ASTVardefAssignmentNode(ASTNode):
	identifier: ASTIdentifierNode
	op: Literal['='] | None
	expr: ASTExprNode | None
	funccallargs: ASTFunccallArgsNode | None

	def __str__(self):
		return f"{self.identifier}{f' {self.op} {self.expr}' if (self.expr) else ''}{self.funccallargs or ''}"

class ASTArgdefNode(ASTNode):
	special: Literal['/', '*'] | None
	type_: ASTTypeNode | None
	identifier: ASTIdentifierNode
	lbrk: Literal['['] | None
	integer: int | None
	rbrk: Literal[']'] | None
	mode: Literal['?', '+', '*', '**', '='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.special or ''}{f'{self.type_} ' if (self.type_) else ''}{self.identifier}{f'{self.lbrk}{self.integer}{self.rbrk}' if (self.integer is not None) else ''}{self.mode or ''}{self.expr or ''}"

class ASTClassArgdefNode(ASTNode):
	type_: ASTTypeNode | None
	classvarname: ASTClassVarnameNode
	mode: Literal['?', '+', '*', '**', '='] | None
	expr: ASTExprNode | None

	def __str__(self):
		return f"{f'{self.type_} ' if (self.type_) else ''}{self.classvarname}{self.mode or ''}{self.expr or ''}"

class ASTCallArgNode(ASTNode):
	star: Literal['*'] | None
	expr: ASTExprNode

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
	expr: ASTExprNode
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
	expr: ASTExprNode
	attrop: ASTAttrOpNode
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.expr}{self.attrop}{self.identifier}"

class ASTClassAttrgetNode(ASTNode):
	attrselfop: ASTAttrSelfOpNode | None
	expr: ASTExprNode | None
	attrop: ASTAttrOpNode | None
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.attrselfop or ''}{self.expr or ''}{self.attrop or ''}{self.identifier}"


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

	@property
	def elseblock(self):
		return self.block[-1]

class ASTForLoopNode(ASTNode):
	for_: 'for'
	identifier: ASTIdentifierNode
	in_: 'in'
	expr: ASTExprNode
	block: list[ASTBlockNode]
	_nl: list['\n'] | None
	else_: Literal['else'] | None

	def __str__(self):
		return f"{self.for_} {self.identifier} {self.in_} {self.expr}{self.forblock}{f' {self.else_}{self.elseblock}' if (self.else_) else ''}"

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
		return f"""{self.do_}{self.block[0]}{f" {S(' ').join(self.catchclause)}" if (self.catchclause) else ''}{f' {self.else_}{self.block[1]}' if (self.else_) else ''}{f" {self.finallyclause}" if (self.finallyclause) else ''}"""

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

class ASTAssignmentNode(ASTNode):
	varname: list[ASTVarnameNode]
	_comma: list[','] | None
	assignment: Literal['='] | str
	expr: ASTExprNode

	def __str__(self):
		return f"{S(', ').join(self.varname)} {self.assignment} {self.expr}"

class ASTClassAssignmentNode(ASTNode):
	classvarname: list[ASTClassVarnameNode]
	_comma: list[','] | None
	assignment: Literal['='] | str
	expr: ASTExprNode

	def __str__(self):
		return f"{S(', ').join(self.classvarname)} {self.assignment} {self.expr}"

class ASTFunccallArgsNode(ASTNode):
	lparen: '('
	callargs: list[ASTCallArgsNode] | None
	_comma: list[','] | None
	callkwargs: list[ASTCallKwargsNode] | None
	rparen: ')'

	def __str__(self):
		return f"{self.lparen}{S(', ').join((*(self.callargs or ()), *(self.callkwargs or ())))}{self.rparen}"

class ASTFunccallNode(ASTNode):
	expr: ASTExprNode
	funccallargs: ASTFunccallArgsNode

	def __str__(self):
		return f"{self.expr}{self.funccallargs}"

class ASTKeyworddefNode(ASTNode):
	defkeyword: ASTDefkeywordNode
	block: ASTBlockNode

	def __str__(self):
		return f"{self.defkeyword}{self.block}"

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


## Keywords

class ASTReturnNode(ASTNode):
	return_: 'return'
	expr: ASTExprNode | None

	def __str__(self):
		return f"{self.return_}{f' {self.expr}' if (self.expr is not None) else ''}"

class ASTRaiseNode(ASTNode):
	raise_: 'raise'

	def __str__(self):
		return f"{self.raise_}"

class ASTThrowNode(ASTNode):
	throw: 'throw'
	expr: ASTExprNode

	def __str__(self):
		return f"{self.throw} {self.expr}"

class ASTResumeNode(ASTNode):
	resume: 'resume'
	integer: int | None

	def __str__(self):
		return f"{self.resume}{f' {self.integer}' if (self.integer) else ''}"

class ASTBreakNode(ASTNode):
	break_: 'break'
	integer: int | None

	def __str__(self):
		return f"{self.break_}{f' {self.integer}' if (self.integer) else ''}"

class ASTContinueNode(ASTNode):
	continue_: 'continue'
	integer: int | None

	def __str__(self):
		return f"{self.continue_}{f' {self.integer}' if (self.integer) else ''}"

class ASTFallthroughNode(ASTNode):
	fallthrough: 'fallthrough'

	def __str__(self):
		return f"{self.fallthrough}"

class ASTDeleteNode(ASTNode):
	delete: 'delete'
	varname: ASTVarnameNode

	def __str__(self):
		return f"{self.delete} {self.varname}"

class ASTDefkeywordNode(ASTNode):
	defkeyword: Literal['main', 'exit']

	def __str__(self):
		return f"{self.defkeyword}"

class ASTClassDefkeywordNode(ASTNode):
	classdefkeyword: Literal['init', 'destroy', 'constr', 'property', 'repr', 'eq', 'ne', *(i+j for i in 'lg' for j in 'te')]

	def __str__(self):
		return f"{self.classdefkeyword}"

class ASTReservedNode(ASTNode):
	reserved: Literal[Lexer.sldef.definitions.reserved.format.literals]

	def __str__(self):
		return f"{self.reserved}"


## Identifiers

class ASTIdentifierNode(ASTNode):
	identifier: r'[^\W\d][\w]*'

	def __str__(self):
		return f"{self.identifier}"

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

class ASTBlockCommentNode(ASTNode):
	blockcomment: str

	def __str__(self):
		return f"{self.blockcomment}"

class ASTLineCommentNode(ASTNode):
	linecomment: str

	def __str__(self):
		return f"{self.linecomment}"

class ASTCommentNode(ASTChoiceNode):
	blockcomment: ASTBlockCommentNode | None
	linecomment: ASTLineCommentNode | None


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
