#!/usr/bin/env python3
# PySlang Abstract Definition Tree

from __future__ import annotations

from . import (
	ASTBinOpNode,
	ASTChoiceNode,
	ASTCommentNode,
	ASTIdentifierNode,
	ASTImportNode,
	ASTLiteralNode,
	ASTNode,
	ASTUnOpNode,
	ASTUnPostOpNode,
	Namespace,
)
from .. import sldef
from ..lexer import Lexer
from ..exceptions import SlException
from typing import Literal
from utils.nolog import *

SLD_DEF = os.path.join(os.path.dirname(sldef.__file__), 'sld.sldef')

class ADTNode(ASTNode):
	pass

class ADTSimpleNode(ADTNode):
	def analyze(self, ns):
		pass

class ADTChoiceNode(ADTNode, ASTChoiceNode):
	pass

class ADTCodeNode(ADTNode):
	import_: list[ASTImportNode] | None
	_codesep: list[Literal[';', '\n']]
	classdef: list[ADTClassdefNode] | None
	comment: list[ASTCommentNode] | None

	def __str__(self):
		return f"{str().join(f'{i};\n' for i in (self.import_ or ()))}{'\n'*bool(self.import_)}{S('\n\n').join(self.classdef or ())}"

	def analyze(self, ns):
		for i in (self.import_ or ()):
			i.analyze(ns)

		for i in (self.classdef or ()):
			i.analyze(ns)

class ADTClassdefNode(ADTNode):
	class_: 'class'
	identifier: list[ADTIdentifierNode]
	_lt: list['<'] | None
	block: ADTBlockNode

	def __str__(self):
		return f"{self.class_} {S(' < ').join(self.identifier)}{self.block}"

	def analyze(self, ns):
		for i in self.identifier:
			i.analyze(ns)

		t = BuiltinTypedef(self.classname)
		ns.define(t)

		blockns = ns.derive(self)
		blockns.T = t
		self.block.analyze(blockns)

	@property
	def classname(self):
		return self.identifier[0]

class ADTIdentifierNode(ADTSimpleNode):
	identifier: ASTIdentifierNode

	def __str__(self):
		return f"{self.identifier}"

class ADTBlockNode(ADTNode):
	lbrace: '{'
	_codesep: list[Literal['\n', ';']] | None
	comment: list[ASTCommentNode] | None
	definition: list[ADTDefinitionNode] | None
	rbrace: '}'

	def __str__(self):
		return f" {self.lbrace}\n{S(';\n').join(self.definition or ()).indent()}{';'*bool(self.definition)}\n{self.rbrace}"

	def analyze(self, ns):
		for i in (self.definition or ()):
			i.analyze(ns)

class ADTDefinitionNode(ADTChoiceNode):
	typename: ADTTypenameNode | None
	castable: ADTCastableNode | None
	iterable: ADTIterableNode | None
	operator: ADTOperatorNode | None
	methoddef: ADTMethoddefNode | None
	fielddef: ADTFielddefNode | None

class ADTTypenameNode(ADTNode):
	typename: 'typename'
	identifier: ADTIdentifierNode
	lbrk: Literal['['] | None
	integer: int | None
	rbrk: Literal[']'] | None

	def __str__(self):
		return f"{self.typename} {self.identifier}{self.lbrk or ''}{self.integer if (self.integer is not None) else ''}{self.rbrk or ''}"

	def analyze(self, ns):
		self.identifier.analyze(ns)

class ADTCastableNode(ADTNode):
	castable: 'castable'
	to: 'to'
	identifier: ADTIdentifierNode

	def __str__(self):
		return f"{self.castable} {self.to} {self.identifier}"

	def analyze(self, ns):
		self.identifier.analyze(ns)

		ns.T.castable.add(self.identifier)

class ADTIterableNode(ADTNode):
	iterable: 'iterable'
	type_: ADTTypeNode

	def __str__(self):
		return f"{self.iterable} {self.type_}"

	def analyze(self, ns):
		self.type_.analyze(ns)

		ns.T.iterable = self.type_

class ADTTypeNode(ADTNode):
	identifier: ADTIdentifierNode
	lbrk: Literal['['] | None
	integer: int | None
	rbrk: Literal[']'] | None

	def __str__(self):
		return f"{self.identifier}{self.lbrk or ''}{self.integer if (self.integer is not None) else ''}{self.rbrk or ''}"

	def analyze(self, ns):
		self.identifier.analyze(ns)

class ADTPreOpNode(ADTSimpleNode):
	unop: ASTUnOpNode

	def __str__(self):
		return f"{self.unop}"

class ADTBinOpNode(ADTNode):
	binop: ASTBinOpNode
	type_: ADTTypeNode

	def __str__(self):
		return f" {self.binop}{self.type_}"

	def analyze(self, ns):
		self.binop.analyze(ns)
		self.type_.analyze(ns)

class ADTPostOpNode(ADTChoiceNode):
	binop: ADTBinOpNode | None
	unpostop: ASTUnPostOpNode | None
	itemget: ADTItemgetNode | None
	funccall: ADTFunccallNode | None

class ADTOperatorNode(ADTNode):
	typeref: ADTTyperefNode
	preop: ADTPreOpNode | None
	operator: 'operator'
	postop: ADTPostOpNode | None

	def __str__(self):
		return f"{self.typeref} {self.preop or ''}{' '*bool(self.preop and str(self.preop)[-1].isalnum())}{self.operator}{self.postop or ''}"

	def analyze(self, ns):
		self.typeref.analyze(ns)
		if (self.preop): self.preop.analyze(ns)
		if (self.postop): self.postop.analyze(ns)

		if (self.preop): ns.T.unop[self.preop.unop] = self.typeref
		if (self.postop):
			if (self.postop.binop): ns.T.binop[self.postop.binop.binop][self.postop.value.type_] = self.typeref
			elif (self.postop.unpostop): ns.T.unop[self.postop.unpostop] = self.typeref
			elif (self.postop.itemget): ns.T.itemget[self.postop.itemget.type_] = self.typeref
			elif (self.postop.funccall): ns.T.call[tuple(self.postop.funccall.argdef)] = self.typeref
			else: raise WTFException(self.postop)

class ADTTyperefNode(ADTNode):
	decimal: int | None
	dot: Literal['.'] | None
	type_: ADTTypeNode

	def __str__(self):
		return f"{self.decimal if (self.decimal is not None) else ''}{self.dot or ''}{self.type_}"

	def analyze(self, ns):
		self.type_.analyze(ns)

class ADTItemgetNode(ADTNode):
	lbrk: '['
	type_: ADTTypeNode
	rbrk: ']'

	def __str__(self):
		return f" {self.lbrk}{self.type_}{self.rbrk}"

	def analyze(self, ns):
		self.type_.analyze(ns)

class ADTFunccallNode(ADTNode):
	lparen: '('
	argdef: list[ADTArgdefNode] | None
	_comma: list[','] | None
	rparen: ')'

	def __str__(self):
		return f"{self.lparen}{S(', ').join(self.argdef or ())}{self.rparen}"

	def analyze(self, ns):
		funcns = ns.derive(self)
		for i in (self.argdef or ()):
			i.analyze(funcns)

class ADTArgdefNode(ADTNode):
	star: Literal['*'] | None
	type_: ADTTypeNode | None
	slash: Literal['/'] | None
	identifier: ADTIdentifierNode | None
	lbrk: Literal['['] | None
	integer: int | None
	rbrk: Literal[']'] | None
	mod: Literal['**', '?', '+', '*', '='] | None
	literal: ASTLiteralNode | None

	def __str__(self):
		return f"{self.star or ''}{self.slash or ''}{self.type_}{f' {self.identifier}' if (self.identifier) else ''}{self.lbrk or ''}{self.integer if (self.integer is not None) else ''}{self.rbrk or ''}{self.mod or ''}{self.literal or ''}"

	def analyze(self, ns):
		if (self.type_ is not None): self.type_.analyze(ns)
		if (self.identifier is not None): self.identifier.analyze(ns)
		if (self.literal is not None): self.literal.analyze(ns)

class ADTMethoddefNode(ADTNode):
	typeref: ADTTyperefNode
	identifier: ADTIdentifierNode
	funccall: ADTFunccallNode

	def __str__(self):
		return f"{self.typeref} {self.identifier}{self.funccall}"

	def analyze(self, ns):
		self.typeref.analyze(ns)
		self.identifier.analyze(ns)
		self.funccall.analyze(ns)

class ADTFielddefNode(ADTNode):
	typeref: ADTTyperefNode
	identifier: list[ADTIdentifierNode]
	_comma: list[','] | None

	def __str__(self):
		return f"{self.typeref} {S(', ').join(self.identifier)}"

class ADT(SlotsInit):
	code: ADTCodeNode
	scope: str

	def __repr__(self):
		return f"<{self.__typename__} '{self.scope}': {self.code}>"

	def __str__(self):
		return f"{self.code}"

	def analyze(self) -> Namespace:
		ns = DefNamespace()
		self.code.analyze(ns)
		return ns

	def validate(self, ns: Namespace = None):
		if (ns is None): ns = self.analyze()
		self.code.validate(ns)

	@classmethod
	def build(cls, st, *, scope='') -> AST:
		code = ADTCodeNode.build(st)
		return cls(code=code, scope=scope)

class Typedef(Slots):
	name: str
	unop: dict
	binop: lambda: Sdict(dict)
	itemget: dict
	call: dict

	@dispatch
	def __init__(self, name: str):
		self.name = name

	@dispatch
	def __init__(self, name: ADTIdentifierNode | ASTIdentifierNode):
		self.__init__(name.identifier)

class BuiltinTypedef(Typedef):
	castable: set
	iterable: ...

	def __str__(self):
		return 'type'

class DefNamespace(Namespace):
	T: '# Typedef'

	def __init__(self, *args, T: Typedef = None, **kwargs):
		super().__init__(*args, **kwargs)
		self.T = T

	@dispatch
	def define(self, T: Typedef):
		super().define(T.name, T)

@export
def interpret_source(src, filename='<string>'):
	try:
		print(f"Source: {{\n{'\n'.join(f'\033[1;2;40m{ii:>6} \033[m{i}' for ii, i in enumerate(S(src).indent().split('\n'), 1))}\n}}\n")

		slang = sldef.load()
		sld = sldef.load(SLD_DEF, Slang=slang)

		st = Lexer().parse_string(sld, src)
		print("Source tree:", st, sep='\n\n', end='\n\n')

		adt = ADT.build(st, scope=filename.join('""'))
		print(f"Abstract definition tree: {{\n{Sstr(adt).indent()}\n}}\n")

		ns = adt.analyze()
		print(f"Definition namespace:\n\t{ns}\n")

		adt.validate(ns)

		print("Interpreted.\n")
	except SlException as ex:
		if (not ex.srclines): ex.srclines = src.split('\n')
		raise

	return ns

@apmain
@aparg('file', metavar='<file.sld>', type=argparse.FileType('r'))
def main(cargs):
	filename = cargs.file.name

	with cargs.file as f:
		src = f.read()

	try: code = interpret_source(src, filename=filename)
	except SlException as ex: sys.exit(str(ex))

if (__name__ == '__main__'): exit(main())

# by Sdore, 2024
# slang.sdore.me
