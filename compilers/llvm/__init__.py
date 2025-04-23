#!/usr/bin/env python3
# PySlang LLVM IR compiler target

from .. import *
from ...ast import *
from llvmlite import ir
from utils import *

class Instrs:
	@init_defaults
	def __init__(self, *, name, ns, filename, scpcells: indexset):
		self.name, self.ns, self.filename, self.scpcells = name, ns, filename, scpcells
		module = self.module = ir.Module(name=self.name)
		module.triple = ''
		init = self.init = ir.Function(module, ir.FunctionType(ir.VoidType(), ()), name=self.name)
		block = self.block = init.append_basic_block()
		self.builder = ir.IRBuilder(block)
		self.scope = {
			'println': ir.Function(module, ir.FunctionType(ir.VoidType(), ()), name='stdio.println'),
		}

	def compile(self):
		self.builder.ret_void()
		return (str(self.module) + '\n').encode()


	## Abstract

	@dispatch
	def add(self, x: ASTCodeNode, *, builder=None):
		for i in (x.statement or ()):
			self.add(i, builder=(builder or self.builder))

	@dispatch
	def add(self, x: ASTBlockNode, *, func):
		block = func.append_basic_block()
		builder = ir.IRBuilder(block)

		if (x.code is not None):
			self.add(x.code, builder=builder)
			assert (not x.statement)
		else:
			for i in x.statement:
				self.add(i, builder=builder)

		builder.ret_void()


	## Primitive

	@dispatch
	def load(self, x: ASTBinOpExprNode, *, builder, **kwargs):
		lhs = builder.load(self.load(x.lhs, **kwargs))
		rhs = builder.load(self.load(x.rhs, **kwargs))

		match x.binop.value:
			case '+': return builder.add(lhs, rhs)
			case _: raise WTFException(x.binop)

	@dispatch
	def load(self, x: ASTValueNode, **kwargs):
		return self.load(x.value, **kwargs)

	@dispatch
	def store(self, name, x: ASTValueNode, **kwargs):
		self.store(name, x.value, **kwargs)

	@dispatch
	def load(self, x: ASTExprNode, **kwargs):
		return self.load((x.expr or x.value), **kwargs)

	@dispatch
	def store(self, name, x: ASTExprNode, **kwargs):
		self.store(name, (x.expr or x.value), **kwargs)


	## Non-final

	@dispatch
	def add(self, x: ASTVardefAssignmentNode, *, builder, type_):
		assert (not x.funccallargs) # TODO

		match type_.name:
			case 'i8' | 'u8': type_ = ir.IntType(8)
			case 'i16' | 'u16': type_ = ir.IntType(16)
			case 'i32' | 'u32': type_ = ir.IntType(32)
			case 'i64' | 'u64': type_ = ir.IntType(64)
			case 'i128' | 'u128': type_ = ir.IntType(128)
			case 'int': type_ = ir.IntType(32) # TODO FIXME
			case _: raise WTFException(type_)

		v = self.load(x.expr, builder=builder, type_=type_)

		if (builder is self.builder):  # global
			global_ = ir.GlobalVariable(self.module, type_, name=x.identifier.identifier)
			self.store(x.identifier, global_)
			if (isinstance(v, ir.Constant)): global_.initializer = v
			else:
				global_.initializer = ir.Constant(type_, ir.Undefined)
				builder.store(v, global_)
		else: raise TODO

	@dispatch
	def load(self, x: ASTCallArgNode, **kwargs):
		return self.load(x.expr, **kwargs)

	@dispatch
	def load(self, x: ASTCallArgsNode, **kwargs):
		for i in x.callarg:
			v = self.load(i, **kwargs)
			if (i.star): yield from v
			else: yield v

	@dispatch
	def load(self, x: ASTCallKwargsNode, **kwargs):
		for i in x.callkwarg:
			yield self.load(i, **kwargs)

	@dispatch
	def load(self, x: ASTFunccallArgsNode, **kwargs):
		for i in (x.callargs or ()):
			yield from self.load(i, **kwargs)

		# TODO: (x.callkwargs or ())

	@dispatch
	def load(self, x: ASTAttrgetNode, **kwargs):
		if (str(x.expr) != 'stdio'): raise NotImplementedError(x)
		if (x.attrop.attrselfop.op != '.'): raise NotImplementedError(x)
		return self.load(x.identifier) # TODO FIXME


	## Final

	@dispatch
	def add(self, x: ASTStatementNode, **kwargs):
		self.add(x.value, **kwargs)

	@dispatch
	def add(self, x: ASTVardefNode, *, builder):
		type_ = self.ns.type(x.type_)

		for i in x.vardefassignment:
			self.add(i, builder=builder, type_=type_)

	@dispatch
	def add(self, x: ASTFunccallNode, **kwargs):
		self.load(x, **kwargs)

	@dispatch
	def load(self, x: ASTFunccallNode, *, builder):
		func = self.load(x.expr, builder=builder)
		args = tuple(self.load(x.funccallargs, builder=builder))
		return builder.call(func, args)

	@dispatch
	def add(self, x: ASTAssignmentNode, *, builder, global_: ir.GlobalVariable = None):
		assert (x.assignment == '=') # TODO

		v = builder.load(self.load(x.expr))

		if (global_ is not None): builder.store(v, global_)
		else: self.store(x.varname, v)

	@dispatch
	def add(self, x: ASTKeyworddefNode, *, builder):
		if (x.defkeyword.defkeyword == 'main'):
			func = ir.Function(self.module, ir.FunctionType(ir.VoidType(), ()), name='main')
			self.add(x.block, func=func)
		else: raise NotImplementedError(x.defkeyword)

	## Identifiers

	@dispatch
	def load(self, x: ASTIdentifierNode, **kwargs):
		return self.load(x.identifier, **kwargs)

	@dispatch
	def store(self, x: ASTIdentifierNode, *args, **kwargs):
		self.store(x.identifier, *args, **kwargs)

	@dispatch
	def load(self, x: ASTVarnameNode, **kwargs):
		return self.load(x.value)


	## Literals

	@dispatch
	def load(self, x: ASTLiteralNode, *, builder=None, type_):
		return type_(x)


	## Comments

	@dispatch
	def add(self, x: ASTBlockCommentNode, *, builder):
		builder.comment(x.blockcomment)

	@dispatch
	def add(self, x: ASTLineCommentNode, *, builder):
		builder.comment(x.linecomment)

	@dispatch
	def add(self, x: ASTCommentNode, **kwargs):
		self.add(x.value, **kwargs)


	## Generic

	@dispatch
	def load(self, name: str) -> ir.Value:
		return self.scope[name]

	@dispatch
	def store(self, name: str, x: ir.Value):
		self.scope[name] = x

@export
class LLVMCompiler(Compiler):
	ext = '.ll'

	@classmethod
	def compile_ast(cls, ast, ns, *, filename):
		instrs = Instrs(name='<module>', ns=ns, filename=filename)
		instrs.add(ast.code)
		code = instrs.compile()
		return code

# by Sdore, 2025
# slang.sdore.me
