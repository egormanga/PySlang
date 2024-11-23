#!/usr/bin/env python3
# PySlang Slang Bytecode (SBC) compiler target

from .. import *
from ...ast import *
from utils import *

NOP	= 0x00
POP	= 0x01
DUP	= 0x02
RET	= 0x03
CODE	= 0x04
IF	= 0x05
LOOP	= 0x06
ELSE	= 0x07
END	= 0x08
CALL	= 0x09
ASGN	= 0x0A
BLTIN	= 0x0B
CONST	= 0x0C
SGET	= 0x0D
SSET	= 0x0E

MAGIC = b"\x0C\x80\x03SBC\x01"

def writeVarInt(v):
	assert (v >= 0)
	r = bytearray()
	while (True):
		c = (v & 0x7f)
		v >>= 7
		if (v): c |= 0x80
		r.append(c)
		if (not v): break
	return bytes(r)

class Instrs:
	unopmap = {
		'!': 'not',
		'+': 'abs',
		'-': 'neg',
		'~': 'inv',
		'++': 'inc',
		'--': 'dec',
		'**': 'sqr',
		'not': 'not',
	}
	binopmap = {
		'+': 'add',
		'-': 'sub',
		'*': 'mul',
		'/': 'div',
		'//': 'idiv',
		'%': 'mod',
		'**': 'pow',
		'<<': 'sl',
		'>>': 'sr',
		'&': 'and',
		'^': 'xor',
		'|': 'or',

		'&&': 'and',
		'^^': 'xor',
		'||': 'or',
		'==': 'eq',
		'!=': 'ne',
		'<': 'lt',
		'>': 'gt',
		'<=': 'le',
		'>=': 'ge',

		'is': 'is',
		'is not': 'isnt',
		'in': 'in',
		'not in': 'nin',
		'isof': 'isof',
		'and': 'and',
		'but': 'and',
		'xor': 'xor',
		'or': 'or',
		'to': 'to',
	}

	@init_defaults
	def __init__(self, *, name, ns, filename, scpcells: indexset):
		self.name, self.ns, self.filename, self.scpcells = name, ns, filename, scpcells
		self.instrs = bytearray(MAGIC)
		self.opmap = dict()

	def compile(self):
		return bytes(self.instrs)

	@dispatch
	def add(self, opcode: int, *args: int):
		self.instrs.append(opcode)
		if (args): self.instrs += bytes(args)

	@dispatch
	def load(self, x: str, **kwargs):
		self.add(SGET, self.scpcells[x])

	@dispatch
	def store(self, x: str):
		self.add(SSET, self.scpcells[x])

	@dispatch
	def builtin(self, builtin: str):
		name = bytearray(builtin.encode('ascii'))
		name[-1] |= 0x80
		self.add(BLTIN)
		self.instrs += name

	@dispatch
	def builtin(self, builtin: str, *, nargs: int):
		if (nargs is not None and len(self.opmap) < 0xf0): self.assign(builtin, 0, nargs)
		if ((builtin, 0, nargs) in self.opmap): self.add(self.opmap[builtin, 0, nargs])
		else: self.builtin(builtin); self.add(CALL, nargs)

	@dispatch
	def assign(self, builtin: str,
			 nargs_code: lambda nargs_code: isinstance(nargs_code, int) and 0 <= nargs_code < 4,
			 nargs_stack: lambda nargs_stack: isinstance(nargs_stack, int) and 0 <= nargs_stack < 64,
			 opcode: int = None):
		self.builtin(builtin)
		if (opcode is None): opcode = first(sorted(set(range(0x10, 0x100)) - set(self.opmap.values())))
		self.add(ASGN, opcode, (nargs_code << 6) | nargs_stack)
		self.opmap[builtin, nargs_code, nargs_stack] = opcode


	## Abstract

	@dispatch
	def add(self, x: ASTCodeNode):
		for i in x.statement or ():
			self.add(i)

	@dispatch
	def add(self, x: ASTBlockNode):
		if (x.code is not None):
			self.add(x.code)
			assert (not x.statement)
		else:
			for i in x.statement:
				self.add(i)


	## Primitive

	@dispatch
	def load(self, x: ASTBinOpExprNode, **kwargs):
		self.load(x.lhs, **kwargs)
		self.load(x.rhs, **kwargs)
		self.add(x.binop)

	@dispatch
	def load(self, x: ASTValueNode, **kwargs):
		found = bool()
		for i in x.__slots__:
			if (v := getattr(x, i, None)):
				assert (not found)
				self.load(v, **kwargs)
				found = True


	## Non-final

	@dispatch
	def load(self, x: ASTCallArgNode, **kwargs):
		if (x.star): raise NotImplementedError(x)
		self.load(x.expr)

	@dispatch
	def load(self, x: ASTCallArgsNode, **kwargs):
		for i in x.callarg:
			self.load(i)

	@dispatch
	def load(self, x: ASTCallKwargsNode, **kwargs):
		for i in x.callkwarg:
			self.load(i)

	@dispatch
	def load(self, x: ASTAttrgetNode, **kwargs):
		if (x.expr.varname.identifier.identifier != 'stdio'): raise NotImplementedError(x)
		if (x.attrop.attrselfop.op != '.'): raise NotImplementedError(x)
		self.builtin(x.identifier.identifier)


	## Final

	@dispatch
	def add(self, x: ASTStatementNode):
		found = bool()
		for i in x.__slots__:
			if (v := getattr(x, i, None)):
				assert (not found)
				self.add(v)
				found = True

	@dispatch
	def add(self, x: ASTVardefNode):
		for i in x.vardefassignment:
			self.load(i.expr, typename=x.typename) #sig=Signature.build(x.typename, ns=self.ns))
			self.store(i.identifier)

	@dispatch
	def add(self, x: ASTFunccallNode):
		self.load(x)
		self.add(POP)

	@dispatch
	def load(self, x: ASTFunccallNode):
		nargs = int()

		for i in x.callargs:
			self.load(i)
			nargs += 1

		#if (isinstance(x.callable, ASTValueNode) and isinstance(x.callable.value, ASTIdentifierNode) and x.callable.value.identifier in self.ns.signatures):
		#	raise NotImplementedError()
			#name = f"{x.callable.value.identifier}({CallArguments.build(x, self.ns)})"
			#self.load(name)
			#self.load(name+'.len')
			#self.add(EXEC)
		#else:
		self.load(x.expr)
		self.add(CALL, nargs)

	@dispatch
	def add(self, x: ASTKeyworddefNode):
		if (x.defkeyword.keyword == 'main'):
			name = '<main>'
			code_ns = self.ns #.derive(name)
			f_instrs = Instrs(name=name, ns=code_ns, filename=self.filename)
			f_instrs.add(x.block)
			self.add(CODE)
			self.instrs += f_instrs.instrs
			self.add(END)
			self.add(CALL, 0)
			#self.add(POP)
		else: raise NotImplementedError(x.defkeyword)


	## Identifiers

	@dispatch
	def load(self, x: ASTIdentifierNode, **kwargs):
		self.load(x.identifier, **kwargs)

	@dispatch
	def store(self, x: ASTIdentifierNode):
		self.store(x.identifier)

	@dispatch
	def load(self, x: ASTVarnameNode, **kwargs):
		found = bool()
		for i in x.__slots__:
			if (v := getattr(x, i, None)):
				assert (not found)
				self.load(v)
				found = True


	## Literals
	@dispatch
	def load(self, x: ASTLiteralNode, *, typename):
		#if (sig is None): sig = Signature.build(x, ns=self.ns)

		#if (hasattr(sig, 'fmt')):
		#	t, v = sig.__class__.__name__, struct.pack(sig.fmt, int(x.literal))
		#elif (isinstance(sig, stdlib.int)):
		#	t, v = 'i', writeVarInt(x.number)
		#elif (isinstance(sig, stdlib.str)):
		#	t, v = 's', eval_literal(x).encode('utf-8')
		#elif (isinstance(sig, stdlib.char)):
		#	t, v = 'c', eval_literal(x).encode('utf-8') # TODO
		#else: raise NotImplementedError(sig)

		t = typename.type.type
		if (isinstance(x.number, int)): v = writeVarInt(x.number)
		else: v = str(x.number).encode() # TODO FIXME

		type_ = bytearray(t.encode('ascii'))
		type_[-1] |= 0x80

		self.add(CONST)
		self.instrs += (type_ + writeVarInt(len(v)) + v)


	## Operators

	@dispatch
	def add(self, x: ASTBinOpNode):
		self.builtin(self.binopmap[x.binchop], nargs=2)

	## Comments

	@dispatch
	def add(self, x: ASTCommentNode):
		pass


	#@dispatch
	#def add(self, x: ASTConditionalNode):
	#	self.load(x.condition)
	#	self.add(IF)
	#	self.add(x.code)
	#	self.add(END)

	#@dispatch
	#def add(self, x: ASTForLoopNode):
	#	self.load(x.iterable)
	#	self.builtin('iter', 1)
	#	self.store(x.name)
	#	self.add(DUP, 0)
	#	self.add(LOOP)
	#	self.add(x.code)
	#	self.builtin('iter', 1)
	#	self.store(x.name)
	#	self.add(DUP, 0)
	#	self.add(END)

	#@dispatch
	#def add(self, x: ASTWhileLoopNode):
	#	self.load(x.condition)
	#	self.add(LOOP)
	#	self.add(x.code)
	#	self.load(x.condition)
	#	self.add(END)

	#@dispatch
	#def add(self, x: ASTElseClauseNode):
	#	assert (self.instrs[-1] == END)
	#	end = self.instrs.pop()
	#	self.add(ELSE)
	#	self.add(x.code)
	#	self.add(end)


class SBCCompiler(Compiler):
	ext = '.sbc'

	@classmethod
	def compile_ast(cls, ast, ns, *, filename):
		instrs = Instrs(name='<module>', ns=ns, filename=filename)
		instrs.add(ast.code)
		code = instrs.compile()
		return code

compiler = SBCCompiler

# by Sdore, 2022-24
#  slang.sdore.me
