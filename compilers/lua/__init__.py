#!/usr/bin/env python3
# PySlang Lua compiler target

from .. import *
from ...ast import *
from utils import *

class Instrs:
	unopmap = {
		'!': ':__not()',
		'+': ':__unp()',
		'-': '-',
		'~': '~',
		'++': ':__preinc()',
		'--': ':__predec()',
		'not': ':__not()',
	}
	unpostopmap = {
		'!': ':__fact()',
		'++': ':__postinc()',
		'--': ':__postdec()',
		'**': ':__sqr()',
	}
	binopmap = {
		'+': '+',
		'-': '-',
		'*': '*',
		'/': '/',
		'//': '//',
		'%': '%',
		'**': '^',
		'<<': '<<',
		'>>': '>>',
		'&': '&',
		'|': '|',
		'^': '~',

		'&&': ':__and(',
		'||': ':__or(',
		'^^': ':__xor(',
		'==': '==',
		'!=': '~=',
		'<': '<',
		'>': '>',
		'<=': '<=',
		'>=': '>=',

		'is': 'rawequal(',
		'is not': 'not rawequal(',
		'in': ':__in(',
		'not in': ':__not_in(',
		'isof': ':__isof(',
		'and': ':__and(',
		'but': ':__and(',
		'or': ':__or(',
		'xor': ':__xor(',
		'to': ':__to(',
	}

	@init_defaults
	def __init__(self, *, name, ns, filename, scpcells: indexset):
		self.name, self.ns, self.filename, self.scpcells = name, ns, filename, scpcells

	@dispatch.meta
	def add(self, x) -> str: ...

	@dispatch.meta
	def classadd(self, x) -> str: ...

	@dispatch.meta
	def load(self, x) -> str: ...


	## Abstract

	@dispatch
	def add(self, x: ASTCodeNode, *, indent=0):
		return ('\n' + '\t'*indent).join(map(self.add, x.statement))

	@dispatch
	def add(self, x: ASTClassCodeNode, *, indent=0):
		return ('\n' + '\t'*indent).join(map(self.add, x.classstatement))

	@dispatch
	def add(self, x: ASTClassdefCodeNode, *, classname: str):
		return '\n'.join(self.add(i, classname=classname) for i in x.classdefstatement)

	@dispatch
	def add(self, x: ASTBlockNode, indent=False):
		linestart = ('\n\t' + '\t'*indent)
		if (x.code):
			assert (not x.statement)
			return (linestart + self.add(x.code, indent=(indent+1)) + '\n')
		else: return (linestart + linestart.join(map(self.add, x.statement)) + '\n')

	@dispatch
	def add(self, x: ASTClassBlockNode, *, indent=False):
		linestart = ('\n\t' + '\t'*indent)
		if (x.classcode):
			assert (not x.classstatement)
			return (linestart + self.add(x.classcode, indent=(indent+1)) + '\n')
		else: return (linestart + linestart.join(map(self.add, x.classstatement)) + '\n')

	@dispatch
	def add(self, x: ASTClassdefBlockNode, *, classname: str):
		return self.add(x.classdefcode, classname=classname)


	## Primitive

	@dispatch
	def load(self, x: ASTUnOpExprNode):
		return self.load(x.value)

	@dispatch
	def load(self, x: ASTUnPreOpExprNode):
		unop = self.load(x.unop)
		expr = self.load(x.expr)
		if (unop.startswith(':')): return f"{expr}{unop}"
		else: return f"{unop}{expr}"

	@dispatch
	def load(self, x: ASTBinOpExprNode):
		binop = self.load(x.binop)
		cast = (binop in '>==<=')
		call = binop.endswith('(')
		return f"({'bool('*cast}{self.load(x.lhs)}{' '*(not binop.startswith(':'))}{binop}{' '*(not call)}{self.load(x.rhs)}{')'*(call+cast)})"

	@dispatch
	def load(self, x: ASTUnPostOpExprNode):
		return f"{self.load(x.expr)}{self.load(x.unpostop)}"

	@dispatch
	def load(self, x: ASTItemgetNode):
		return f"{self.load(x.expr[0])}[{self.load(x.expr[1])}]"

	@dispatch
	def load(self, x: ASTValueNode, **kwargs):
		return self.load(x.value, **kwargs)

	@dispatch
	def load(self, x: ASTExprNode, **kwargs):
		return self.load((x.expr or x.value), **kwargs)


	## Non-final

	@dispatch
	def load(self, x: ASTTypeNode):
		return self.load(x.identifier)

	@dispatch
	def add(self, x: ASTCatchClauseNode):
		return f"if {f'type(__error) == {self.load(x.type_)}' if (x.type_) else '__error'} then{f' local {self.load(x.identifier)} = __error;' if (x.identifier) else ''} __error_, __error = __error, nil{self.add(x.block, indent=True)}\tend"

	@dispatch
	def add(self, x: ASTFinallyClauseNode):
		return self.add(x.block)

	@dispatch
	def load(self, x: ASTVardefAssignmentNode, *, type_: ASTTypeNode, classname=None):
		return f"""{f"{classname}." if (classname is not None) else ''}{self.load(x.identifier)} = {self.load(x.expr) if (x.expr) else f"{self.load(type_)}{self.load(x.funccallargs) if (x.funccallargs) else ':new()'}"}"""

	@dispatch
	def add(self, x: ASTArgdefNode, *, index: int):
		# TODO: type, mode
		identifier = self.load(x.identifier)
		return f"local {identifier} = arg[{index}] or arg.{identifier}"

	@dispatch
	def load(self, x: ASTArgdefNode):
		if (not x.expr): return ''
		else: return f"{self.load(x.identifier)} = {self.load(x.expr)}"

	@dispatch
	def add(self, x: ASTClassArgdefNode, *, index: int):
		# TODO: type, mode
		classvarname = self.load(x.classvarname)
		member = classvarname.startswith('self.')
		return f"{'local '*(not member)}{classvarname}{'<const>'*(not member and 'const' in (x.type_.modifier or ()))} = arg[{index}] or arg.{classvarname.removeprefix('self.')}"

	@dispatch
	def load(self, x: ASTClassArgdefNode):
		if (not x.expr): return ''
		else: return f"{self.load(x.classvarname)} = {self.load(x.expr)}"

	@dispatch
	def load(self, x: ASTCallArgNode):
		if (x.star): return f"table.unpack({self.load(x.expr)})"
		else: return self.load(x.expr)

	@dispatch
	def load(self, x: ASTCallArgsNode):
		return ', '.join(map(self.load, x.callarg))

	@dispatch
	def load(self, x: ASTAttrgetNode, *, funccall=None):
		op = x.attrop.attrselfop.op
		if (op != '.'): raise NotImplementedError(x)
		if (funccall and op == '.'): op = ':'
		return f"{self.load(x.expr)}{op}{self.load(x.identifier)}"

	@dispatch
	def load(self, x: ASTClassAttrgetNode):
		if (x.attrselfop):
			expr, op = 'self', x.attrselfop.op
		else:
			expr, op = self.load(x.expr), x.attrop.attrselfop.op
		if (op != '.'): raise NotImplementedError(x)
		return f"{expr}{op}{self.load(x.identifier)}"


	## Final

	@dispatch
	def add(self, x: ASTStatementNode | ASTClassStatementNode):
		return self.add(x.value)

	@dispatch
	def add(self, x: ASTClassdefStatementNode, *, classname: str):
		return self.add(x.value, classname=classname)

	@dispatch
	def add(self, x: ASTFuncdefNode):
		defaults = tuple(filter(None, map(self.load, (x.argdef or ()))))
		return f"""function {self.load(x.identifier)}{f'''(...) local arg<const> = table.pack(...); {f"setmetatable(arg, {{__index = {{{', '.join(defaults)}}}}}); " if (defaults) else ''}{'; '.join(self.add(i, index=ii) for ii, i in enumerate(x.argdef, 1))}''' if (x.argdef) else '()'}{self.add(x.block) if (x.block) else f"{';'*bool(x.argdef)} return {self.load(x.expr)} "}end"""

	@dispatch
	def add(self, x: ASTKeywordExprNode):
		return self.add(x.value)

	@dispatch
	def add(self, x: ASTConditionalNode):
		return f"""if {'\telseif '.join(f"{expr_ if (expr_ in ('true', 'false')) else f'({expr_}):__bool()' if (not expr_.startswith('bool(')) else expr_.removeprefix('bool(').removesuffix(')')} then{self.add(block, indent=True)}" for expr, block in zip(x.expr, x.block) if (expr_ := self.load(expr)))}{f"\telse {self.add(x.block[-1], indent=True)}" if (x.else_) else ''}\tend"""

	@dispatch
	def add(self, x: ASTForLoopNode):
		return f"for {self.load(x.identifier)} in pairs({self.load(x.expr)}) do{self.add(only(x.block), indent=True)}\tend" # TODO FIXME: `else'

	@dispatch
	def add(self, x: ASTWhileLoopNode):
		expr = self.load(x.expr)
		return f"while {expr if (expr in ('true', 'false')) else f'({expr}):__bool()' if (not expr.startswith('bool(')) else expr.removeprefix('bool(').removesuffix(')')} do{self.add(only(x.block), indent=True)}\tend" # TODO FIXME: `else'

	@dispatch
	def add(self, x: ASTDoBlockNode):
		return f"do local __success, __error = pcall(function(){self.add(x.block[0], indent=True)}\tend); if not __success then {'; '.join(map(self.add, x.catchclause))}{f'; else{self.add(x.block[1], indent=True)}' if (x.else_) else ''}\tend{f'{self.add(x.finallyclause)}\t' if (x.finallyclause) else ''}if __error then error(__error) end; end"

	@dispatch
	def add(self, x: ASTClassdefNode):
		classname = self.load(x.identifier[0])
		bases = tuple(map(self.load, x.identifier[1:]))
		assert (len(bases) <= 1) # TODO: multiple bases
		return (f"{classname} = "
		        + (f"{only(bases)}:new()" if (bases) else "setmetatable({new = function(self, obj) obj = obj or {}; self.__index = self; setmetatable(obj, self):init(); return obj; end, init = function() end, destroy = function() end}, {__call = function(self, ...) obj = self:new(); obj:constr(...); return obj; end})")
		        + f"\n{self.add(x.classdefblock, classname=classname)}\n")

	@dispatch
	def add(self, x: ASTVardefNode, *, classname=None):
		return f"{', '.join(self.load(i, type_=x.type_, classname=classname) for i in x.vardefassignment)}"

	@dispatch
	def add(self, x: ASTAssignmentNode):
		varname = tuple(map(self.load, x.varname))
		inplaceop = x.assignment.removesuffix('=')
		return f"{', '.join(varname)} = {f'({only(varname)} {inplaceop} ' if (inplaceop) else ''}{self.load(x.expr)}{')'*bool(inplaceop)}" # TODO FIXME: not `only(varname)'

	@dispatch
	def add(self, x: ASTClassAssignmentNode):
		classvarname = tuple(map(self.load, x.classvarname))
		inplaceop = x.assignment.removesuffix('=')
		return f"{', '.join(classvarname)} = {f'({only(classvarname)} {inplaceop} ' if (inplaceop) else ''}{self.load(x.expr)}{')'*bool(inplaceop)}"

	@dispatch
	def load(self, x: ASTFunccallArgsNode):
		return f"({', '.join(map(self.load, (x.callargs or ())))})"

	@dispatch
	def add(self, x: ASTFunccallNode):
		return f"{self.load(x)}"

	@dispatch
	def load(self, x: ASTFunccallNode):
		return f"{self.load(x.expr, funccall=True)}{self.load(x.funccallargs)}"

	@dispatch
	def add(self, x: ASTKeyworddefNode):
		match x.defkeyword.defkeyword:
			case 'main': return f"if not pcall(debug.getlocal, 4, 1) then -- main{self.add(x.block)}end\n"
			case _: raise NotImplementedError(x.defkeyword)

	@dispatch
	def add(self, x: ASTClassKeyworddefNode, *, classname: str):
		match x.classdefkeyword.classdefkeyword:
			case 'init' | 'destroy':
				return f"function {classname}:{x.classdefkeyword.classdefkeyword}(){self.add(x.classblock) if (x.classblock) else ' '}end"
			case 'constr':
				defaults = tuple(filter(None, map(self.load, (x.classargdef or ()))))
				return f"""function {classname}:constr{f'''(...) local arg<const> = table.pack(...); {f"setmetatable(arg, {{__index = {{{', '.join(defaults)}}}}}); " if (defaults) else ''}{'; '.join(self.add(i, index=ii) for ii, i in enumerate(x.classargdef, 1))}''' if (x.classargdef) else '()'}{self.add(x.classblock) if (x.classblock) else ' '}end"""
			case _: raise NotImplementedError(x.classdefkeyword)


	## Keywords

	@dispatch
	def add(self, x: ASTRaiseNode):
		return "__error = __error_"

	@dispatch
	def add(self, x: ASTDeleteNode):
		varname = self.load(x.varname)
		return f"pcall({varname}.destroy, {varname}); {varname} = nil"


	## Identifiers

	@dispatch
	def load(self, x: ASTIdentifierNode):
		return x.identifier

	@dispatch
	def load(self, x: ASTVarnameNode, *, funccall=None):
		if (x.attrget): return self.load(x.attrget, funccall=funccall)
		else: return self.load(x.identifier)

	@dispatch
	def load(self, x: ASTClassVarnameNode):
		return self.load(x.classattrget or x.identifier)


	## Literals

	@dispatch
	def load(self, x: ASTListNode):
		return f"list{{type={self.load(x.type_)}, {S(', ').join(map(self.load, x.expr))}}}"

	@dispatch
	def load(self, x: ASTLiteralNode):
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

		if (x.boolean is not None): return f"bool({x.boolean})"
		if (x.number is not None): return f"int({x.number})"
		if (x.character is not None): return f"char{x.character}"
		if (x.string is not None): return f"str{x.string}"
		if (x.list is not None): return self.load(x.list)
		raise WTFException(x)


	## Operators

	@dispatch
	def load(self, x: ASTUnOpNode):
		return self.unopmap[x.value]

	@dispatch
	def load(self, x: ASTUnPostOpNode):
		return self.unpostopmap[x.value]

	@dispatch
	def load(self, x: ASTBinOpNode):
		return self.binopmap[x.value]


	## Comments

	@dispatch
	def add(self, x: ASTBlockCommentNode):
		return f"--[[{x.blockcomment.removeprefix('#|').removesuffix('|#').replace(']]', r']\]')}]]\n"

	@dispatch
	def add(self, x: ASTLineCommentNode):
		return f"--{x.linecomment.removeprefix('#')}"

	@dispatch
	def add(self, x: ASTCommentNode, *, classname=None):
		return self.add(x.value)


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


class LuaCompiler(Compiler):
	ext = '.lua'

	@classmethod
	def compile_ast(cls, ast, ns, *, filename):
		instrs = Instrs(name='<module>', ns=ns, filename=filename)
		return ("require 'Slang'\n\n" + instrs.add(ast.code) + '\n').encode()

compiler = LuaCompiler

# by Sdore, 2024
# slang.sdore.me
