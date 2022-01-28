#!/usr/bin/env python3
# PySlang AST

from utils.nolog import *

class SlException(Exception, Slots): pass
class SlSyntaxException(SlException): pass
class SlSyntaxNoToken(SlSyntaxException): pass
class SlSyntaxEmpty(SlSyntaxNoToken): pass

class SlSyntaxError(SlSyntaxException):
	desc: ...
	srclines: ...
	lineno: ...
	offset: ...
	length: ...
	char: ...
	usage: ...

	def __init__(self, desc='Syntax error', srclines=(), *, lineno, offset, length, char=0, scope=None, usage=None):
		self.desc, self.srclines, self.lineno, self.offset, self.length, self.char, self.usage = (f'\033[2m(in {scope})\033[0m ' if (scope is not None) else '')+desc, srclines, lineno, offset, length, char, usage

	def __repr__(self):
		return f"{self.__class__.__name__}({repr(' '.join(map(str.strip, self.desc.split(ENDL))))}, lineno={self.lineno}, offset={self.offset}, length={self.length}, char={self.char})"

	def __str__(self):
		l, line = lstripcount(self.srclines[self.lineno-1].partition('\n')[0]) if (self.srclines) else (0, '')
		offset = (self.offset-l) if (self.offset >= 0) else (len(line)+self.offset+1)

		return f"{self.desc} {self.at}"+(':\n'+\
			' '*2+'\033[1m'+line[:offset]+'\033[91m'*(self.offset >= 0)+line[offset:]+'\033[0m\n'+\
			' '*(2+offset)+'\033[95m'+'~'*self.char+'^'+'~'*(self.length-1-self.char)+'\033[0m' if (line) else '') + \
			(f"\n\n\033[1;95mCaused by:\033[0m\n{self.__cause__ if (isinstance(self.__cause__, SlException)) else ' '+str().join(traceback.format_exception(type(self.__cause__), self.__cause__, self.__cause__.__traceback__))}" if (self.__cause__ is not None) else '')

	@property
	def at(self):
		return f"at line {self.lineno}, offset {self.offset}" if (self.offset >= 0) else f"at the end of line {self.lineno}"

class SlSyntaxExpectedError(SlSyntaxError):
	expected: ...
	found: ...

	def __init__(self, expected, found, *, lineno, offset, length=0, scope=None, **kwargs):
		assert (expected != found)

		self.expected, self.found = expected, found

		try: expected = expected.name
		except AttributeError: pass

		try: found = f"{found.typename} {found}"
		except AttributeError: pass

		super().__init__(f"Expected {expected},\n{' '*(len(scope)+6 if (scope is not None) else 0)}found {found}", lineno=lineno, offset=offset, length=length, scope=scope, **kwargs)

class SlSyntaxExpectedNothingError(SlSyntaxExpectedError):
	def __init__(self, found, **kwargs):
		super().__init__(expected='nothing', found=found, **kwargs)

class SlSyntaxExpectedMoreTokensError(SlSyntaxExpectedError):
	def __init__(self, for_, *, offset=-1, **kwargs):
		assert (offset < 0)
		super().__init__(expected=f"More tokens for {for_}", found='nothing', offset=offset, **kwargs)

class SlSyntaxExpectedOneOfError(SlSyntaxExpectedError):
	def __init__(self, expected, found='nothing', **kwargs):
		choices = expected.choices.copy()

		for ii, i in enumerate(choices):
			try: choices[ii] = i.name
			except AttributeError: pass

		super().__init__(expected=f"{expected.name} ({S(', ').join(S(choices).uniquize(), last=' or ')})", found=found, **kwargs)

class SlSyntaxMultiExpectedError(SlSyntaxExpectedError):
	sl: ...
	errlist: ...

	def __init__(self, expected, found, *, scope=None, errlist=None, **kwargs):
		self.errlist = errlist
		self.sl = len(scope)+6 if (scope is not None) else 0
		super().__init__(
			expected=S(',\n'+' '*(self.sl+9)).join(Stuple((f"{getattr(i.expected, 'name', i.expected)} at {f'offset {i.offset}' if (i.offset >= 0) else 'the end of line'}" if (not isinstance(i, SlSyntaxMultiExpectedError)) else str(i.expected))+(f' (for {i.usage})' if (i.usage is not None) else '') for i in expected).strip('nothing').uniquize(str.casefold), last=',\n'+' '*(self.sl+6)+'or ') or 'nothing',
			found=S(',\n'+' '*(self.sl+6)).join(Stuple(f"{getattr(i.found, 'name', i.found)} at {f'offset {i.offset}' if (i.offset >= 0) else 'the end of line'}" if (not isinstance(i, SlSyntaxMultiExpectedError)) else str(i.found) for i in found).strip('nothing').uniquize(str.casefold), last=',\n'+' '*(self.sl+2)+'and ') or 'nothing',
			#{i.offset+1 if (i.offset < -1) else ''}
			scope=scope,
			**kwargs
		)

	@property
	def at(self):
		return f"\n{' '*self.sl}at line {self.lineno}"

	@classmethod
	def from_list(cls, err: [SlSyntaxError], scope=None, **kwargs):
		if (not err): raise ValueError("Error list must not be empty.")

		sl = len(scope)+6 if (scope is not None) else 0

		for i in err:
			if (isinstance(i, SlSyntaxMultiExpectedError)):
				i.expected = Sstr(i.expected).indent(sl).lstrip()
				i.found = Sstr(i.found).indent(sl).lstrip()

		lineno = max(i.lineno for i in err)

		return cls(
			expected=sorted(err, key=lambda x: getattr(x.expected, 'name', None) or str(x.expected), reverse=True),
			found=sorted(err, key=lambda x: x.offset),
			lineno=lineno,
			offset=max((i for i in err if i.lineno == lineno), key=lambda x: x.offset if (x.offset >= 0) else inf).offset,
			length=min((i.length for i in err if i.length), default=0) if (not any(i.offset < 0 for i in err)) else 0,
			scope=scope,
			errlist=list(err),
			**kwargs
		)

# by Sdore, 2021-2022
#  slang.sdore.me
