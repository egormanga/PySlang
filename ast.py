#!/usr/bin/env python3
# PySlang AST

from utils.nolog import *

class SlSyntaxException(Exception, Slots): pass
class SlSyntaxNoToken(SlSyntaxException): pass
class SlSyntaxEmpty(SlSyntaxNoToken): pass

class SlSyntaxError(SlSyntaxException):
	desc: ...
	srclines: ...
	lineno: ...
	offset: ...
	length: ...
	char: ...
	usage: None

	def __init__(self, desc='Syntax error', srclines=(), *, lineno, offset, length, char=0, scope=None):
		self.desc, self.srclines, self.lineno, self.offset, self.length, self.char = (f'\033[2m(in {scope})\033[0m ' if (scope is not None) else '')+desc, srclines, lineno, offset, length, char

	def __str__(self):
		l, line = lstripcount(self.srclines[self.lineno-1].partition('\n')[0]) if (self.srclines) else (0, '')
		offset = (self.offset-l) if (self.offset >= 0) else (len(line)+self.offset+1)

		return f"{self.desc} {self.at}"+(':\n'+\
			' '*2+'\033[1m'+line[:offset]+'\033[91m'*(self.offset >= 0)+line[offset:]+'\033[0m\n'+\
			' '*(2+offset)+'\033[95m'+'~'*self.char+'^'+'~'*(self.length-1-self.char)+'\033[0m' if (line) else '') + \
			(f"\n\n\033[1;95mCaused by:\033[0m\n{self.__cause__ if (isinstance(self.__cause__, (SlSyntaxError, SlValidationError, SlCompilationError))) else ' '+str().join(traceback.format_exception(type(self.__cause__), self.__cause__, self.__cause__.__traceback__))}" if (self.__cause__ is not None) else '')

	@property
	def at(self):
		return f"at line {self.lineno}, offset {self.offset}" if (self.offset >= 0) else f"at the end of line {self.lineno}"

class SlSyntaxExpectedError(SlSyntaxError):
	expected: ...
	found: ...

	def __init__(self, expected='nothing', found='nothing', *, lineno=None, offset=None, length=0, scope=None):
		assert (expected != found)
		if (not isinstance(found, str)): lineno, offset, length, found = found.lineno, found.offset, found.length, found.typename if (hasattr(found, 'typename')) else found
		assert (lineno is not None and offset is not None)
		super().__init__(f"Expected {expected.lower()},\n{' '*(len(scope)+6 if (scope is not None) else 3)}found {found.lower()}", lineno=lineno, offset=offset, length=length, scope=scope)
		self.expected, self.found = expected, found

class SlSyntaxExpectedNothingError(SlSyntaxExpectedError):
	def __init__(self, found, **kwargs):
		super().__init__(found=found, **kwargs)

class SlSyntaxExpectedMoreTokensError(SlSyntaxExpectedError):
	def __init__(self, for_, *, offset=-1, **kwargs):
		assert (offset < 0)
		super().__init__(expected=f"More tokens for {for_}", offset=offset, **kwargs)

class SlSyntaxMultiExpectedError(SlSyntaxExpectedError):
	sl: ...
	errlist: ...

	def __init__(self, expected, found, *, scope=None, errlist=None, **kwargs):
		self.errlist = errlist
		self.sl = len(scope)+6 if (scope is not None) else 0
		super().__init__(
			expected=S(',\n'+' '*(self.sl+9)).join(Stuple((i.expected+f" at {f'offset {i.offset}' if (i.offset >= 0) else 'the end of line'}" if (not isinstance(i, SlSyntaxMultiExpectedError)) else i.expected)+(f' (for {i.usage})' if (i.usage) else '') for i in expected).strip('nothing').uniquize(str.casefold), last=',\n'+' '*(self.sl+6)+'or ') or 'nothing',
			found=S(',\n'+' '*(self.sl+6)).join(Stuple(i.found+f" at {f'offset {i.offset}' if (i.offset >= 0) else 'the end of line'}" if (not isinstance(i, SlSyntaxMultiExpectedError)) else i.expected for i in found).strip('nothing').uniquize(str.casefold), last=',\n'+' '*(self.sl+2)+'and ') or 'nothing',
			#{i.offset+1 if (i.offset < -1) else ''}
			scope=scope,
			**kwargs
		)

	@property
	def at(self):
		return f"\n{' '*self.sl}at line {self.lineno}"

	@classmethod
	def from_list(cls, err, scope=None, **kwargs):
		sl = len(scope)+6 if (scope is not None) else 0

		for i in err:
			if (isinstance(i, SlSyntaxMultiExpectedError)):
				loff = 0
				i.expected = ' '*loff+S(i.expected).indent(sl+loff).lstrip()
				i.found = ' '*loff+S(i.found).indent(sl+loff).lstrip()

		lineno = max(err, key=operator.attrgetter('lineno')).lineno
		return cls(
			expected=sorted(err, key=operator.attrgetter('expected'), reverse=True),
			found=sorted(err, key=operator.attrgetter('offset')),
			lineno=lineno,
			offset=max((i for i in err if i.lineno == lineno), key=lambda x: x.offset if (x.offset >= 0) else inf).offset,
			length=min(i.length for i in err if i.length) if (not any(i.offset < 0 for i in err)) else 0,
			scope=scope,
			errlist=list(err),
			**kwargs
		)

