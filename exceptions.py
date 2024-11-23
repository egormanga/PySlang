#!/usr/bin/env python3
# PySlang Exceptions

from utils.nolog import *

class SlException(Exception, Slots):
	desc: ...
	srclines: ...
	lineno: ...
	offset: ...
	length: ...
	char: ...
	usage: ...

	def __init__(self, desc, *args, lineno, offset, length, char=0, scope=None, usage=None, srclines=()):
		self.desc, self.lineno, self.offset, self.length, self.char, self.usage, self.srclines = f"{f'\033[2m(in {scope})\033[0m ' if (scope is not None) else ''}{desc}", lineno, offset, length, char, usage, srclines
		super().__init__(*args)

	def __repr__(self):
		return f"""{self.__class__.__name__}({self.__repr_args__()}{f", {S(', ').join(self.args)}" if (self.args) else ''})"""

	def __repr_args__(self):
		return f"{' '.join(map(str.strip, self.desc.split('\n')))!r}, lineno={self.lineno}, offset={self.offset}, length={self.length}, char={self.char}"

	def __str__(self):
		l, line = (lstripcount(self.srclines[self.lineno-1].partition('\n')[0]) if (self.srclines) else (0, ''))
		offset = (self.offset-l if (self.offset >= 0) else len(line)+self.offset+1)

		return f"""{self.desc}{f" ({S(', ').join(self.args)})" if (self.args) else ''}{self.at}"""+(':\n'+\
			' '*2+'\033[1m'+line[:offset]+'\033[91m'*(self.offset >= 0)+line[offset:]+'\033[0m\n'+\
			' '*(2+offset)+'\033[95m'+'~'*self.char+'^'+'~'*(min(self.length, len(line)-offset)-1 - self.char)+'\033[0m' if (line) else '') + \
			(f"\n\n\033[1;95mCaused by:\033[0m\n{self.__cause__ if (isinstance(self.__cause__, SlException)) else ' '+str().join(traceback.format_exception(type(self.__cause__), self.__cause__, self.__cause__.__traceback__))}" if (self.__cause__ is not None) else '')

	@property
	def at(self):
		return (f" at line {self.lineno}, offset {self.offset}" if (self.offset >= 0) else f" at the end of line {self.lineno}")

	@classmethod
	def from_node(cls, node, *args, **kwargs):
		return cls(*args, **setdefault(kwargs, lineno=node.lineno, offset=node.offset, length=node.length))

class SlSyntaxException(SlException): pass
class SlSyntaxNoToken(SlSyntaxException): pass
class SlSyntaxEmpty(SlSyntaxNoToken): pass

class SlSyntaxError(SlSyntaxException):
	def __init__(self, desc="Syntax error", *args, **kwargs):
		super().__init__(desc, *args, **kwargs)

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
		super().__init__(expected=f"more tokens for {for_}", found='nothing', offset=offset, **kwargs)

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
	_expected: ...

	def __init__(self, expected, found, *, scope=None, errlist=None, **kwargs):
		self.errlist = errlist
		self.sl = (len(scope)+6 if (scope is not None) else 0)
		self._expected = tuple(j for i in expected for j in (i._expected if (isinstance(i, SlSyntaxMultiExpectedError)) else (i,)))
		super().__init__(
			expected = (S(',\n'+' '*(self.sl+9)).join(Stuple((f"{S(', ').join(Stuple(getattr(i.expected, 'name', i.expected) for i in choices).uniquize(str.casefold), last=' or ')} at line {lineno}, {f'offset {offset}' if (offset >= 0) else 'the end of line'}" + (f' (for {usage})' if (usage is not None) else '')) for (lineno, offset, usage), choices in itertools.groupby(self._expected, key=lambda x: (x.lineno, x.offset, x.usage))).strip('nothing').uniquize(str.casefold), last=f",\n{' '*(self.sl+6)}or ") or 'nothing'), #if (not isinstance(i, SlSyntaxMultiExpectedError)) else str(i.expected)
			found = (S(',\n'+' '*(self.sl+6)).join(Stuple((f"{getattr(i.found, 'name', i.found)} at line {i.lineno}, {f'offset {i.offset}' if (i.offset >= 0) else 'the end of line'}" if (not isinstance(i, SlSyntaxMultiExpectedError)) else str(i.found)) for i in found).strip('nothing').uniquize(str.casefold), last=f",\n{' '*(self.sl+2)}and ") or 'nothing'),
			#{i.offset+1 if (i.offset < -1) else ''}
			scope = scope,
			**kwargs
		)

	@property
	def at(self):
		return (f"\n{' '*self.sl}at line {self.lineno}, offset {self.offset}" if (len({(i.lineno, i.offset) for i in self.errlist}) > 1) else '')

	@classmethod
	def from_list(cls, err: list[SlSyntaxError], scope=None, **kwargs):
		if (not err): raise ValueError("Error list must not be empty.")

		sl = (len(scope)+6 if (scope is not None) else 0)

		for i in err:
			if (isinstance(i, SlSyntaxMultiExpectedError)):
				i.expected = Sstr(i.expected).indent(sl).lstrip()
				i.found = Sstr(i.found).indent(sl).lstrip()

		lineno = max(i.lineno for i in err)

		return cls(
			expected = (sorted(err, key=lambda x: getattr(x.expected, 'name', None) or str(x.expected), reverse=True)),
			found = sorted(err, key=lambda x: x.offset),
			lineno = lineno,
			offset = max((i for i in err if i.lineno == lineno), key=lambda x: x.offset if (x.offset >= 0) else inf).offset,
			length = (min((i.length for i in err if i.length), default=0) if (not any(i.offset < 0 for i in err)) else 0),
			scope = scope,
			errlist = list(err),
			**kwargs
		)

class SlValidationException(SlException): pass
class SlValidationError(SlValidationException):
	def __init__(self, desc="Validation error", *args, **kwargs):
		super().__init__(desc, *args, **kwargs)

# by Sdore, 2021-24
#  slang.sdore.me
