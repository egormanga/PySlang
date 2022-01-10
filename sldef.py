#!/usr/bin/env python3
# PySlang sldef parser

from utils.nolog import *

DEFAULT_DEF = os.path.join(os.path.dirname(__file__), 'Slang.sldef')

class Tokenizer:
	operators = '|*+?'
	specials = '()[]'

	@classmethod
	def tokenize(cls, s):
		tokens = list()

		while (s):
			if (not (s := s.lstrip())): break

			for i in ('reference', 'literal', 'regex', 'operator', 'special'):
				r = getattr(cls, 'find_'+i)(s)
				n, t = r if (isinstance(r, tuple)) else (r, s[:r]) if (isinstance(r, int) and r > 0) else (0, None)
				if (not t): continue
				s = s[n:]
				tokens.append(t)
				break
			else: break

		assert (not s)
		return tokens

	@classmethod
	def find_reference(cls, s):
		if (not s or not s[0].isidentifier()): return
		i = 1
		for i in range(1, len(s)):
			if (not s[i].isalnum() and s[i] != '_'): break
		else: i += 1
		if (s[:i].isidentifier()): return i
		return (0, i)

	@classmethod
	def find_literal(cls, s):
		if (not s): return
		if (s[0] == '\\'):
			if (len(s) < 2): return
			if (s[1] in '\\#\'"abefnrtv'): return 2
			if (s[1] == 'x'):
				if (len(s) < 4): return
				if (s[2] in string.hexdigits and s[3] in string.hexdigits):
					return 4  # \xhh
				return
			if (s[1] in string.octdigits):
				if (len(s) > 2 and s[2] in string.octdigits):
					if (len(s) > 3 and s[3] in string.octdigits):
						return 4  # \ooo
					return 3  # \oo
				return 2  # \o
		if (s[0] == "'"):
			if (len(s) < 2): return
			for i in range(1, len(s)):
				if (s[i] == '\n'): break
				if (s[i] == s[0]): return i+1
			return (0, i)

	@classmethod
	def find_regex(cls, s):
		if (not s): return
		if (s[0] == '/'):
			esc = bool()
			for i in range(1, len(s)):
				if (esc): esc = False; continue
				if (s[i] == '\\'): esc = True; continue
				if (s[i] == '\n'): break
				if (s[i] == s[0]): return i+1
			return (0, i)

	@classmethod
	def find_operator(cls, s):
		if (not s): return
		for i in sorted(cls.operators, key=len, reverse=True):
			if (s.startswith(i)): return (len(i), i)

	@classmethod
	def find_special(cls, s):
		if (not s): return
		for i in sorted(cls.specials, key=len, reverse=True):
			if (s.startswith(i)): return len(i)

class Format(Slots):
	class Token(Slots):
		token: ...

		def __init__(self, token):
			self.token = token

		def __repr__(self):
			return f"<\033[1;92m{self.__class__.__name__}\033[0m:" + (f"\n{S(repr(self.token)).indent()}\n" if (not isinstance(self.token, str)) else f" \033[95m{repr(self.token)}\033[0m") + ">"

	class Reference(Token):
		token: str

		def __str__(self):
			return self.token

	class Literal(Token):
		token: str

		@dispatch
		def __init__(self, token: str):
			self.token = token

		@dispatch
		def __init__(self, token: lambda token: hasattr(token, 'token')):
			self.__init__(token.token)

		def __str__(self):
			return repr(self.token)

	class Escape(Literal):
		token: str

		@dispatch
		def __init__(self, token: str):
			self.token = token.encode().decode('unicode_escape')

		def __str__(self):
			return ('\\' + (self.token.encode('unicode_escape').decode().lstrip('\\') or '\\'))

	class Pattern(Token):
		token: str

		def __str__(self):
			return self.token.join('//')

	class Optional(Token):
		def __str__(self):
			return f"{self.token}?"

	class ZeroOrMore(Optional):
		def __str__(self):
			return f"{self.token}*"

	class OneOrMore(Token):
		def __str__(self):
			return f"{self.token}+"

	class TokenList(Token):
		tokens: list

		@autocast
		def __init__(self, tokens: list):
			self.tokens = tokens

		def __repr__(self):
			return f"<\033[1;91m{self.__class__.__name__}\033[0m: [\n{S(NL).join(map(repr, self.tokens)).indent()}\n]>"

		def append(self, token):
			if (self.tokens and isinstance(self.tokens[-1], Format.TokenList)): self.tokens[-1].tokens.append(token)
			else: self.tokens.append(token)

		def pop(self):
			if (self.tokens and isinstance(self.tokens[-1], Format.TokenList)): return self.tokens[-1].tokens.pop()
			assert (self.tokens and self.tokens[-1])
			return self.tokens.pop()

		def flatten(self):
			for ii, i in enumerate(self.tokens):
				if (isinstance(i, Format.TokenList)): self.tokens[ii] = i.flatten()

			return (self.tokens[0] if (len(self.tokens) == 1) else self)

	class Sequence(TokenList):
		def __str__(self):
			return ' '.join(map(str, self.tokens)).join('()')

	class Choice(TokenList):
		def __str__(self):
			return ' | '.join(map(str, self.tokens)).join('()')

	class Joint(TokenList):
		def __str__(self):
			return ' '.join(map(str, self.tokens)).join('[]')

	format: 'TokenList'

	@dispatch
	def __init__(self, format: TokenList):
		self.format = format

	def __repr__(self):
		return f"<\033[1;93m{self.__class__.__name__}\033[0m:\n{S(repr(self.format)).indent()}\n>"

	def __str__(self):
		return str(self.format)

	@classmethod
	def parse(cls, tokens, *, _joint=False):
		if (_joint): format, end = cls.Joint([]), ']'
		else: format, end = cls.Choice([cls.Sequence([])]), ')'

		while (tokens):
			token = tokens.pop(0)
			if (token == end): break

			if (token in ('(', '[')): token = cls.parse(tokens, _joint=token == '[').format
			elif (token == '|'):
				if (not isinstance(format, cls.Choice) or format.tokens[-1].tokens and format.tokens[-1].tokens[-1] == '|'): raise WTFException(token)
				format.tokens.append(cls.Sequence([]))
				continue
			elif (token == '+'): token = cls.OneOrMore(format.pop())
			elif (token == '*'): token = cls.ZeroOrMore(format.pop())
			elif (token == '?'): token = cls.Optional(format.pop())
			elif (m := re.fullmatch(r'/(.+)/', token)): token = cls.Pattern(m[1])
			elif (m := re.fullmatch(r'\\.+', token)): token = cls.Escape(m[0])
			elif (m == r'\#'): token = cls.Literal('#')
			elif (m := re.fullmatch(r'''(['"])(.+)\1''', token)): token = cls.Literal(m[2])
			else: token = cls.Reference(token)

			format.append(token)

		format = format.flatten()
		if (not isinstance(format, cls.TokenList)): format = cls.Sequence([format])

		return cls(format)

@export
class SlDef(Slots):
	class Definition(Slots):
		name: str
		format: list
		special: bool
		final: bool

		def __init__(self, name, format, *, special=False, final=False):
			self.name, self.format, self.special, self.final = name, format, special, final

		def __name_repr__(self):
			return f"{':'*self.special}{'@'*self.final}{self.name}"

		def __repr__(self):
			return f"<\033[1;94m{self.__class__.__name__}\033[0m \033[96m{self.__name_repr__()}\033[0m:\n{S(repr(self.format)).indent()}\n>"

		def __str__(self):
			return f"{self.__name_repr__()}: {self.format}"

		@classmethod
		def _choices(cls, format):
			res = [[]]
			for i in format:
				if (i == '|'): res.append([])
				elif (isiterablenostr(i)): res[-1].append(cls._choices(i))
				else: res[-1].append(i)
			return res

		@property
		def choices(self):
			return self._choices(self.format)

		@property
		def literals(self):
			return tuple(j.strip().strip("'").join("''").encode().decode('unicode_escape') for i in self.choices if assert_(len(i) == 1) for j in i)

		@property
		def charset(self):
			return str().join(self.literals)

		@classmethod
		def parse(cls, *, name, tokens, special, final):
			format = Format.parse(tokens)
			res = cls(name, format, special=special, final=final)
			return res

	def __init__(self, definitions=None, finals=None):
		if (definitions is not None): self.definitions |= definitions
		if (finals is not None): self.finals += finals

	definitions: AttrDict
	finals: list

	@classmethod
	def build(cls, src):
		defs = dict()

		pl = None
		for l in src.split('\n'):
			if (l.rstrip().startswith('#')): continue

			l = re.split(r'[^\\]#', l, maxsplit=1)[0]

			if (pl is not None and l[:1].isspace()): l = pl+' '+l.strip()
			else: l, pl = l.strip(), None

			if (not l): continue

			if (l.endswith('|')): pl = l; continue

			if (l.startswith(':')):
				name, _, value = l[1:].partition(':')
				name = ':'+name
			else: name, _, value = l.partition(':')

			assert (name not in defs)

			defs[name] = value.strip()

		assert (pl is None)

		definitions = dict()
		finals = list()

		for k, v in defs.items():
			tokens = Tokenizer.tokenize(v)

			if (special := k.startswith(':')): k = k[1:]
			if (final := k.startswith('@')): k = k[1:]; finals.append(k)

			definitions[k] = cls.Definition.parse(name=k, tokens=tokens, special=special, final=final)

		return cls(definitions, finals)

@export
def load(file=DEFAULT_DEF):
	src = open(file).read()
	sldef = SlDef.build(src)
	return sldef

def main():
	sldef = load()

	res = tuple(sldef.definitions.values())

	print('\n\n'.join(map(str, res)))
	#print('\n\n'.join(map(repr, res)))
	print(f"\n# Finals: {', '.join(sldef.finals)}")

	check = SlDef.build('\n'.join(map(str, res))).definitions.values()

	print('\n\n'.join(map(repr, check)))

	for i, j in zip(map(str, res), map(str, check)):
		assert (i == j)

if (__name__ == '__main__'): exit(main())

# by Sdore, 2021
# slang.sdore.me
