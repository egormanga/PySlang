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

		if (s): raise WTFException(s)
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
		name: str
		final: bool

		def __init__(self, token, *, name, final):
			self.token, self.name, self.final = token, name, final

		def __repr__(self):
			return f"<\033[1;92m{self.__class__.__name__}\033[0m:" + (f"\n{S(repr(self.token)).indent()}\n" if (not isinstance(self.token, str)) else f" \033[95m{repr(self.token)}\033[0m") + ">"

		def flatten(self):
			self.token = self.token.flatten()
			return self

		@property
		def typename(self):
			return self.__class__.__name__.casefold()

	class StringToken(Token):
		token: str

		def __init__(self, token, *, final):
			self.token, self.final = token, final

		def __str__(self):
			return self.token

		def flatten(self):
			return self

		@property
		def name(self):
			return str(self)

	class Reference(StringToken):
		final = False

		def __init__(self, token: str):
			self.token = token

	class Literal(StringToken):
		@dispatch
		def __init__(self, token: str, *, final):
			super().__init__(token, final=final)

		@dispatch
		def __init__(self, token: lambda token: hasattr(token, 'token'), *, final):
			self.__init__(token.token, final=final)

		def __str__(self):
			if ('#' not in self.token): return repr(self.token)
			return r' \# '.join(repr(i) if (i) else '' for i in self.token.split('#')).strip().join('[]')

		@property
		def name(self):
			return repr(self.token)

		@property
		def literal(self):
			return self.token

		@property
		def length(self):
			return len(self.token)

	class Escape(Literal):
		_table = {r"\x%02x" % rf"\{i}".encode().decode('unicode_escape').encode()[0]: rf"\{i}" for i in 'abfnrtv'}

		token: str

		def __init__(self, token: str, *, final):
			super().__init__(token.encode().decode('unicode_escape') if (token != r'\#') else '#', final=final)

		def __str__(self):
			s = ('\\' + (self.token.encode('unicode_escape').decode().lstrip('\\') or '\\'))
			return self._table.get(s, s)

	class Pattern(StringToken):
		def __str__(self):
			return self.token.join('//')

		@property
		def pattern(self):
			return self.token

	class Optional(Token):
		def __str__(self):
			return f"{str(self.token).join('()') if (isinstance(self.token, Format.TokenList)) else self.token}?"

	class ZeroOrMore(Token):
		def __str__(self):
			return f"{str(self.token).join('()') if (isinstance(self.token, Format.TokenList)) else self.token}*"

	class OneOrMore(Token):
		def __str__(self):
			return f"{str(self.token).join('()') if (isinstance(self.token, Format.TokenList)) else self.token}+"

	class TokenList(Token):
		tokens: list

		@autocast
		def __init__(self, tokens: list, *, name, final):
			self.tokens, self.name, self.final = tokens, name, final

		def __repr__(self):
			return f"<\033[1;91m{self.__class__.__name__}\033[0m: [\n{S(NL).join(map(repr, self.tokens)).indent()}\n]>"

		def append(self, token):
			if (self.tokens and isinstance(self.tokens[-1], Format.TokenList)): self.tokens[-1].tokens.append(token)
			else: self.tokens.append(token)

		def pop(self):
			if (self.tokens and isinstance(self.tokens[-1], Format.TokenList)): return self.tokens[-1].tokens.pop()
			if (not self.tokens or not self.tokens[-1]): raise WTFException(self)
			return self.tokens.pop()

		def flatten(self):
			for ii, i in enumerate(self.tokens):
				self.tokens[ii] = i.flatten()

			return (self.tokens[0] if (len(self.tokens) == 1) else self)

		@property
		def literals(self):
			if (not all(isinstance(i, Format.Literal) for i in self.tokens)): raise AttributeError('literals')
			return tuple(i.token for i in self.tokens)

	class TokenSequence(TokenList):
		def __str__(self):
			return ' '.join(str(i).join('()') if (isinstance(i, Format.Choice)) else str(i) for i in self.tokens)

		@property
		def sequence(self):
			return self.tokens

	class Sequence(TokenSequence):
		pass

	class Choice(TokenList):
		def __str__(self):
			return ' | '.join(str(i).join('()') if (isinstance(i, Format.Choice)) else str(i) for i in self.tokens)

		@property
		def choices(self):
			return self.tokens

		@property
		def charset(self):
			if (not all(isinstance(i, Format.Literal) and len(i.token) == 1 for i in self.tokens)): raise AttributeError('charset')
			return str().join(i.token for i in self.tokens)

	class Joint(TokenSequence):
		def __str__(self):
			return super().__str__().join('[]')

		def append(self, token):
			self.tokens.append(token)

		def flatten(self):
			tokens = list()
			for i in self.tokens:
				if (tokens and isinstance(tokens[-1], Format.Literal) and isinstance(i, Format.Literal)):
					tokens[-1] = Format.Literal(tokens[-1].token + i.token, final=tokens[-1].final)
				else: tokens.append(i)
			self.tokens = tokens

			return super().flatten()

	name: str
	final: bool
	format: '# TokenList'

	@dispatch
	def __init__(self, name, final, format: TokenList):
		self.name, self.final, self.format = name, final, format

	def __repr__(self):
		return f"<\033[1;93m{self.__class__.__name__}\033[0m '{self.name}':\n{S(repr(self.format)).indent()}\n>"

	def __str__(self):
		return str(self.format)

	def flatten(self):
		self.format = self.format.flatten()

	@classmethod
	def parse(cls, name, final, tokens, *, _group=False):
		if (_group == '['): format, end = cls.Joint([], name=name, final=final), ']'
		else: format, end = cls.Choice([cls.Sequence([], name=name, final=final)], name=name, final=final), ')'

		while (tokens):
			token = tokens.pop(0)
			if (token == end): break

			if (token in ('(', '[')): token = cls.parse(name, final, tokens, _group=token).format
			elif (token == '|'):
				if (not isinstance(format, cls.Choice) or format.tokens[-1].tokens and format.tokens[-1].tokens[-1] == '|'): raise WTFException(token)
				format.tokens.append(cls.Sequence([], name=name, final=final))
				continue
			elif (token == '+'): token = cls.OneOrMore(format.pop(), name=name, final=final)
			elif (token == '*'): token = cls.ZeroOrMore(format.pop(), name=name, final=final)
			elif (token == '?'): token = cls.Optional(format.pop(), name=name, final=final)
			elif (m := re.fullmatch(r'/(.+)/', token)): token = cls.Pattern(m[1], final=final)
			elif (m := re.fullmatch(r'\\.+', token)): token = cls.Escape(m[0], final=final)
			elif (m == r'\#'): token = cls.Literal('#', final=final)
			elif (m := re.fullmatch(r'''(['"])(.+)\1''', token)): token = cls.Literal(m[2], final=final)
			else: token = cls.Reference(token)

			format.append(token)

		return cls(name, final, format)

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
		def parse(cls, *, name, tokens, special, final, flatten=True):
			format = Format.parse(name, final, tokens)
			if (flatten): format.flatten()
			return cls(name, format.format, special=special, final=final)

	def __init__(self, definitions=None, finals=None):
		if (definitions is not None): self.definitions |= definitions
		if (finals is not None): self.finals += finals

	definitions: AttrDict
	finals: list

	@classmethod
	def parse(cls, src):
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

			if (name in defs): raise WTFException(name)

			defs[name] = value.strip()

		if (pl is not None): raise WTFException(pl)

		return defs

	@classmethod
	def build(cls, src):
		definitions = dict()
		finals = list()

		for k, v in cls.parse(src).items():
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

@apmain
@aparg('file', metavar='file.sld', nargs='?', default=DEFAULT_DEF)
def main(cargs):
	sldef = load(cargs.file)

	res = tuple(sldef.definitions.values())

	print('\n\n'.join(map(str, res)))
	print('\n')
	print('\n\n'.join(map(repr, res)))
	print(f"\n\033[2m# Finals: {', '.join(sldef.finals)}\033[0m")

	for i, j in zip((f"{k}: {v}" for k, v in SlDef.parse(open(cargs.file).read()).items()), map(str, res)):
		if (i != j): sys.exit(f"\n\033[1;91mMismatch!\033[0m\n\033[92mExpected:\033[0m  {i}\n\033[93mGot:\033[0m       {j}")

	check = SlDef.build('\n'.join(map(str, res))).definitions.values()

	#print('\n\n'.join(map(repr, check)))

	for i, j in zip(map(str, res), map(str, check)):
		if (i != j): raise Exception(i, j)

if (__name__ == '__main__'): exit(main())

# by Sdore, 2021-22
#  slang.sdore.me
