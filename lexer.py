#!/usr/bin/env python3
# PySlang lexer

from .sldef import Format, SlDef
from .exceptions import *
from utils.nolog import *

class singledispatchmethod(functools.singledispatchmethod):
	def __get__(self, obj, cls=None):
		return suppress_tb(super().__get__(obj, cls=cls))

def tok(s):
	try: return first(re.finditer(r'(.+?)(?:\b|$)', s))[1].join("''")
	except StopIteration: return 'nothing'

def toklen(s):
	try: return first(re.finditer(r'(?<=.)(?:\b|$)', s)).end()
	except StopIteration: return 0

class Token(Slots):
	token: str
	name: str
	typename: str
	lineno: int
	offset: int
	length: int

	def __init__(self, token, *, name, typename, lineno, offset, length):
		self.token, self.name, self.typename, self.lineno, self.offset, self.length = token, name, typename, lineno, offset, length

	def __repr__(self):
		return f"""<Token {self.name} of type {self.typename}: '{self.token}' at line {self.lineno}, offset {self.offset}, length {self.length}>"""

	def __str__(self):
		return f"""\033[1;94m{self.typename.capitalize()}\033[0m \033[1m{self.name}\033[0m \033[2mat line {self.lineno}, offset {self.offset}, length {self.length}\033[0m: {repr(self.token)}"""

	@property
	def lastpos(self):
		return (self.lineno, self.offset + self.length)

class Expr(Token):
	tokens: list

	def __init__(self, tokens, *, name, lineno, offset, length):
		self.tokens, self.name, self.lineno, self.offset, self.length = tokens, name, lineno, offset, length

	def __repr__(self):
		return f"<Expr {self.name}: {self.tokens} at line {self.lineno}, offset {self.offset}, length {self.length}>"

	def __str__(self):
		return f"\033[1;92mExpr\033[0m \033[1m{self.name}\033[0m \033[2mat line {self.lineno}, offset {self.offset}, length {self.length}\033[0m "+S('\n\n').join(map(str, self.tokens)).indent().join(('{\n', '\n}'))

	@property
	def token(self):
		assert (len(self.tokens) == 1)
		return self.tokens[0].token

	@property
	def typename(self):
		assert (len(self.tokens) == 1)
		return self.tokens[0].typename

	@property
	def lastpos(self):
		return max(i.lastpos for i in self.tokens)

@export
class Lexer:
	def __init__(self):
		self._parse_token_cache_literal = dict()
		self._parse_token_cache_pattern = dict()

	def lstripspace(self, sldef: SlDef, s: str):
		return lstripcount(s, sldef.definitions.whitespace.format.charset)

	def lstripcont(self, sldef: SlDef, s: str):
		return (self.lstripspace(sldef, i) for i in s.removeprefix(sldef.definitions.continuation.format.token).split('\n', maxsplit=1))

	@singledispatchmethod
	def _parse_token(self, token: Format.Token, sldef: SlDef, src: str, *, lineno: int, offset: int, usage: str, was: set) -> [Expr]: ...

	@_parse_token.register
	def parse_token(self, token: Format.Reference, sldef: SlDef, src, *, lineno, offset, usage, was):
		if ((location := (lineno, offset, token.name)) in was):
			if ((*location, None) in was): raise SlSyntaxExpectedNothingError(tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=token.name)
			was |= {(*location, None)}

		ns, _, name = token.name.rpartition('.')
		if (ns): sldef = sldef.definitions[ns]
		tl = self.parse_token(sldef.definitions[name].format, sldef, src, lineno=lineno, offset=offset, usage=token.name, was=(was | {location}))
		if (not tl): raise SlSyntaxExpectedMoreTokensError(token, lineno=lineno, offset=-1, length=0, usage=token.name)

		assert (tl[0].offset == offset)

		oldlineno, oldoffset = lineno, offset
		lineno, offset = max(map(operator.attrgetter('lastpos'), tl))
		nlcnt = (lineno - oldlineno)
		lines = src.split('\n', maxsplit=nlcnt)
		length = (offset + sum(map(len, lines[:-1])) + nlcnt - tl[0].offset)

		#dlog(f"{token} for {usage} at {lineno, offset} {length=}:", *map(str, tl),
		#	length,
		#	repr(src[length:]),
		#sep='\n', end='\n\n')

		return [Expr(tl, name=token.name, lineno=oldlineno, offset=oldoffset, length=length)]

	@_parse_token.register
	def parse_token(self, token: Format.Literal, sldef: SlDef, src, *, lineno, offset, usage, was):
		try: return [self._parse_token_cache_literal[token.literal, lineno, offset]]
		except KeyError: pass

		if (not src.startswith(token.literal)): raise SlSyntaxExpectedError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=usage)
		assert (token.length == len(token.literal))

		r = self._parse_token_cache_literal[token.literal, lineno, offset] = Token(token.literal, name=token.name, typename=token.typename, lineno=lineno, offset=offset, length=token.length) # TODO: name
		return [r]

	@_parse_token.register
	def parse_token(self, token: Format.Pattern, sldef: SlDef, src, *, lineno, offset, usage, was):
		try: return [self._parse_token_cache_pattern[token.pattern, lineno, offset]]
		except KeyError: pass

		m = re.match(token.pattern, src, flags=(re.M | re.S | re.X))
		if (m is None): raise SlSyntaxExpectedError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=usage)
		try: t = m[1]
		except IndexError: t = m[0]

		r = self._parse_token_cache_pattern[token.pattern, lineno, offset] = Token(t, name=token.name, typename=token.typename, lineno=lineno, offset=offset, length=len(t))
		return [r]

	@_parse_token.register
	def parse_token(self, token: Format.Optional, sldef: SlDef, src, *, lineno, offset, usage, was):
		try: return self.parse_token(token.token, sldef, src, lineno=lineno, offset=offset, usage=usage, was=was)
		except SlSyntaxExpectedNothingError: raise
		except SlSyntaxError: return []

	def _parse_token_list(self, token: Format.TokenList, sldef: SlDef, src, *, lineno, offset, usage, was, _empty=False, _full=False, _stripspace=True):
		res = list()
		errors = list()

		for ii, i in enumerate((token.sequence if (isinstance(token, Format.TokenSequence)) else itertools.repeat(token.token))):
			if (not isinstance(token, Format.TokenSequence) and (_empty or res) and (not src or src.isspace())): break

			try: tl = self.parse_token(i, sldef, src, lineno=lineno, offset=offset, usage=usage, was=was)
			except SlSyntaxExpectedNothingError:
				if (isinstance(token, Format.TokenSequence)): break
				else: raise
			except SlSyntaxError as ex:
				if (isinstance(token, Format.TokenSequence)): raise
				errors.append(ex)
				break
			else:
				if (not tl): continue
				res += tl

			assert (tl[0].offset == offset)

			oldlineno, oldoffset = lineno, offset
			lineno, offset = max(map(operator.attrgetter('lastpos'), tl))
			nlcnt = (lineno - oldlineno)
			src = last(src.split('\n', maxsplit=nlcnt))
			loff = (oldoffset if (nlcnt == 0) else 0)
			srcoff = (offset - loff)
			 # - max((i.offset+i.lineno for i in tl if i.lineno == lineno), default=0))

			src_ = src
			while (offset > (loff + len(first(l := src_.partition('\n'))))):
				src_ = l[2]
				lineno += 1
				offset -= (loff + len(l[0] + l[1]))
				loff = 0

			#dlog(f"[…] {i} for {usage}:", *map(str, tl),
			#	max(map(operator.attrgetter('lastpos'), tl)),
			#	(oldlineno, oldoffset),
			#	nlcnt,
			#	(lineno, offset),
			#	repr(src),
			#	srcoff,
			#	repr(src[srcoff:]),
			#	self.lstripspace(sldef, src[srcoff:]),
			#sep='\n', end='\n\n')

			assert (0 <= srcoff <= len(src))
			if (ii or _stripspace):
				ws, src = self.lstripspace(sldef, src[srcoff:])
				offset += ws
				assert (not src[:1].replace('\n', '').isspace())
			else: src = src[srcoff:]

			if (src.startswith(sldef.definitions.continuation.format.token)):
				(ws, comment), (offset, src) = self.lstripcont(sldef, src)
				if (comment): tl += self.parse_token(sldef.definitions.linecomment.format, sldef, comment, lineno=lineno, offset=ws, usage=usage, was=was)
				lineno += 1
				assert (not src[:1].replace('\n', '').isspace())

		assert (not src[:1].replace('\n', '').isspace())

		if (not _empty and not res or _full and src and not src.isspace()):
			if (errors): raise SlSyntaxMultiExpectedError.from_list(errors, usage=usage)
			else: raise SlSyntaxExpectedMoreTokensError(token, lineno=lineno, offset=-1, length=0, usage=usage)
		#elif (errors): dlogexception(SlSyntaxMultiExpectedError.from_list(errors, usage=usage), extra=_empty, end='\n\n')

		return res

	@_parse_token.register
	@suppress_tb
	def parse_token(self, token: Format.ZeroOrMore, sldef: SlDef, src, *, lineno, offset, usage, was, _full=False):
		return self._parse_token_list(token, sldef, src, lineno=lineno, offset=offset, usage=usage, was=was, _empty=True, _full=_full)

	@_parse_token.register
	@suppress_tb
	def parse_token(self, token: Format.OneOrMore, sldef: SlDef, src, *, lineno, offset, usage, was):
		return self._parse_token_list(token, sldef, src, lineno=lineno, offset=offset, usage=usage, was=was, _empty=False)

	@_parse_token.register
	def parse_token(self, token: Format.Sequence, sldef: SlDef, src, *, lineno, offset, usage, was):
		return self._parse_token_list(token, sldef, src, lineno=lineno, offset=offset, usage=usage, was=was)

	@_parse_token.register
	def parse_token(self, token: Format.Choice, sldef: SlDef, src, *, lineno, offset, usage, was):
		errors = list()

		for i in token.choices:
			try: return self.parse_token(i, sldef, src, lineno=lineno, offset=offset, usage=usage, was=was)
			except SlSyntaxError as ex: errors.append(ex)
		else:
			#if (errors): raise SlSyntaxMultiExpectedError.from_list(errors, usage=usage)
			if (errors): raise (first((i for i in errors if i.usage == 'block'), default=None) or SlSyntaxMultiExpectedError.from_list(errors, usage=usage))
			else: raise SlSyntaxExpectedOneOfError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=usage)

	@_parse_token.register
	def parse_token(self, token: Format.Joint, sldef: SlDef, src, *, lineno, offset, usage, was):
		res = self._parse_token_list(token, sldef, src, lineno=lineno, offset=offset, usage=usage, was=was, _stripspace=False)
		return [Token(str().join(i.token for i in res), name=(res[-1].name if (len(res) == 1) else token.name), typename=(res[-1].typename if (len(res) == 1) else token.typename), lineno=res[0].lineno, offset=res[0].offset, length=sum(i.length for i in res))]

	parse_token = _parse_token; del _parse_token

	def parse_string(self, sldef: SlDef, src: str, name: str = '<string>', *, lnooff: int = 0, offset: int = 0):
		lineno = (lnooff + 1)
		token = sldef.definitions.code.format
		tl = self.parse_token(token, sldef, src.rstrip('\n')+'\n', lineno=lineno, offset=offset, usage=token.name, was=frozenset(), _full=True)

		#if (nlcnt := sum(i.nlcnt for i in tl)): length = (sum((l[-1].offset + l[-1].length) for k, v in itertools.groupby(tl, key=operator.attrgetter('lineno')) if (l := tuple(v))) - tl[0].offset)
		#else:
		length = ((tl[-1].offset + tl[-1].length - tl[0].offset) if (tl) else 0)

		#if ((leftover := src[length:]) and not leftover.isspace()): raise SlSyntaxExpectedNothingError(tok(leftover), lineno=(lineno + nlcnt), offset=(tl[-1].offset + tl[-1].length if (tl[-1].lineno == lineno + nlcnt) else 0), length=toklen(leftover))
		return Expr(tl, name=name, lineno=lineno, offset=offset, length=length)

# by Sdore, 2021-24
#  slang.sdore.me
