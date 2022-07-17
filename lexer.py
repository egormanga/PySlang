#!/usr/bin/env python3
# PySlang lexer

from . import sldef
from .sldef import Format
from .exceptions import *
from utils.nolog import *

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
		return f"""\033[1;94m{self.typename.capitalize()}\033[0m \033[1m{self.name}\033[0m \033[2mat line {self.lineno}, offset {self.offset}, length {self.length}\033[0m: '{self.token}'"""

	@property
	def nlcnt(self):
		return int(self.token == '\n')

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
	def nlcnt(self):
		return sum(i.nlcnt for i in self.tokens)

@export
@singleton
class Lexer(Slots):
	sldef: ...

	def __init__(self):
		self.sldef = sldef.load()

	def lstripspace(self, s):
		return lstripcount(s, self.definitions.whitespace.format.charset)

	@functools.singledispatchmethod
	def _parse_token() -> [Expr]: ...

	@_parse_token.register
	def parse_token(self, token: Format.Reference, src, *, lineno, offset, was):
		if ((place := (lineno, offset, token.name)) in was):
			if ((*place, True) in was): raise SlSyntaxExpectedNothingError(tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=token.name)
			was |= {(*place, True)}
		tl = self.parse_token(self.sldef.definitions[token.name].format, src, lineno=lineno, offset=offset, was=was | {place})
		return [Expr(tl, name=token.name, lineno=lineno, offset=offset, length=(tl[-1].offset + tl[-1].length - tl[0].offset))]

	@_parse_token.register
	def parse_token(self, token: Format.Literal, src, *, lineno, offset, was):
		if (not src.startswith(token.literal)): raise SlSyntaxExpectedError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src))
		return [Token(token.literal, name=token.name, typename=token.typename, lineno=lineno, offset=offset, length=token.length)] # TODO: name

	@_parse_token.register
	def parse_token(self, token: Format.Pattern, src, *, lineno, offset, was):
		m = re.match(token.pattern, src)
		if (m is None): raise SlSyntaxExpectedError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src))
		try: t = m[1]
		except IndexError: t = m[0]
		return [Token(t, name=token.name, typename=token.typename, lineno=lineno, offset=offset, length=len(t))]

	@_parse_token.register
	def parse_token(self, token: Format.Optional, src, *, lineno, offset, was):
		try: return self.parse_token(token.token, src, lineno=lineno, offset=offset, was=was)
		except SlSyntaxError: return ()

	@_parse_token.register
	def parse_token(self, token: Format.ZeroOrMore, src, *, lineno, offset, was):
		res = list()

		while (True):
			try: tl = self.parse_token(token.token, src, lineno=lineno, offset=offset, was=was)
			except SlSyntaxError: break
			if (not tl): continue
			res += tl

			if (nlcnt := sum(i.nlcnt for i in tl)):
				src = src.split('\n', maxsplit=nlcnt)[-1]
				lineno += nlcnt
				offset = 0
				length = 0
			else: length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

		return res

	@_parse_token.register
	def parse_token(self, token: Format.OneOrMore, src, *, lineno, offset, was):
		res = list()
		errors = list()

		while (True):
			try: tl = self.parse_token(token.token, src, lineno=lineno, offset=offset, was=was)
			except SlSyntaxError as ex: errors.append(ex); break
			if (not tl): continue
			res += tl

			if (nlcnt := sum(i.nlcnt for i in tl)):
				src = src.split('\n', maxsplit=nlcnt)[-1]
				lineno += nlcnt
				offset = 0
				length = 0
			else: length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

		if (not res):
			if (errors): raise SlSyntaxMultiExpectedError.from_list(errors, usage=token.name)
			else: raise SlSyntaxExpectedMoreTokensError(token, lineno=lineno, offset=-1, length=0, usage=token.name)

		return res

	@_parse_token.register
	def parse_token(self, token: Format.Sequence, src, *, lineno, offset, was):
		res = list()

		for i in token.sequence:
			tl = self.parse_token(i, src, lineno=lineno, offset=offset, was=was)
			if (not tl): continue
			res += tl

			if (nlcnt := sum(i.nlcnt for i in tl)):
				src = src.split('\n', maxsplit=nlcnt)[-1]
				lineno += nlcnt
				offset = 0
				length = 0
			else: length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

		return res

	@_parse_token.register
	def parse_token(self, token: Format.Choice, src, *, lineno, offset, was):
		errors = list()

		for i in token.choices:
			e = None
			try: return self.parse_token(i, src, lineno=lineno, offset=offset, was=was)
			except SlSyntaxError as ex: e = ex; errors.append(ex)
			finally:
				if (e is None): dlog(i, src.join('«»'))
		else:
			if (errors): raise SlSyntaxMultiExpectedError.from_list(errors, usage=token.name)
			else: raise SlSyntaxExpectedOneOfError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=token.name)

	@_parse_token.register
	def parse_token(self, token: Format.Joint, src, *, lineno, offset, was):
		res = list()

		for i in token.sequence:
			tl = self.parse_token(i, src, lineno=lineno, offset=offset, was=was)
			if (not tl): continue
			res += tl

			if (nlcnt := sum(i.nlcnt for i in tl)):
				src = src.split('\n', maxsplit=nlcnt)[-1]
				lineno += nlcnt
				offset = 0
				length = 0
			else: length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

		length = (res[-1].offset + res[-1].length - res[0].offset) # TODO FIXME
		assert (length == sum(i.length for i in res))
		return [Token(str().join(i.token for i in res), name=res[-1].name if (len(res) == 1) else token.name, typename=res[-1].typename if (len(res) == 1) else token.typename, lineno=res[0].lineno, offset=res[0].offset, length=length)]

	parse_token = _parse_token; del _parse_token

	def parse_string(self, src, name='<string>', *, lnooff=0, offset=0):
		lineno = lnooff+1
		tl = self.parse_token(self.definitions.code.format, src.rstrip().join(('{\n', '\n}')), lineno=lineno-1, offset=offset, was=set())
		return Expr(tl, name=name, lineno=lineno, offset=offset, length=sum(i.length for i in tl))

	@property
	def definitions(self):
		return self.sldef.definitions

# by Sdore, 2021-22
#  slang.sdore.me
