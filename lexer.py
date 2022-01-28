#!/usr/bin/env python3
# PySlang lexer

from . import sldef
from .ast import *
from .sldef import Format
from utils.nolog import *

def tok(s):
	try: return first(re.finditer(r'(.+?)\b', s))[1].join("''")
	except StopIteration: return 'nothing'

def toklen(s):
	try: return first(re.finditer(r'(?<=.)\b', s)).end()
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

@export
@singleton
class Lexer:
	def __init__(self):
		self.sldef = sldef.load()

	def lstripspace(self, s):
		return lstripcount(s, self.definitions.whitespace.format.charset)

	@dispatch
	def parse_token(self, src, token: Format.Reference, *, lineno, offset, was):
		if ((place := (lineno, offset, token.name)) in was):
			if ((*place, True) in was): raise SlSyntaxExpectedNothingError(tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=token.name)
			was |= {(*place, True)}
		tl = self.parse_token(src, self.sldef.definitions[token.name].format, lineno=lineno, offset=offset, was=was | {place})
		return [Expr(tl, name=token.name, lineno=lineno, offset=offset, length=(tl[-1].offset + tl[-1].length - tl[0].offset))]

	@dispatch
	def parse_token(self, src, token: Format.Literal, *, lineno, offset, was):
		if (not src.startswith(token.literal)): raise SlSyntaxExpectedError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src))
		return [Token(token.literal, name=token.name, typename=token.typename, lineno=lineno, offset=offset, length=token.length)] # TODO: name

	@dispatch
	def parse_token(self, src, token: Format.Pattern, *, lineno, offset, was):
		m = re.match(token.pattern, src)
		if (m is None): raise SlSyntaxExpectedError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src))
		try: t = m[1]
		except IndexError: t = m[0]
		return [Token(t, name=token.name, typename=token.typename, lineno=lineno, offset=offset, length=len(t))]

	@dispatch
	def parse_token(self, src, token: Format.Optional, *, lineno, offset, was):
		try: return self.parse_token(src, token.token, lineno=lineno, offset=offset, was=was)
		except SlSyntaxError: return ()

	@dispatch
	def parse_token(self, src, token: Format.ZeroOrMore, *, lineno, offset, was):
		res = list()

		while (True):
			try: tl = self.parse_token(src, token.token, lineno=lineno, offset=offset, was=was)
			except SlSyntaxError as ex: break
			if (not tl): continue
			res += tl

			length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

		return res

	@dispatch
	def parse_token(self, src, token: Format.OneOrMore, *, lineno, offset, was):
		res = list()

		while (True):
			try: tl = self.parse_token(src, token.token, lineno=lineno, offset=offset, was=was)
			except SlSyntaxError: break
			if (not tl): continue
			res += tl

			length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

		if (not res): raise SlSyntaxExpectedMoreTokensError(token, lineno=lineno, offset=offset, length=0, usage=token.name)

		return res

	@dispatch
	def parse_token(self, src, token: Format.Sequence, *, lineno, offset, was):
		res = list()

		for i in token.sequence:
			tl = self.parse_token(src, i, lineno=lineno, offset=offset, was=was)
			if (not tl): continue
			res += tl

			length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			#isid = src[length-1:length].isidentifier()
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

			#if (src and (not l or src[0].isidentifier() == isid)): raise SlSyntaxExpectedNothingError(tok(src), lineno=ltline, offset=offset, length=, usage=) # TODO FIXME

		return res

	@dispatch
	def parse_token(self, src, token: Format.Choice, *, lineno, offset, was):
		errors = list()

		for i in token.choices:
			try: return self.parse_token(src, i, lineno=lineno, offset=offset, was=was)
			except SlSyntaxError as ex:
				if (isinstance(ex.expected, Token) and ex.expected.final): errors.append(ex)
		else:
			if (errors): raise SlSyntaxMultiExpectedError.from_list(errors, usage=token.name)
			else: raise SlSyntaxExpectedOneOfError(token, tok(src), lineno=lineno, offset=offset, length=toklen(src), usage=token.name)

	@dispatch
	def parse_token(self, src, token: Format.Joint, *, lineno, offset, was):
		res = list()

		for i in token.sequence:
			tl = self.parse_token(src, i, lineno=lineno, offset=offset, was=was)
			if (not tl): continue
			res += tl

			length = (tl[-1].offset + tl[-1].length - tl[0].offset)
			ws, src = self.lstripspace(src[length:])
			offset += (length + ws)

		length = (res[-1].offset + res[-1].length - res[0].offset)
		assert (length == sum(i.length for i in res))
		return [Token(str().join(i.token for i in res), name=res[-1].name if (len(res) == 1) else token.name, typename=res[-1].typename if (len(res) == 1) else token.typename, lineno=res[0].lineno, offset=res[0].offset, length=length)]

	def parse_expr(self, src, *, lineno=1, offset=0):
		line = src
		l, src = self.lstripspace(src[offset:])
		offset += l

		if (src[:1] in self.definitions.codesep.format.literals): return (offset, None)

		errors = list()

		for i in self.finals:
			try:
				tl = self.parse_token(src, self.definitions[i].format, lineno=lineno, offset=offset, was=set())

				length = (tl[-1].offset + tl[-1].length - tl[0].offset)
				ws, _ = self.lstripspace(src[length:])
				end = (offset + length + ws)

				#if (leftover := src[length+ws:].strip()):
				#	if (leftover and (len(leftover) > 1 or leftover not in self.definitions.codesep.format.literals)): # TODO: comments
				#		errors.append(SlSyntaxExpectedNothingError(tok(leftover), lineno=lineno, offset=end))
				#		continue
			except SlSyntaxError as ex:
				if (ex.usage is None): ex.usage = i
				errors.append(ex)
				continue
			#except SlSyntaxError: continue
			return (end, Expr(tl, name=i, lineno=lineno, offset=offset, length=end-offset))
		else: raise SlSyntaxError("Invalid token", lineno=lineno, offset=offset, length=toklen(src[offset:])) from SlSyntaxMultiExpectedError.from_list(errors) if (errors) else None

	def parse_string(self, src, *, lnooff=0):
		src = src.rstrip()

		exprs = list()
		lines = src.count('\n')
		lineno = lnooff+1
		offset = int()
		continueln = bool()

		while (src):
			offset, expr = self.parse_expr(src, lineno=lineno, offset=offset)

			#if (expr is None):
			#	if (not continueln): break
			#	continueln = False
			#	offset += 1
			#	lineoff = -offset
			#	continue

			#for tok in expr.tokens:
			#	while (isinstance(tok, Expr)):
			#		tok = tok.tokens[-1]
			#	if (continueln and tok.token[0] != self.definitions.comment.format.literal): raise SlSyntaxError("Expected newline or comment after line continuation", src, lineno=lines-src[offset:].count('\n')+lineno, offset=tok.offset, length=tok.length)
			#	if (tok.token[0] != self.definitions.comment.format.literal): continueln = (tok.token == self.definitions.continuation.format.literal and tok.offset)

			while (src[offset:offset+1] == '\n'):
				lineoff, offset = offset, 0
				src = src[lineoff+1:]
				lineno += 1

			src = src[offset:]
			exprs.append(expr)

		return exprs

	@property
	def definitions(self):
		return self.sldef.definitions

	@property
	def finals(self):
		return self.sldef.finals

# by Sdore, 2021-2022
#  slang.sdore.me
