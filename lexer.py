#!/usr/bin/env python3
# Slang lexer

from . import sldef
from utils.nolog import *

@export
@singleton
class Lexer:
	def __init__(self):
		self.sldef = sldef.load()

	def parse_token(self, token, src):
		for i in token.choices:
			length = len(src)
			res = list()
			for j in i:
				j = j.strip()
				if (j.isidentifier()):
					try: res += self.parse_token(self.definitions[j], src)
					except SlParseError as ex: raise SlParseError(ex.length, length-len(src)) from ex
					continue
				t = j.strip("'").join("''").encode().decode('unicode_escape')
				if (not src.startswith(t)): raise SlParseError(len(t), length-len(src))
			return res
		else: raise SlParseError() # FIXME

	def read_token(self, src, *, lineno, offset, lineoff):
		(l, src), line = lstripcount(src[offset:], self.definitions.whitespace.charset), src
		offset += l
		if (src[:1] in self.definitions.codesep.literals): return (offset, None)
		err = (0, 0)
		for i in self.finals:
			try: t = self.parse_token(self.definitions[i], src)
			except SlParseError as ex: err = (ex.length, ex.char); continue
			return (offset+n, Token(ii, s, lineno=lineno, offset=offset+lineoff))
		else: raise SlSyntaxError("Invalid token", [None]*(lineno-1)+line.split('\n'), lineno=lineno, offset=offset+lineoff, length=err[0]+l, char=err[1])

	def parse_expr(self, src, *, lineno=1, lineoff=0):
		r = list()
		lines = src.count('\n')
		offset = int()
		continueln = False
		while (True):
			offset, tok = self.read_token(src, lineno=lines-src[offset:].count('\n')+lineno, offset=offset, lineoff=lineoff)
			if (tok is None):
				if (not continueln): break
				continueln = False
				offset += 1
				lineoff = -offset
				continue
			elif (continueln and tok.token[0] not in self.definitions.comment.literals): raise SlSyntaxError("Expected newline or comment after line continuation", src, lineno=lines-src[offset:].count('\n')+lineno, offset=tok.offset, length=tok.length)
			r.append(tok)
			if (tok.token[0] not in self.definitions.comment.literals): continueln = (tok.token in self.definitions.continuation.literals and tok.offset)
		return offset, r

	def parse_string(self, src, lnooff=0):
		src = src.rstrip()
		tl = list()
		lines = src.count('\n')+lnooff
		lineoff = int()
		while (src):
			offset, r = self.parse_expr(src, lineno=lines-src.count('\n')+1, lineoff=lineoff)
			lineoff += offset
			if (offset < len(src)):
				if (src[offset] == '\n'): lineoff = int()
				else: lineoff += 1
			src = src[offset+1:]
			tl.append(r)
		return tl

	@property
	def definitions(self):
		return self.sldef.definitions

	@property
	def finals(self):
		return self.sldef.finals

@export
class SlParseError(Exception):
	def __init__(self, length, char):
		self.length, self.char = length, char

# by Sdore, 2021-2022
#  slang.sdore.me
