#!/usr/bin/env python3
# PySlang compilers

import abc
from ..exceptions import SlException

class Compiler(abc.ABC):
	ext = ''

	@abc.abstractclassmethod
	def compile_ast(cls, ast):
		pass

class SlCompilationError(SlException):
	desc: ...

	def __init__(self, desc, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.desc = desc

	def __exline__(self):
		return f"Compilation error: {self.desc}"

# by Sdore, 2022-24
#  slang.sdore.me
