#!/usr/bin/env python3
# PySlang

from .ast import AST
from .lexer import Lexer
from .exceptions import *
from utils import *

@export
def compile(src, filename='<string>', *, compiler, optimize=0):
	try:
		print(f"Source: {{\n{S(src).indent()}\n}}\n")

		st = Lexer.parse_string(src)
		#print("Source tree:", *st, sep='\n\n', end='\n\n')

		ast = AST.build(st, scope=filename.join('""'))
		print(f"Abstract syntax tree: {ast}\n")

		#print(f"Nodes: {pformat(list(walk_ast_nodes(ast)))}\n")

		#if (optimize):
		#	optimize_ast(ast, validate_ast(ast), optimize)
		#	print(f"Optimized: {ast.code}\n")
		#	#print(f"Optimized Nodes: {pformat(list(walk_ast_nodes(ast)))}\n")

		#ns = validate_ast(ast)

		#code = compiler.compile_ast(ast, ns, filename=filename)
		#print("Compiled.\n")
	except SlException as ex:
		if (not ex.srclines): ex.srclines = src.split('\n')
		sys.exit(str(ex))
	else: return code

@apmain
@aparg('file', metavar='<file.sl>', type=argparse.FileType('r'))
@aparg('-o', metavar='output', dest='output')
@aparg('-f', metavar='compiler', dest='compiler', default='sbc')#required=True)
@aparg('-O', metavar='level', help='Code optimization level', type=int)#, default=DEFAULT_OLEVEL)
def main(cargs):
	filename = cargs.file.name

	if (cargs.output is None and not filename.rpartition('.')[0]):
		argparser.add_argument('-o', dest='output', required=True)
		cargs = argparser.parse_args()

	compiler = None#importlib.import_module('.compilers.'+cargs.compiler, package=__package__).compiler

	src = cargs.file.read()
	code = compile(src, filename=filename, compiler=compiler, optimize=cargs.O)

	#with open(cargs.output or filename.rpartition('.')[0]+compiler.ext, 'wb') as f:
	#	f.write(code)

# by Sdore, 2021-22
#  slang.sdore.me
