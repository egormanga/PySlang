#!/usr/bin/env python3
# PySlang

from .ast import AST
from .lexer import Lexer
from .exceptions import *
from utils import *

@export
def compile_source(src, filename='<string>', *, compiler, optimize=0):
	try:
		print(f"Source: {{\n{'\n'.join(f'\033[1;2;40m{ii:>6} \033[m{i}' for ii, i in enumerate(S(src).indent().split('\n'), 1))}\n}}\n")

		st = Lexer.parse_string(src)
		print("Source tree:", st, sep='\n\n', end='\n\n')

		ast = AST.build(st, scope=filename.join('""'))
		print(f"Abstract syntax tree: {{\n{Sstr(ast).indent()}\n}}\n")

		if (optimize):
			ast.optimize(level=optimize)
			print(f"Optimized tree: {{\n{Sstr(ast).indent()}\n}}\n")

		ns = ast.validate()

		code = compiler.compile_ast(ast, ns, filename=filename)

		print("Compiled.\n")
	except SlException as ex:
		if (not ex.srclines): ex.srclines = src.split('\n')
		sys.exit(str(ex))

	return code

@apmain
@aparg('file', metavar='<file.sl>', type=argparse.FileType('r'))
@aparg('-o', metavar='output', dest='output', type=argparse.FileType('wb'))
@aparg('-c', metavar='compiler', dest='compiler', default='sbc')
@aparg('-O', metavar='optlevel', dest='optlevel', help="Code optimization level", type=int)
def main(cargs):
	filename = cargs.file.name

	if (cargs.output is None and not filename.rpartition('.')[0]):
		argparser.add_argument('-o', dest='output', type=argparse.FileType('wb'), required=True)
		cargs = argparser.parse_args()

	compiler = importlib.import_module(f".compilers.{cargs.compiler}", package=__package__).compiler

	with cargs.file as f:
		src = f.read()

	code = compile_source(src, filename=filename, compiler=compiler, optimize=cargs.optlevel)

	with (cargs.output or open((os.path.splitext(os.path.basename(filename))[0] + compiler.ext), 'wb')) as f:
		f.write(code)

# by Sdore, 2021-24
#  slang.sdore.me
