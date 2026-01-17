#!/usr/bin/env python3
# PySlang

from . import sldef
from .ast import AST
from .compilers import Compiler
from .exceptions import SlException, source_context
from .lexer import Lexer
from utils import *

@export
def compile_source(src, filename='<string>', *, compiler, optimize=0):
	print(f"Source: {{\n{'\n'.join(f'\033[1;2;40m{ii:>6} \033[m\t{i}' for ii, i in enumerate(src.splitlines(), 1))}\n}}\n")

	with source_context(src, filename):
		st = Lexer().parse_string(sldef.load(), src)
		#print("Source tree:", st, sep='\n\n', end='\n\n')

		ast = AST.build(st, scope=filename.join('""'))
		ast_lines = {k: list(v) for k, v in itertools.groupby(ast.code.statement, operator.attrgetter('lineno'))}
		print(f"Abstract syntax tree: {{\n{'\n'.join(f'\033[1;2;40m{i:>6} \033[m{S('\n').join(ast_lines.get(i, ())).indent()}' for i in range(1, max(ast_lines)+1))}\n}}\n")

		ns = ast.analyze()
		ast.validate(ns)

		if (optimize):
			ast.optimize(ns, level=optimize)
			print(f"Optimized tree: {{\n{Sstr(ast).indent()}\n}}\n")
			ast.validate(ns)

		code = compiler.compile_ast(ast, ns, filename=filename)

	print("Compiled.\n")
	return code

@apmain
@aparg('file', metavar='<file.sl>', type=argparse.FileType('r'))
@aparg('-o', metavar='output', dest='output')
@aparg('-c', metavar='compiler', dest='compiler', default='sbc')
@aparg('-O', metavar='optlevel', dest='optlevel', help="Code optimization level", type=int)
def main(cargs):
	filename = cargs.file.name

	if (cargs.output is None and not filename.rpartition('.')[0]):
		argparser.add_argument('-o', dest='output', required=True)
		cargs = argparser.parse_args()

	m = importlib.import_module(f".compilers.{cargs.compiler}", package=__package__)
	compiler = only(v for k, v in vars(m).items() if k in m.__all__ and isinstance(v, type) and issubclass(v, Compiler))

	with cargs.file as f:
		src = f.read()

	try: code = compile_source(src, filename=filename, compiler=compiler, optimize=cargs.optlevel)
	except SlException as ex: sys.exit(str(ex))

	output = (cargs.output or '')
	if (not output or os.path.isdir(output)): output = os.path.join(output, (os.path.splitext(os.path.basename(filename))[0] + compiler.ext))

	with open(output, 'wb') as f:
		f.write(code)

# by Sdore, 2021-26
#  slang.sdore.me
