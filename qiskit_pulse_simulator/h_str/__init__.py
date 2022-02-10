import sympy

from .doit import doit
from .symbols import VariableSymbol, OperatorSymbol, ChannelSymbol
from .parser import parser


def sympify_h_str(h_str: list) -> sympy.Add:
    ast = map(parser.parse, h_str)
    return sympy.Add(*map(doit, ast))
