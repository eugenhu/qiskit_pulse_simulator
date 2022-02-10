import sympy

from .symbols import VariableSymbol, OperatorSymbol, ChannelSymbol


__all__ = (
    'doit',
)


def doit(ast, *, scope=None):
    if scope is None:
        scope = {}
    
    return doit_funcs[ast[0]](*ast[1:], scope=scope)


def doit_sum(index: str, start: int, until: int, body: tuple, *, scope: dict) -> sympy.Basic:
    summands = []
    for i in range(start, until+1):
        scope[index] = i
        term = doit(body, scope=scope)
        summands.append(term)
    return sympy.Add(*summands)


def doit_add(*args, scope: dict) -> sympy.Basic:
    return sympy.Add(*[doit(arg, scope=scope) for arg in args])


def doit_minus(*args, scope: dict) -> sympy.Basic:
    assert len(args) == 2
    x, y = (doit(arg, scope=scope) for arg in args)
    return x - y


def doit_times(*args, scope: dict) -> sympy.Basic:
    return sympy.Mul(*[doit(arg, scope=scope) for arg in args])


def doit_divide(*args, scope: dict) -> sympy.Basic:
    assert len(args) == 2
    x, y = (doit(arg, scope=scope) for arg in args)
    return x/y


def doit_number(*args, scope: dict) -> float:
    assert len(args) == 1
    return args[0]


def doit_symbol(*args, scope: dict) -> sympy.Symbol:
    assert len(args) == 1, len(args)
    name = doit(args[0], scope=scope)
    if name[0].isupper():
        return OperatorSymbol(name)
    else:
        return VariableSymbol(name)


def doit_channel(*args, scope: dict) -> sympy.Symbol:
    assert len(args) == 1, len(args)
    return ChannelSymbol(doit(args[0], scope=scope))


def doit_bound(*args, scope: dict) -> float:
    assert len(args) == 1
    return scope[args[0]]


def doit_string_concat(*args, scope: dict) -> str:
    return ''.join(doit(arg, scope=scope) for arg in args)


def doit_string_math(*args, scope: dict) -> str:
    assert len(args) == 1, len(args)
    
    x = doit(args[0], scope=scope)
    
    if isinstance(x, int):
        return str(x)
    elif isinstance(x, float):
        if abs((x + .5)%1 - .5) < 1e-6:
            return str(int(x))
        else:
            raise ValueError(f"Cannot convert {x} to an integer string.")
    else:
        raise TypeError(type(x))


def doit_string_literal(*args, scope: dict) -> str:
    assert len(args) == 1, len(args)
    return args[0]


doit_funcs = {
    'sum': doit_sum,
    'add': doit_add,
    'minus': doit_minus,
    'times': doit_times,
    'divide': doit_divide,
    'number': doit_number,
    'symbol': doit_symbol,
    'channel': doit_channel,
    'bound': doit_bound,
    'string-concat': doit_string_concat,
    'string-math': doit_string_math,
    'string-literal': doit_string_literal,
}
