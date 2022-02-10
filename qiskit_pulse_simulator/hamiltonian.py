from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from operator import itemgetter
import re
from typing import Sequence, Tuple

import qutip
from qutip import Qobj
from sympy import lambdify

from .h_str import ChannelSymbol, OperatorSymbol, VariableSymbol, sympify_h_str


__all__ = (
    'HamiltonianModel',
)


@dataclass
class HamiltonianModel:
    h0: Qobj
    hc: dict
    c_ops: list
    dims: list

    def transform(self, new_basis) -> HamiltonianModel:
        P = Qobj(new_basis, dims=[self.dims, self.dims])

        h0 = P.dag() * self.h0 * P

        hc = {}
        for ch, op in self.hc.items():
            hc[ch] = P.dag() * op * P

        c_ops = []
        if self.c_ops is not None:
            for x in self.c_ops:
                if isinstance(x, Qobj):
                    c_ops.append(P.dag() * x * P)
                elif isinstance(x, list):
                    c_ops.append([x[0], P.dag() * x[1] * P])
                else:
                    raise TypeError(type(x))

        return HamiltonianModel(h0, hc, c_ops, self.dims)

    def __mul__(self, factor: float) -> HamiltonianModel:
        h0 = factor * self.h0

        hc = {}
        for ch, op in self.hc.items():
            hc[ch] = factor * op

        c_ops = []
        if self.c_ops is not None:
            for x in self.c_ops:
                if isinstance(x, Qobj):
                    c_ops.append(factor * x)
                elif isinstance(x, list):
                    c_ops.append([x[0], factor * x[1]])
                else:
                    raise TypeError(type(x))

        return HamiltonianModel(h0, hc, c_ops, self.dims)

    def __rmul__(self, factor: float) -> HamiltonianModel:
        return self.__mul__(factor)

    def __div__(self, x: float) -> HamiltonianModel:
        return self.__mul__(1/x)

    def add_relaxation(self, index: int, T1: float) -> None:
        op = sqrt(1/T1) * qutip.destroy(self.dims[index])
        op = multipartite_op(op, index, self.dims)
        self.c_ops.append(op)

    def add_pure_dephasing(self, index: int, Tphi: float) -> None:
        op = sqrt(1/Tphi) * (qutip.destroy(self.dims[index]) * qutip.create(self.dims[index]))
        op = multipartite_op(op, index, self.dims)
        self.c_ops.append(op)

    @classmethod
    def from_dict(cls, ham: dict) -> HamiltonianModel:
        qub_dims = {int(k): v for k, v in ham.get('qub', {}).items()}
        osc_dims = {int(k): v for k, v in ham.get('osc', {}).items()}

        # Sort by index (dicts in Python 3.7 preserve insertion order).
        qub_dims = dict(sorted(qub_dims.items(), key=itemgetter(0)))
        osc_dims = dict(sorted(osc_dims.items(), key=itemgetter(0)))

        if len(osc_dims) > 0:
            raise NotImplementedError

        # qubit_0 * … * qubit_n * osc_0 * … * osc_n
        dims = [*qub_dims.values(), *osc_dims.values()]

        h_expr = sympify_h_str(ham['h_str'])

        var_symbols = []
        ch_symbols = []
        op_symbols = []

        for symbol in h_expr.free_symbols:
            if isinstance(symbol, VariableSymbol):
                var_symbols.append(symbol)
            elif isinstance(symbol, ChannelSymbol):
                ch_symbols.append(symbol)
            elif isinstance(symbol, OperatorSymbol):
                op_symbols.append(symbol)
            else:
                raise TypeError(type(symbol))

        h_expr = h_expr.subs({s: ham['vars'][s.name] for s in var_symbols})

        ops = []
        for symbol in op_symbols:
            opname, index = split_index(symbol.name)
            dim = qub_dims[index]
            op = create_op(opname, dim)
            ops.append(multipartite_op(op, index, dims))

        h0_expr = h_expr.subs(dict.fromkeys(ch_symbols, 0.0))

        h0 = lambdify(op_symbols, h0_expr)(*ops)

        hc = {}
        for symbol in ch_symbols:
            hc_expr = h_expr.coeff(symbol)
            hc_ = lambdify(op_symbols, hc_expr)(*ops)

            if hc_ == 0:
                continue

            hc[symbol.name] = hc_

        return cls(h0, hc, [], dims)


def split_index(name: str) -> Tuple[str, int]:
    match = re.match(r'(\w*?)(-?\d+)$', name)
    assert match
    head, index = match.groups()
    return head, int(index)


def create_op(opname: str, dim: int) -> Qobj:
    from qutip import destroy, create, qeye, num, sigmax, sigmay, sigmaz, sigmap, sigmam

    assert dim > 1, dim

    if dim == 2:
        op = {
            'X': sigmax(),
            'Y': sigmay(),
            'Z': sigmaz(),
            'Sp': sigmap(),
            'Sm': sigmam(),
            'I': qeye(2),
            'O': num(2),
        }[opname]
    else:
        op = {
            'X': destroy(dim) + create(dim),
            'Y': -1j * destroy(dim) + 1j * create(dim),
            'Z': qeye(dim) - 2 * num(dim),
            'Sp': create(dim),
            'Sm': destroy(dim),
            'I': qeye(dim),
            'O': num(dim),
        }[opname]

    return op


def multipartite_op(op: Qobj, index: int, dims: Sequence[int]) -> Qobj:
    from qutip import qeye, tensor

    assert len(dims) > index

    d = dims[index]
    assert op.dims == [[d], [d]]

    tensor_args = []
    for i, d in enumerate(dims):
        if i == index:
            tensor_args.append(op)
        else:
            tensor_args.append(qeye(d))

    return tensor(tensor_args)
