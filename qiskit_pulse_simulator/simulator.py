from __future__ import annotations
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Union

import numpy as np
import qiskit.pulse as qpulse
from qiskit.qobj import PulseLibraryItem
import qutip.solver

from .generator import Generator
from .hamiltonian import HamiltonianModel
from .progressbar import NestedProgressBar


__all__ = (
    'AcquireInstruction',
    'PulseSimulator',
)


class AcquireInstruction(NamedTuple):
    t0: int
    duration: int
    qubits: list
    memory_slot: list


class PulseSimulator:
    def __init__(
            self,
            system: HamiltonianModel,
            generators: Dict[str, Generator],
    ) -> None:
        # Convert channel names to lower case.
        generators = {ch.lower(): gen for ch, gen in generators.items()}

        self.system = system
        self.generators = generators
        self.acquire_instructions = []

    def load_instructions(
            self,
            instructions: list,
            pulse_library: list,
            dt: float,
            filter_channels: Optional[Callable[[str], bool]] = None,
    ) -> None:
        pulse_lib = create_pulse_library_map(pulse_library)
        for inst in instructions:
            ch = getattr(inst, 'ch', None)

            if ch is not None and filter_channels and filter_channels(ch) is False:
                continue

            if ch is not None:
                gen = self.generators[ch.lower()]
            else:
                gen = None

            if inst.name == 'fc':
                assert gen
                gen.shift_phase(inst.t0, inst.phase)
            elif inst.name == 'shiftf':
                assert gen
                frequency = inst.frequency*1e9 * dt
                gen.shift_frequency(inst.t0, frequency)
            elif inst.name == 'parametric_pulse':
                assert gen
                gen.play(inst.t0, self._parametric_pulse(inst.pulse_shape, inst.parameters))
            elif inst.name == 'delay':
                assert gen
                gen.delay(inst.t0, inst.duration)
            elif inst.name == 'acquire':
                self.acquire_instructions.append(
                    AcquireInstruction(inst.t0, inst.duration, inst.qubits, inst.memory_slot)
                )
            elif inst.name in pulse_lib:
                assert gen
                gen.play(inst.t0, pulse_lib[inst.name])
            else:
                raise ValueError(f"Unsupported instruction: {inst.name}")

    @staticmethod
    def _parametric_pulse(pulse_shape: str, parameters: dict) -> np.ndarray:
        parameters = parameters.copy()
        for k, v in parameters.items():
            if isinstance(v, tuple):
                parameters[k] = v[0] + 1j*v[1]

        if pulse_shape == 'gaussian':
            pulse_cls = qpulse.library.discrete.gaussian
        elif pulse_shape ==  'gaussian_square':
            pulse_cls = qpulse.library.discrete.gaussian_square
        elif pulse_shape == 'drag':
            pulse_cls = qpulse.library.discrete.drag
        elif pulse_shape == 'constant':
            pulse_cls = qpulse.library.discrete.drag
        else:
            raise ValueError(f"Unknown pulse shape: {pulse_shape}")

        return pulse_cls(**parameters).samples

    def mesolve(
            self,
            rho0=None,
            tlist=None,
            e_ops=None,
            options=None,
            progress_bar=True,
    ) -> qutip.solver.Result:
        if rho0 is None:
            rho0 = qutip.fock_dm(self.system.dims)

        return self._solve(
            'mesolve',
            rho0,
            tlist,
            e_ops,
            options,
            progress_bar,
        )

    def sesolve(
            self,
            psi0=None,
            tlist=None,
            e_ops=None,
            options=None,
            progress_bar=True,
    ) -> qutip.solver.Result:
        if psi0 is None:
            psi0 = qutip.basis(self.system.dims)

        return self._solve(
            'sesolve',
            psi0,
            tlist,
            e_ops,
            options,
            progress_bar,
        )

    def _solve(
            self,
            solver: str,
            state0: qutip.Qobj,
            tlist: Optional[Sequence[float]] = None,
            e_ops: Optional[Sequence] = None,
            options: Optional[Union[Dict, qutip.Options]] =None,
            progress_bar: Any = None,
    ) -> qutip.solver.Result:
        if tlist is None:
            tlist = [0.0, self.duration]

        if progress_bar is True:
            progress_bar = NestedProgressBar('tidx = {i}')

        if options is None:
            options = {}

        if isinstance(options, dict):
            if 'nsteps' not in options:
                if len(tlist) > 0:
                    options['nsteps'] = round(np.mean(np.diff(tlist)) * 1000)
            options = qutip.Options(**options)

        result = getattr(qutip, solver)(
            self._H(),
            state0,
            tlist,
            e_ops=e_ops,
            options=options,
            progress_bar=progress_bar,
        )

        return result

    def _H(self) -> list:
        H = [self.system.h0]

        t0 = 0
        t1 = self.duration

        _num_coeffs = []
        for name, op in self.system.hc.items():
            gen = self.generators.get(name.lower(), None)
            if gen is None or gen.duration == 0:
                continue
            func = gen.compile(t0, t1)
            _num_coeffs.append(len(func.coeffs))
            H.append([op, func])

        if not np.equal(_num_coeffs[0], _num_coeffs).all():
            raise ValueError(
                "Cubic_Spline must have same number of coeffs"
                " (otherwise weird qutip bugs happen when integrating)."
            )

        return H

    @property
    def duration(self) -> int:
        duration = 0
        for gen in self.generators.values():
            duration = max(gen.duration, duration)
        return duration


def create_pulse_library_map(pulse_lib_list: list) -> dict:
    pulse_lib = {}

    for x in pulse_lib_list:
        if isinstance(x, PulseLibraryItem):
            pass
        elif isinstance(x, dict):
            x = PulseLibraryItem.from_dict(x)
        else:
            raise TypeError(type(x))

        pulse_lib[x.name] = x.samples

    return pulse_lib
