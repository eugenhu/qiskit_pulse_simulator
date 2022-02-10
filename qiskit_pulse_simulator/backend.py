from __future__ import annotations
from collections import defaultdict
import copy
from datetime import datetime
from itertools import product
from operator import attrgetter
from typing import List, Optional, Sequence, Tuple, Union, overload
import uuid

import numpy as np
import qiskit.compiler
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendPropertyError, BackendV1 as Backend, Options, JobV1 as Job
from qiskit.providers.models import (
    BackendProperties,
    PulseBackendConfiguration,
    PulseDefaults,
    UchannelLO,
)
from qiskit.providers.models.pulsedefaults import Discriminator, MeasurementKernel
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.qobj import PulseQobj
from qiskit.qobj.utils import MeasLevel
from qiskit.result import Result
import qutip
from qutip.measurement import measurement_statistics

from .generator import SimpleGenerator
from .hamiltonian import HamiltonianModel
from .job import StubJob
from .progressbar import NestedProgressBar
from .simulator import PulseSimulator
from .util import direct_rotation


__all__ = (
    'IBMQesquePulseSimulatorBackend',
)


Runnable = Union[QuantumCircuit, Schedule, ScheduleBlock]


class IBMQesquePulseSimulatorBackend(Backend):
    def __init__(
            self,
            hamiltonian: dict,
            u_channel_lo: List[List[Union[UchannelLO, dict]]],
            qubit_lo_freq: Optional[List[float]] = None,
            T1: Optional[List[float]] = None,
            T2: Optional[List[float]] = None,
            lo_freq: Optional[float] = 3.5e9,
            bandwidth: Optional[float] = 2e9,
            dt: float = 1/4.5e9,
            stubs: Optional[dict] = None,
    ) -> None:
        n_qubits = len(hamiltonian['qub'])

        system = HamiltonianModel.from_dict(hamiltonian)
        system = system * dt

        eigvals, eigvecs = np.linalg.eigh(system.h0)
        _, new_basis = direct_rotation(eigvals, eigvecs)
        system = system.transform(new_basis)

        # Drop spurious diagonal terms from floating-point inaccuracies.
        system.h0 = qutip.Qobj(
            np.diag(np.diag(system.h0)),
            dims=system.h0.dims,
        )

        # Add relaxation and pure dephasing.
        for i, (t1, t2) in enumerate(zip(T1, T2)):
            system.add_relaxation(i, t1/dt)
            tphi = 1/(1/t2 - .5/t1)
            if tphi > 0:
                system.add_pure_dephasing(i, tphi/dt)

        # Calculate qubit frequencies if driving frequencies not provided.
        if qubit_lo_freq is None:
            ground = qutip.basis(system.dims)
            qubit_lo_freq = []
            for i in range(n_qubits):
                levels = [0]*i + [1] + [0]*(len(system.dims) - (i+1))
                excited = qutip.basis(system.dims, levels)
                freq = qutip.expect(system.h0, excited) - qutip.expect(system.h0, ground)
                freq /= 2*np.pi * dt
                qubit_lo_freq.append(freq)

        # Ensure u_channel_lo is a List[List[UchannelLO]].
        new_u_channel_lo = []
        for u_terms in u_channel_lo:
            new_u_terms = []
            for u_term in u_terms:
                if isinstance(u_term, dict):
                    u_term = UchannelLO.from_dict(u_term)
                q = u_term.q
                scale = u_term.scale
                if isinstance(scale, tuple):
                    scale = scale[0] + 1j*scale[1]
                u_term = UchannelLO(q, scale)
                new_u_terms.append(u_term)
            new_u_channel_lo.append(new_u_terms)
        u_channel_lo = new_u_channel_lo

        self.system = system

        self._hamiltonian = hamiltonian
        self._n_qubits = n_qubits
        self._qubit_lo_freq = qubit_lo_freq
        self._u_channel_lo = u_channel_lo

        self._dt = dt
        self._lo_freq = lo_freq
        self._bandwidth = bandwidth

        configuration, defaults, properties = self._create_stubs(**(stubs or {}))

        super().__init__(configuration, provider=None)
        self._defaults = defaults
        self._properties = properties

    def _create_stubs(
            self,
            backend_name: str = 'pulse_simulator',
            backend_version: str = '0.0.1',
            gates: Optional[list] = None,
            max_shots: int = 32_000,
            coupling_map: Optional[list] = None,
            qubit_lo_range: Optional[list] = None,
            meas_lo_range: Optional[list] = None,
            qubit_freq_est: Optional[list] = None,
            meas_freq_est: Optional[list] = None,
            meas_map: Optional[list] = None,
            pulse_library: Optional[list] = None,
            cmd_def: Optional[list] = None,
            properties: Optional[BackendProperties] = None,
    ) -> Tuple[PulseBackendConfiguration, PulseDefaults, BackendProperties]:
        gates = gates or []
        basis_gates = list(map(attrgetter('name'), gates))
        n_uchannels = len(self._u_channel_lo)

        qubit_freq_est = qubit_freq_est or self._qubit_lo_freq
        meas_freq_est = meas_freq_est or list(7.35e9 + 0.12e9*np.random.rand(self._n_qubits))

        meas_map = meas_map or [list(range(self._n_qubits))]

        qubit_lo_range = qubit_lo_range or [[x - .5e9, x + .5e9] for x in qubit_freq_est]
        meas_lo_range = meas_lo_range or [[x - .5e9, x + .5e9] for x in meas_freq_est]

        pulse_library = pulse_library or []
        cmd_def = cmd_def or []

        # PulseBackendConfiguration and PulseDefaults expects frequencies in GHz.
        qubit_freq_est = (np.array(qubit_freq_est)/1e9).tolist()
        meas_freq_est = (np.array(meas_freq_est)/1e9).tolist()
        qubit_lo_range = (np.array(qubit_lo_range)/1e9).tolist()
        meas_lo_range = (np.array(meas_lo_range)/1e9).tolist()

        # dt in ns.
        dt = self._dt * 1e9
        dtm = dt

        configuration = PulseBackendConfiguration(
            backend_name=backend_name,
            backend_version=backend_version,
            n_qubits=self._n_qubits,
            basis_gates=basis_gates,
            gates=gates,
            local=True,
            simulator=True,
            conditional=False,
            open_pulse=True,
            memory=True,
            max_shots=max_shots,
            coupling_map=coupling_map,
            n_uchannels=n_uchannels,
            u_channel_lo=self._u_channel_lo,
            meas_levels=[2],
            qubit_lo_range=qubit_lo_range,
            meas_lo_range=meas_lo_range,
            dt=dt,
            dtm=dtm,
            rep_times=[1000.0],
            meas_kernels=['hw_qmfk'],
            discriminators=['hw_qmfk'],
            hamiltonian=self._hamiltonian,
            meas_map=meas_map,
            credits_required=False,
            parametric_pulses=['gaussian', 'gaussian_square', 'drag', 'constant'],
        )

        defaults = PulseDefaults(
            qubit_freq_est=qubit_freq_est,
            meas_freq_est=meas_freq_est,
            buffer=0,
            pulse_library=pulse_library,
            cmd_def=cmd_def,
            meas_kernel=MeasurementKernel('hw_qmfk', {}),
            discriminator=Discriminator('hw_qmfk', {}),
        )

        now = datetime.now()
        properties = properties or BackendProperties(backend_name, backend_version, now, [], [], [])

        return configuration, defaults, properties

    def _calculate_u_lo_freq(self, qubit_lo_freq: Sequence[float]) -> Sequence[float]:
        u_lo_freq = []

        for u_terms in self._u_channel_lo:
            freq = 0.0
            for u_term in u_terms:
                assert u_term.scale.imag == 0.0, "Why is there an imaginary component?"
                freq += u_term.scale.real * qubit_lo_freq[u_term.q]
            u_lo_freq.append(freq)

        return u_lo_freq

    @classmethod
    def _default_options(cls) -> Options:
        return Options(
            shots=4000,
            meas_level=MeasLevel.CLASSIFIED,
            solver='sesolve',
            nsteps=1000,
            use_openmp=False,
            progress_bar=True,
        )

    def run(self, run_input: Union[Runnable, Sequence[Runnable]], **options) -> Job:
        options = {**self.options.__dict__, **options}

        qobj = self._assemble(run_input, options)
        sims = self._create_simulator(qobj)

        if options['progress_bar']:
            progress_bar = NestedProgressBar("Circuit ({n}/{N})")
        else:
            progress_bar = None

        if len(sims) > 10:
            all_data = qutip.parallel_map(
                self._run_sim,
                sims,
                (options,),
                progress_bar=progress_bar
            )
        else:
            if progress_bar: progress_bar.start(len(sims))
            all_data = []
            for i, sim in enumerate(sims):
                if progress_bar: progress_bar.update(i)
                data = self._run_sim(sim, options, progress_bar=progress_bar)
                all_data.append(data)
            if progress_bar: progress_bar.finished()

        del progress_bar

        result = self._format_result(qobj, all_data)

        job = StubJob(
            backend=self,
            job_id=str(uuid.uuid4()),
            qobj=qobj,
            result=result,
        )

        result.job_id = job.job_id()

        return job

    @overload
    def create_simulator(self, run_input: Runnable, **options) -> PulseSimulator:
        ...
    @overload
    def create_simulator(self, run_input: Sequence[Runnable], **options) -> List[PulseSimulator]:
        ...
    def create_simulator(self, run_input, **options):
        options = {**self.options.__dict__, **options}
        qobj = self._assemble(run_input, options)
        sims = self._create_simulator(qobj)

        if len(sims) == 1:
            return sims[0]
        else:
            return sims

    def _assemble(self, run_input: Union[Runnable, Sequence[Runnable]], options: dict) -> PulseQobj:
        if not isinstance(run_input, Sequence):
            run_input = [run_input]

        schedules = self._schedule(run_input)
        qobj = qiskit.compiler.assemble(schedules, self, **options)

        # Reconstruct QuantumCircuit cregs info normally found in QasmQobjExperiment.header.
        # This is slightly hacky.
        for i, x in enumerate(run_input):
            if not isinstance(x, QuantumCircuit):
                continue

            qobj_exp = qobj.experiments[i]
            header = self._make_circuit_qobj_header(x)

            # Sanity check.
            assert header['memory_slots'] == qobj_exp.header.memory_slots

            for k, v in header.items():
                setattr(qobj_exp.header, k, v)

        assert isinstance(qobj, PulseQobj)
        return qobj

    def _schedule(self, run_input: Sequence[Runnable]) -> List[Union[Schedule, ScheduleBlock]]:
        schedules = []
        for x in run_input:
            if isinstance(x, QuantumCircuit):
                x = qiskit.compiler.schedule(x, self)
            elif isinstance(x, (Schedule, ScheduleBlock)):
                pass
            else:
                raise TypeError(type(x))

            schedules.append(x)

        return schedules

    def _make_circuit_qobj_header(self, circuit: QuantumCircuit) -> dict:
        # Copied from qiskit.assembler.assemble_circuits._assemble_circuit
        n_qubits = 0
        memory_slots = 0
        qubit_labels = []
        clbit_labels = []
        qreg_sizes = []
        creg_sizes = []

        for qreg in circuit.qregs:
            qreg_sizes.append([qreg.name, qreg.size])
            for j in range(qreg.size):
                qubit_labels.append([qreg.name, j])
            n_qubits += qreg.size

        for creg in circuit.cregs:
            creg_sizes.append([creg.name, creg.size])
            for j in range(creg.size):
                clbit_labels.append([creg.name, j])
            memory_slots += creg.size

        header = {
            'n_qubits': n_qubits,
            'memory_slots': memory_slots,
            'qubit_labels': qubit_labels,
            'clbit_labels': clbit_labels,
            'qreg_sizes': qreg_sizes,
            'creg_sizes': creg_sizes,
            'global_phase': circuit.global_phase,
        }

        return header

    def _create_simulator(self, qobj: PulseQobj) -> List[PulseSimulator]:
        if getattr(qobj.config, 'qubit_lo_freq', None):
            qubit_lo_freq = [f*1e9 for f in qobj.config.qubit_lo_freq]
        else:
            qubit_lo_freq = self._qubit_lo_freq

        pulse_library = getattr(qobj.config, 'pulse_library', [])

        sims = []
        for exp in qobj.experiments:
            sim = PulseSimulator(
                self.system,
                self._create_generators(qubit_lo_freq),
            )
            sim.load_instructions(
                exp.instructions,
                pulse_library,
                dt=self._dt,
                # Ignore measure channels.
                filter_channels=lambda ch: not ch.startswith('m'),
            )
            sims.append(sim)

        return sims

    def _create_generators(self, qubit_lo_freq: List[float]) -> Dict[str, Generator]:
        dt = self._dt
        d_freq = qubit_lo_freq
        u_freq = self._calculate_u_lo_freq(d_freq)

        if self._lo_freq is not None:
            lo = self._lo_freq * dt
            sideband = {}
            sideband.update({
                f'D{i}': (freq - self._lo_freq) * dt
                for i, freq in enumerate(d_freq)
            })
            sideband.update({
                f'U{i}': (freq - self._lo_freq) * dt
                for i, freq in enumerate(u_freq)
            })
            modulation = 'am'
            subpixels = round(10*lo)
        else:
            lo = {}
            lo.update({
                f'D{i}': freq * dt
                for i, freq in enumerate(d_freq)
            })
            lo.update({
                f'U{i}': freq * dt
                for i, freq in enumerate(u_freq)
            })
            sideband = None
            modulation = 'qam'
            subpixels = round(max(10*freq for freq in lo.values()))

        if self._bandwidth is not None:
            bandwidth = self._bandwidth * dt
        else:
            bandwidth = None

        channels = self.system.hc.keys()
        generators = {}
        for ch in channels:
            generators[ch] = gen = SimpleGenerator(
                lo[ch] if isinstance(lo, dict) else lo,
                bandwidth,
                modulation,
                subpixels,
            )

            if sideband:
                gen.shift_frequency(0, sideband[ch])
                gen.amp *= 1/np.sinc(sideband[ch])
                if bandwidth:
                    gen.amp *= 1/np.exp(-0.5 * sideband[ch]**2 / (bandwidth/1.1775)**2)

            if modulation == 'am':
                gen.amp *= 2.0

        return generators

    @staticmethod
    def _run_sim(sim: PulseSimulator, options: dict, *, progress_bar=None) -> dict:
        solver = getattr(sim, options['solver'])

        result = solver(
            tlist=np.linspace(0, sim.duration, sim.duration+1),
            options=qutip.Options(
                nsteps=options.get('nsteps'),
                use_openmp=options.get('use_openmp'),
            ),
            progress_bar=progress_bar,
        )

        pvm = {s: qutip.fock_dm(sim.system.dims, list(s))
               for s in product(*map(range, sim.system.dims))}
        final_state = result.states[-1]
        if final_state.isherm:
            final_state = final_state.trunc_neg()
        pvals = measurement_statistics(final_state, list(pvm.values()))[1]
        pmf = dict(zip(pvm.keys(), pvals))

        reg_from_qubit = {}
        _t0 = None
        for inst in sim.acquire_instructions:
            if _t0 is None: _t0 = inst.t0
            if inst.t0 != _t0: raise ValueError("Mid-circuit measurements not supported.")
            for q, c in zip(inst.qubits, inst.memory_slot):
                reg_from_qubit[q] = c

        probs = defaultdict(int)
        for key, pval in pmf.items():
            state = 0
            for i, n in enumerate(key):
                c = reg_from_qubit.get(i)
                if c is None: continue
                state += (n > 0) * 2**c
            probs[hex(state)] += pval

        states = np.array(list(probs.keys()), object)
        pvals = np.array(list(probs.values()), float)

        rng = np.random.default_rng()
        counts = dict(zip(
            states,
            rng.multinomial(options['shots'], pvals),
        ))

        data = {
            'counts': counts,
        }

        return data

    def _format_result(self, qobj: PulseQobj, all_data: List[dict]) -> Result:
        exp_results = []
        for qobj_exp, data in zip(qobj.experiments, all_data):
            exp_result = {
                'shots': qobj.config.shots,
                'success': True,
                'data': data,
                'meas_level': qobj.config.meas_level,
                'status': 'DONE',
                'header': qobj_exp.header.to_dict(),
            }

            if hasattr(qobj.config, 'meas_return'):
                exp_result['meas_return'] = qobj.config.meas_return

            exp_results.append(exp_result)

        result = Result.from_dict({
            'backend_name': self.configuration().backend_name,
            'backend_version': self.configuration().backend_version,
            'qobj_id': qobj.qobj_id,
            'job_id': '',
            'success': True,
            'results': exp_results,
            'date': datetime.now(),
            'status': 'Successful completion',
            'header': qobj.header.to_dict(),
        })

        return result

    def defaults(self) -> PulseDefaults:
        defaults = copy.copy(self._defaults)
        return defaults

    def properties(self) -> BackendProperties:
        properties = copy.copy(self._properties)
        return properties

    @classmethod
    def from_backend(cls, backend: Backend) -> IBMQesquePulseSimulatorBackend:
        config = backend.configuration()
        defaults = backend.defaults()
        properties = backend.properties()

        hamiltonian = config.hamiltonian
        u_channel_lo = config.u_channel_lo

        if properties is not None:
            try:
                T1 = [properties.qubit_property(i, 'T1')[0] for i in range(config.n_qubits)]
            except BackendPropertyError:
                T1 = None
            try:
                T2 = [properties.qubit_property(i, 'T2')[0] for i in range(config.n_qubits)]
            except BackendPropertyError:
                T2 = None
        else:
            T1 = None
            T2 = None

        stubs = {
            'backend_name': config.backend_name + '_simulator',
            'gates': config.gates,
            'coupling_map': config.coupling_map,
            'qubit_lo_range': config.qubit_lo_range,
            'meas_lo_range': config.meas_lo_range,
            'meas_map': config.meas_map,
            'properties': properties,
        }

        if defaults is not None:
            stubs.update({
                'meas_freq_est': defaults.meas_freq_est,
                'pulse_library': defaults.pulse_library,
                'cmd_def': defaults.cmd_def,
            })

        return cls(
            hamiltonian,
            u_channel_lo,
            T1=T1,
            T2=T2,
            stubs=stubs,
        )
