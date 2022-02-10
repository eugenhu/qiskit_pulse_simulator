from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Optional, TypeVar

import numpy as np
import scipy.ndimage
import qutip


T = TypeVar('T')


class Generator(ABC):
    @abstractmethod
    def shift_phase(self, t0: int, phase: float) -> None:
        ...

    @abstractmethod
    def shift_frequency(self, t0: int, frequency: float) -> None:
        ...

    @abstractmethod
    def play(self, t0: int, samples: np.ndarray) -> None:
        ...

    @abstractmethod
    def delay(self, t0: int, duration: int) -> None:
        ...

    @abstractmethod
    def compile(self, t0: Optional[int] = None, t1: Optional[int] = None) -> qutip.Cubic_Spline:
        ...

    @property
    @abstractmethod
    def duration(self) -> int:
        ...

    def copy(self: T) -> T:
        return deepcopy(self)


@dataclass
class FrameInstruction:
    t0: int


@dataclass
class ShiftPhase(FrameInstruction):
    phase: float


@dataclass
class ShiftFrequency(FrameInstruction):
    frequency: float


class SimpleGenerator(Generator):
    def __init__(
            self,
            lo: float,
            bandwidth: Optional[float] = None,
            modulation: Optional[str] = None,
            subpixels: Optional[int] = None,
    ) -> None:
        self.lo = lo
        self.bandwidth = bandwidth or np.inf
        self.modulation = modulation or 'qam'
        self.subpixels = subpixels or round(10*lo)

        self.frame_instructions = []
        self.waveform_memory = np.zeros(1024, complex)
        self._duration = 0

        self.amp = 1.0

    def shift_phase(self, t0: int, phase: float) -> None:
        self.frame_instructions.append(ShiftPhase(t0, phase))

    def shift_frequency(self, t0: int, frequency: float) -> None:
        self.frame_instructions.append(ShiftFrequency(t0, frequency))

    def play(self, t0: int, samples: np.ndarray) -> None:
        t1 = t0 + len(samples)
        self._ensure_waveform_memory_size(t1)
        self.waveform_memory[t0:t1] = samples
        self._duration = max(self._duration, t1)

    def delay(self, t0: int, duration: int) -> None:
        t1 = t0 + duration
        self._ensure_waveform_memory_size(t1)
        self._duration = max(self._duration, t1)

    def _ensure_waveform_memory_size(self, size: int) -> None:
        cur_size = self.waveform_memory.size
        if size < cur_size:
            return

        grow = 2**math.ceil(math.log2(size/cur_size))
        new_size = grow * cur_size
        self.waveform_memory = np.pad(self.waveform_memory, pad_width=(0, new_size - cur_size))

    def compile(self, t0: Optional[int] = None, t1: Optional[int] = None) -> qutip.Cubic_Spline:
        t0 = t0 or 0
        t1 = t1 or self._duration

        if t1 - t0 <= 0:
            raise ValueError("Interval is empty.")

        subpixels = self.subpixels
        y = self.samples(t0, t1)
        S = qutip.Cubic_Spline(t0 + .5/subpixels, t1 - .5/subpixels, y)
        return S

    def samples(self, t0: Optional[int] = None, t1: Optional[int] = None) -> np.ndarray:
        t0 = t0 or 0
        t1 = t1 or self._duration

        lo = self.lo
        bandwidth = self.bandwidth
        subpixels = self.subpixels

        if t1 <= t0:
            return np.empty(0, float)

        self._ensure_waveform_memory_size(t1)
        waveform = self.waveform_memory[t0:t1]

        t = t0 + np.arange(t1 - t0)
        ssb = np.exp(1j * self._carrier_phase_mod(t))
        waveform = ssb * waveform

        waveform = np.repeat(waveform, subpixels)

        if np.isfinite(bandwidth):
            waveform = scipy.ndimage.gaussian_filter1d(
                waveform,
                sigma=1/(2*np.pi * bandwidth/1.1775) * subpixels,
                mode='constant',
            )

        t2 = np.linspace(t0, t1, len(t) * subpixels, endpoint=False)
        carrier = np.exp(2j*np.pi* t2 * lo)

        if self.modulation == 'qam':
            signal = (carrier * waveform).real
        elif self.modulation == 'am':
            signal = carrier.real * waveform.real
        else:
            raise ValueError(self.modulation)

        signal *= self.amp

        return signal

    def _carrier_phase_mod(self, t: np.ndarray) -> np.ndarray:
        self.frame_instructions.sort(key=lambda x: x.t0)

        instructions = self.frame_instructions
        tbreaks = [x.t0 for x in instructions]
        tidx = np.searchsorted(t, tbreaks, side='left')

        out = np.zeros(len(t), float)

        for i, inst in enumerate(instructions):
            if isinstance(inst, ShiftPhase):
                phase = inst.phase
                out[tidx[i]:] += phase
            elif isinstance(inst, ShiftFrequency):
                t0 = inst.t0
                freq = inst.frequency
                t_ = t[tidx[i]:]
                out[tidx[i]:] += 2*np.pi * (t_ - t0) * freq
            else:
                raise TypeError(type(inst))

        return out

    @property
    def duration(self) -> int:
        return self._duration
