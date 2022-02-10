from math import ceil, floor, log10
import time
from typing import List


# Conforms to qutip.ui.progressbar.BaseProgressBar.
class NestedProgressBar:
    def __init__(self, desc: str = '') -> None:
        self.desc = desc
        self._n = []
        self._N = []

        self.fill_char = '=>.'
        self.bar_width = 25
        self.status_width = 80

    @property
    def level(self) -> int:
        return len(self._n)

    def start(self, iterations: int, chunk_size=None) -> None:
        if self.level == 0:
            self.t_start = time.time()
        self._n.append(0)
        self._N.append(iterations)

    def update(self, n):
        self._n[-1] = n
        self._print_status()

    def _print_status(self, terminal=False) -> None:
        pct_done = self._percent_done()
        pct_string = f'{round(pct_done)}%'
        prog_bar = self._progress_bar()
        time_elapsed = self.time_elapsed().strip()
        time_remaining = self.time_remaining_est(pct_done)

        status = f'[{prog_bar}]'

        i = len(status)//2 - len(pct_string)//2
        status = status[:i] + pct_string + status[i+len(pct_string):]

        if not terminal:
            status += f' {time_elapsed}<{time_remaining}'
        else:
            status += f' {time_elapsed}'

        Nd = self._N[0]
        nd = min(self._n[0], Nd-1)
        pad = ceil(log10(Nd))
        i = f'{nd:{pad}d}'
        n = f'{nd+1:{pad}d}'
        N = f'{Nd}'

        if self.desc:
            desc = self.desc.format(i=i, n=n, N=N)
            status = desc + ': ' + status

        self._print(status)

        if terminal:
            print()

    def _print(self, status: str) -> None:
        status += ' '*(self.status_width - len(status))
        print('\r' + status, end='', flush=True)

    def _percent_done(self) -> float:
        return 100 * sum(self._parts_done())

    def _progress_bar(self) -> str:
        ratios = self._parts_done()
        parts = []
        for i, ratio in enumerate(ratios):
            fill_char = self.fill_char[:i+1][-1]
            parts.append(fill_char * floor(ratio*self.bar_width))
        bar = ''.join(parts)
        bar += ' ' * (self.bar_width - len(bar))
        return bar

    def _parts_done(self) -> List[float]:
        size = [1]
        for N in self._N[:-1]:
            size.append(size[-1]/N)

        parts_done = []
        for a, n, N in zip(size, self._n, self._N):
            parts_done.append(a * n/N)

        for N in self._N:
            if N == 1 and len(parts_done) > 1:
                parts_done.pop(0)
            else:
                break

        return parts_done

    def finished(self) -> None:
        if self.level == 0:
            return
        elif self.level == 1:
            self._n[0] = self._N[0]
            self._print_status(terminal=True)
            self._n.pop()
            self._N.pop()
        else:
            self._n.pop()
            self._N.pop()
            self.update(self._n[-1] + 1)

    def time_elapsed(self) -> str:
        dt = time.time() - self.t_start
        return self._format_time(dt)

    def time_remaining_est(self, p: float) -> str:
        if p > 0.0:
            dt = (time.time() - self.t_start) * (100.0 - p) / p
        else:
            dt = 0
        return self._format_time(dt)

    def _format_time(self, t: float) -> str:
        second = round(t % 60)
        minute = int(t//60 % 60)
        hour = int(t//3600)

        text = f'{minute:02d}:{second:02d}'
        if hour > 0:
            text = f'{hour:02d}:' + text

        return text
