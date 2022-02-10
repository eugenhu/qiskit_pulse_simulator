from typing import Any

from qiskit.providers import Backend, JobError, JobStatus, JobV1 as Job
from qiskit.result import Result


class StubJob(Job):
    def __init__(self, backend: Backend, job_id: str, qobj: Any, result: Result) -> None:
        super().__init__(backend, job_id)
        self._qobj = qobj
        self._result = result

    def submit(self) -> None:
        raise JobError("Job has already been submitted.")

    def result(self, timeout=None) -> Result:
        if self._result is None:
            raise JobError("StubJob has no result set.")

        return self._result

    def cancel(self) -> None:
        pass

    def status(self) -> JobStatus:
        if self._result is not None:
            return JobStatus.DONE
        else:
            return JobStatus.ERROR

    def backend(self) -> Backend:
        return self._backend

    def qobj(self) -> Any:
        return self._qobj
