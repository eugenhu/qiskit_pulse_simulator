import numpy as np
from typing import Tuple


def direct_rotation(eigvals: np.ndarray, eigvecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate standard basis ([1, 0, …], [0, 1, …], etc.) into closest eigenspaces specified by `eigvals`,
    `eigvecs`.

    Returns `(new_eigvals, new_basis)` where `new_basis` is the new basis and `new_eigvals` are the
    corresponding eigenvalues.
    """
    dim = eigvecs.shape[0]
    close_eigvals = []
    eigspaces = []

    for i, x in enumerate(eigvals):
        v = eigvecs[:, [i]]

        for j, y in enumerate(close_eigvals):
            if np.allclose(x, y):
                break
        else:
            close_eigvals.append([x])
            eigspaces.append([v])
            continue

        close_eigvals[j].append(x)
        eigspaces[j].append(v)

    unique_eigvals = list(map(np.mean, close_eigvals))
    eigspaces = list(map(np.hstack, eigspaces))

    new_eigvals = []
    new_basis = []
    for i in range(dim):
        overlaps = [np.linalg.norm(V[i]) for V in eigspaces]
        j = np.argmax(overlaps)

        V = eigspaces[j]
        Vh = V.T.conjugate()

        x = Vh[:, i]
        x /= np.linalg.norm(x)

        R = np.identity(len(x)) - (x[:, None] @ x[None, :].conjugate())
        u, s, _ = np.linalg.svd(R)

        V_ = V @ u[:, s > np.min(s)]

        new_eigvals.append(unique_eigvals[j])
        new_basis.append(V @ x)

        eigspaces[j] = V_

    new_eigvals = np.array(new_eigvals)
    new_basis = np.array(new_basis)
    new_basis = new_basis.T

    return new_eigvals, new_basis
