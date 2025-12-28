# quantum_rules.py
import numpy as np

def update_link_probabilities(P, lam, eta, omega, dt, t, noise_std=0.02):
    """
    Improved QIPSF link model:
      P_ij(t+dt) = P_ij(t) * exp(-lam * dt) + eta * sin(omega * t) + noise_ij
    with small edge-wise noise to avoid global synchronisation.
    """
    decay_factor = np.exp(-lam * dt)
    P_new = P * decay_factor + eta * np.sin(omega * t)

    if noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=P.shape)
        noise = (noise + noise.T) / 2.0  # keep symmetry
        np.fill_diagonal(noise, 0.0)
        P_new += noise

    return np.clip(P_new, 0.0, 1.0)


def apply_entanglement_correlation(P, correlated_pairs, rho0, mu, t):
    """
    Same as before, but typically with smaller rho0 or fewer pairs to avoid
    strong global lock-in.
    """
    if not correlated_pairs:
        return P

    C_t = rho0 * np.exp(-mu * t)
    P_new = P.copy()

    for (i, j, k, l) in correlated_pairs:
        if i > j:
            i, j = j, i
        if k > l:
            k, l = l, k

        pij = P_new[i, j]
        pkl = P_new[k, l]

        avg = 0.5 * (pij + pkl)

        pij_new = pij * (1.0 - C_t) + avg * C_t
        pkl_new = pkl * (1.0 - C_t) + avg * C_t

        P_new[i, j] = P_new[j, i] = pij_new
        P_new[k, l] = P_new[l, k] = pkl_new

    return np.clip(P_new, 0.0, 1.0)


def stability_index(P):
    n = P.shape[0]
    return float(np.sum(P) / (n * n))
