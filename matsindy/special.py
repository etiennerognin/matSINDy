import numpy as np


def simulate_from_literature(name, A0, time, gradU, **params):
    """Simulate a model from the literature for the given scenario.

    Parameters
    ----------
    name : {'UCM', 'FENE-P', 'Hinch-Larson'}
        Model name. Models parameters are given as kwargs.
    A0: (3, 3) ndarray
        Initial condition.
    time : (N,) ndarray
        Time vector.
    gradU : (N, 3, 3) ndarray
        Velocity gradient time series

    Returns
    -------
    (N, 3, 3) ndarray
        Simulated conformation tensor.
    """

    # Define callable rate of change
    if name == 'UCM':
        def dA(it, dt, Ait):
            out = Ait @ gradU[it] + gradU[it].T @ Ait
            out += np.eye(3)
            out += -Ait
            return dt*out

    elif name == 'FENE-P':
        def dA(it, dt, Ait):
            out = Ait @ gradU[it] + gradU[it].T @ Ait
            out += np.eye(3)
            out += -Ait/np.abs(1. - np.trace(Ait)/params['L_max']**2)
            return dt*out

    elif name == 'FENE-CR':
        def dA(it, dt, Ait):
            out = Ait @ gradU[it] + gradU[it].T @ Ait
            out += (np.eye(3)-Ait)/np.abs(1.-np.trace(Ait)/params['L_max']**2)
            return dt*out

    elif name == 'Hinch-Larson':
        def dA(it, dt, Ait):
            out = Ait @ gradU[it] + gradU[it].T @ Ait
            out += -2/params['L_max']**2*max(0, np.trace(Ait @ gradU[it]))*Ait
            out += np.eye(3)
            out += -Ait
            return dt*out

    elif name == 'FENE-kink':
        global L2eff
        # Effective length
        L2eff = params['L_max']**1.5
        L2eq = params['L_max']**1.5

        def dA(it, dt, Ait):
            global L2eff
            # Update L2eff
            L2eff += dt*(max(0, np.trace(Ait @ gradU[it]))
                         * (1-L2eff/params['L_max']**2)
                         + L2eq - L2eff*2
                         )
            out = Ait @ gradU[it] + gradU[it].T @ Ait
            out += np.eye(3)
            out += -Ait/np.abs(1. - np.trace(Ait)/L2eff)
            return dt*out
    else:
        raise NotImplementedError(f'Model "{name}" not implemented yet.')

    # Use explicit first-order Euler scheme. Fast, and also makes
    # sense if dt is the time step used in Ito molecular dynamics.
    A_sim = np.empty_like(gradU)
    A_sim[0] = A0
    N = len(time)
    dt = np.diff(time)

    sub = params.get('subdivide', 1)
    for i in range(N-1):
        Ap = A_sim[i]
        for j in range(sub):
            Ap = Ap + dA(i, dt[i]/sub, Ap)
        A_sim[i+1] = Ap

    return A_sim
