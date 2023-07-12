import numpy as np


def make_window(width, which='rectangular'):
    """Create test functions for the weak formulation

    Parameters
    ----------
    width : int
        Width of the support of the test function. Note that leading and
        trailing zeros will be added.
    which : {'rectangular', 'triangular', 'Gaussian', 'polynomial1',
              'polynomial2', 'exponential'}
        Shape of test function (see source)

    Returns
    -------
    ndarray size (width+2,)
    """
    width += 2

    if which == 'rectangular':
        window = np.ones(width)
    elif which == 'triangular':
        window = np.bartlett(width)
    elif which == 'Gaussian':
        x = np.arange(width)
        window = np.exp(-(x+0.5-width/2)**2/(0.02*width**2))
    elif which == 'polynomial1':
        x = np.arange(width)
        window = x**1*(width-x-1)**1
    elif which == 'polynomial2':
        x = np.arange(width)
        window = x**2*(width-x-1)**2
    elif which == 'exponential':
        x = np.arange(width)
        window = np.exp(3*x/width-3)-np.exp(-3)
    else:
        raise NotImplementedError

    # Post cleaning
    window[0] = 0
    window[-1] = 0
    window = window/np.sum(window)
    return window



def weak_form(observable, dt, window):
    """Retrun weak form of the input tensor along axis 0 using a test function defined by `window`.
    The window is translated to width-2 between each point.
    In prticular, if width=3, recalling that the first and last element of window is zero,
    the weak_form is equaivalent to a first order finite differentiation."""
    Nt = len(observable)
    width = len(window)

    Nout = 2+int((Nt-2*(width-1))/(width-2))

    if observable.ndim == 1:
        out = np.empty(Nout)
        for i in range(Nout):
            i_start = i*(width-2)
            out[i] = np.sum(window*observable[i_start:i_start+width], axis=0)*dt
    else:
        out = np.empty((Nout, 3, 3))
        for i in range(Nout):
            i_start = i*(width-2)
            out[i] = np.sum(window[:, None, None]*observable[i_start:i_start+width], axis=0)*dt
    return out

def weak_diff(observable, dt, window):
    """Retrun weak form of the derivative of the input tensor along axis 0 using a test function defined by `window` at scale `width`.
    The window is translated to half the width between each point.
    Note: using first order differentation compatible with integration by simple sum (and not trapez)"""

    width = len(window)
    # Test function including left zero:
    wind_diff = np.zeros(width)
    # derivative:
    wind_diff[1:] = -np.diff(window)/dt

    return weak_form(observable, dt, wind_diff)
