import numpy as np
from .model import *

_rcond = 1e-9

def SSR(X, y):
    """Stepwise Sparse Regressor. See Boninsegna, Nüske, and Clementi J. Chem. Phys. 148, 241723 (2018)

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    for i in range(Nf):
        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = np.linalg.lstsq(X[:,active_terms], y, rcond=_rcond)[0]
        error = np.average(np.square(y-np.sum(X[:,active_terms]*coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef
        j = np.nanargmin(np.square(coefs))
        # print('Removing term {}.'.format(j))
        active_terms[j] = False
    return models_list

def bagging_SSR(X, y, n_estimators=10, n_samples=0.2):
    """Stepwise Sparse Regressor. See SSR. At each step of the SSR,
    for n_estimators:
    1. Rows are sampled with replacement using n_samples.
    2. A least squares fitting is done on each estimators.
    3. Resulting coefficients are averaged.
    4. The active term with the lowest coefficient is removed.

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    n_estimators : int, default 10
        Number of estimators for the LS regression.
    n_samples : {int, float}, default 0.2
        Number of sample to draw. If integer, then this is the number of samples,
        if float in [0, 1] then it is the portion of the total number of rows.

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    if type(n_samples) is float:
        n_samples = int(n_samples*y.size)

    for i in range(Nf):
        coefs = np.zeros(Nf)*np.nan
        coefs_table = np.zeros((n_estimators, Nf))*np.nan
        for k in range(n_estimators):
            idx = np.random.randint(y.size, size=n_samples)
            coefs_table[k, active_terms] = np.linalg.lstsq(X[idx][:,active_terms], y[idx], rcond=_rcond)[0]

        coefs[active_terms] = np.average(coefs_table[:, active_terms], axis=0)
        error = np.average(np.square(y-np.sum(X[:,active_terms]*coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef
        j = np.nanargmin(np.square(coefs))
        # print('Removing term {}.'.format(j))
        active_terms[j] = False
    return models_list
