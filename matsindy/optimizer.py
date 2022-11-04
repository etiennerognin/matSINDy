import numpy as np
from .model import model

_rcond = 1e-9


def SSR(X, y):
    """Stepwise Sparse Regressor. See Boninsegna, Nüske, and Clementi J. Chem.
    Phys. 148, 241723 (2018)

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
        coefs[active_terms] = np.linalg.lstsq(X[:, active_terms],
                                              y, rcond=_rcond)[0]
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef
        j = np.nanargmin(np.abs(coefs))
        # print('Removing term {}.'.format(j))
        active_terms[j] = False
    return models_list


def SSRD(X, y):
    """Stepwise Sparse Regressor with Decorrelation. This is a backward
    elimination process. At each step,
    a least square fit is done on active features. Three features are pre-
    selected: two forming the pair with highest anti-correlation, and one being
    the feature with the smallest absolute coefficient from the regression.
    The error induced by removing individually one of these three features is
    computed and the model with the smallest error is selected for the next
    iteration.

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

    # Correlation matrix without diagonal
    XtX = X.T @ X
    XtX -= np.diag(np.diag(XtX))

    # Starting with full library
    coefs = np.linalg.lstsq(X, y, rcond=_rcond)[0]
    error = np.average(np.square(y-np.sum(X*coefs, axis=1)))

    for i in range(Nf):
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find highest correlated pair
        beta2 = np.outer(coefs, coefs)
        ii = np.nanargmin(XtX*beta2)
        (j, k) = np.unravel_index(ii, XtX.shape)

        # Find the smallest coef
        L = np.nanargmin(np.abs(coefs))

        # Select best option
        error = 1e12
        select_index = None
        coefs = None
        for index in (j, k, L):
            active_terms[index] = False
            coefs1 = np.zeros(Nf)*np.nan
            coefs1[active_terms] = np.linalg.lstsq(X[:, active_terms],
                                                   y, rcond=_rcond)[0]
            error1 = np.average(np.square(y-np.sum(X[:, active_terms] *
                                                   coefs1[active_terms],
                                                   axis=1)))
            if error1 < error:
                select_index = index
                error = error1
                coefs = coefs1
            active_terms[index] = True

        active_terms[select_index] = False

    return models_list


def Ridge_SSR(X, y, alpha=0.0001):
    """Stepwise Sparse Regressor with RIDGE regularisation.

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    alpha : float, default 0.0001
        Ridge hyperparameter, see:
        https://scikit-learn.org/stable/modules/linear_model.html#regression

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=alpha, fit_intercept=False)

    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    for i in range(Nf):
        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = reg.fit(X[:, active_terms], y).coef_
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef
        j = np.nanargmin(np.abs(coefs))
        # print('Removing term {}.'.format(j))
        active_terms[j] = False
    return models_list


def STLSQ(X, y, threshold=0.01):
    """Sequential thresholded least-squares.
    See https://doi.org/10.1073/pnas.1517384113

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    threshold : float, default 0.01
        Threshold value. Note that no assumption is made on the scale of each
        feature.

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    Nf = X.shape[1]
    models_list = []

    # Initial guess: least squares on full library
    coefs = np.linalg.lstsq(X, y, rcond=_rcond)[0]
    error = np.average(np.square(y - np.sum(X*coefs, axis=1)))
    # Record model
    models_list.append(model(coefficients=coefs, error=error))

    active_terms = np.ones(Nf, dtype=bool)
    while any(np.abs(coefs) < threshold):
        # kill active terms below `threshold`
        active_terms[np.abs(coefs) < threshold] = False

        # Re-fit
        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = np.linalg.lstsq(X[:, active_terms],
                                              y, rcond=_rcond)[0]
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

    return models_list


def Ridge_STLSQ(X, y, threshold=0.01, alpha=0.0001):
    """Sequential thresholded least-squares with RIDGE regularisation.
    See https://doi.org/10.1073/pnas.1517384113

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    threshold : float, default 0.01
        Threshold value. Note that no assumption is made on the scale of each
        feature.
    alpha : float, default 0.0001
        Ridge hyperparameter, see:
        https://scikit-learn.org/stable/modules/linear_model.html#regression

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=alpha, fit_intercept=False)

    Nf = X.shape[1]
    models_list = []

    # Initial guess: least squares on full library
    coefs = reg.fit(X, y).coef_
    error = np.average(np.square(y - np.sum(X*coefs, axis=1)))
    # Record model
    models_list.append(model(coefficients=coefs, error=error))

    active_terms = np.ones(Nf, dtype=bool)
    while any(np.abs(coefs) < threshold):
        # kill active terms below `threshold`
        active_terms[np.abs(coefs) < threshold] = False

        # Re-fit
        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = reg.fit(X[:, active_terms], y).coef_
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

    return models_list


def backward_elimination(X, y):
    """Stepwise backward elimination.

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
    from tqdm import trange

    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []
    features_id = np.arange(Nf)

    # Start from ordianry least squares
    coefs = np.linalg.lstsq(X[:, active_terms], y, rcond=_rcond)[0]
    error = np.average(np.square(y-np.sum(X[:, active_terms]*coefs, axis=1)))
    for i in trange(Nf):
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        coefs_table = np.zeros((Nf, Nf))*np.nan
        errors = np.zeros(Nf)*np.nan

        # Try and remove one active term
        for k in features_id[active_terms]:
            active_terms[k] = False
            coefs_table[k, active_terms] = np.linalg.lstsq(X[:, active_terms],
                                                           y, rcond=_rcond)[0]
            errors[k] = np.average(np.square(y-np.sum(X[:, active_terms] *
                                             coefs_table[k, active_terms],
                                             axis=1)))
            active_terms[k] = True
        error = np.nanmin(errors)
        j = np.nanargmin(errors)
        coefs = coefs_table[j]
        # Find and remove smallest error
        active_terms[j] = False
    return models_list



# def bagging_SSR(X, y, n_estimators=10, n_samples=0.2):
#     """Stepwise Sparse Regressor. See SSR. At each step of the SSR,
#     for n_estimators:
#     1. Rows are sampled with replacement using n_samples.
#     2. A least squares fitting is done on each estimators.
#     3. Resulting coefficients are averaged.
#     4. The active term with the lowest coefficient is removed.
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#     n_estimators : int, default 10
#         Number of estimators for the LS regression.
#     n_samples : {int, float}, default 0.2
#         Number of sample to draw. If integer, then this is the number of
#         samples, if float in [0, 1] then it is the portion of the total number
#         of rows.
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#     Nf = X.shape[1]
#     active_terms = np.ones(Nf, dtype=bool)
#     models_list = []
#
#     if type(n_samples) is float:
#         n_samples = int(n_samples*y.size)
#
#     idx = list(np.random.randint(y.size, size=n_samples)
#                for i in range(n_samples))
#
#     for i in range(Nf):
#         coefs = np.zeros(Nf)*np.nan
#         coefs_table = np.zeros((n_estimators, Nf))*np.nan
#         for k in range(n_estimators):
#             coefs_table[k, active_terms] = np.linalg.lstsq(X[idx[k]][:, active_terms],
#                                                            y[idx[k]], rcond=_rcond)[0]
#
#         coefs[active_terms] = np.average(coefs_table[:, active_terms], axis=0)
#         error = np.average(np.square(y-np.sum(X[:, active_terms] *
#                                               coefs[active_terms], axis=1)))
#         # Record model
#         models_list.append(model(coefficients=coefs, error=error))
#
#         # Find and remove smallest coef
#         j = np.nanargmin(np.abs(coefs))
#         # print('Removing term {}.'.format(j))
#         active_terms[j] = False
#     return models_list


# def library_bagging_SSR(X, y, n_estimators=10, n_samples=0.2):
#     """Stepwise Sparse Regressor. See SSR. At each step of the SSR,
#     for n_estimators:
#     1. Rows are sampled with replacement using n_samples.
#     2. A least squares fitting is done on each estimators.
#     3. Resulting coefficients are averaged.
#     4. The active term with the lowest coefficient is removed.
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#     n_estimators : int, default 10
#         Number of estimators for the LS regression.
#     n_samples : {int, float}, default 0.2
#         Number of sample to draw. If integer, then this is the number of
#         samples, if float in [0, 1] then it is the portion of the total number
#         of rows.
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#     Nf = X.shape[1]
#     active_terms = np.ones(Nf, dtype=bool)
#     models_list = []
#
#     if type(n_samples) is float:
#         n_samples = int(n_samples*y.size)
#
#     idx = list(np.random.randint(y.size, size=n_samples)
#                for i in range(n_samples))
#
#     for i in range(Nf):
#         coefs = np.zeros(Nf)*np.nan
#         coefs_table = np.zeros((n_estimators, Nf))*np.nan
#         for k in range(n_estimators):
#             coefs_table[k, active_terms] = np.linalg.lstsq(X[idx[k]][:, active_terms],
#                                                            y[idx[k]], rcond=_rcond)[0]
#
#         coefs[active_terms] = np.average(coefs_table[:, active_terms], axis=0)
#         error = np.average(np.square(y-np.sum(X[:, active_terms] *
#                                               coefs[active_terms], axis=1)))
#         # Record model
#         models_list.append(model(coefficients=coefs, error=error))
#
#         # Find and remove smallest coef
#         j = np.nanargmin(np.abs(coefs))
#         # print('Removing term {}.'.format(j))
#         active_terms[j] = False
#     return models_list


# def robust_SSR(X, y):
#     """A robust version of Stepwise Sparse Regressor, using an smooth l1 loss.
#     Run scipy.optimize.least_squares with loss='soft_l1'
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#
#     from scipy.optimize import least_squares
#
#     Nf = X.shape[1]
#     active_terms = np.ones(Nf, dtype=bool)
#     models_list = []
#
#     for i in range(Nf):
#         coefs = np.zeros(Nf)*np.nan
#
#         def res(xi):
#             return y-np.sum(X[:, active_terms]*xi, axis=1)
#
#         xi0 = np.linalg.lstsq(X[:, active_terms], y, rcond=_rcond)[0]
#         print('Doing l1 loss...', i)
#         coefs[active_terms] = least_squares(res, xi0, loss='soft_l1', f_scale=0.01).x
#         error = np.average(np.square(y-np.sum(X[:, active_terms] *
#                                               coefs[active_terms], axis=1)))
#         # Record model
#         models_list.append(model(coefficients=coefs, error=error))
#
#         # Find and remove smallest coef
#         j = np.nanargmin(np.square(coefs))
#         # print('Removing term {}.'.format(j))
#         active_terms[j] = False
#     return models_list


# def outlier_SSR(X, y):
#     """A robust version of Stepwise Sparse Regressor, detecting and removing
#     outliers.
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#     Nf = X.shape[1]
#     active_terms = np.ones(Nf, dtype=bool)
#     models_list = []
#
#     for i in range(Nf):
#         # Outlier detection
#         use = np.ones(len(y), dtype=bool)
#         sigma0 = 0.
#         for k in range(100):
#             coefs0 = np.linalg.lstsq(X[use][:, active_terms], y[use],
#                                      rcond=_rcond)[0]
#             res = y - np.sum(X[:, active_terms]*coefs0, axis=1)
#             sigma = np.average(np.square(res[use]))
#             use[np.square(res) > 1*sigma] = False
#             if abs(sigma - sigma0) < 1e-6:
#                 break
#             sigma0 = sigma
#         removed = len(y) - np.sum(use)
#         print(f"Removed: {removed}, after {k} iterations.")
#
#         coefs = np.zeros(Nf)*np.nan
#         coefs[active_terms] = coefs0
#         error = np.average(np.square(y[use]-np.sum(X[use][:, active_terms] *
#                                      coefs[active_terms], axis=1)))
#         # Record model
#         models_list.append(model(coefficients=coefs, error=error))
#
#         # Find and remove smallest coef
#         j = np.nanargmin(np.square(coefs))
#         # print('Removing term {}.'.format(j))
#         active_terms[j] = False
#     return models_list
#
#
# def robust_bagging_SSR(X, y, n_estimators=10, n_samples=0.2):
#     """Stepwise Sparse Regressor. See SSR. At each step of the SSR,
#     for n_estimators:
#     1. Rows are sampled with replacement using n_samples.
#     2. A least squares fitting is done on each estimators.
#     3. Resulting coefficients are averaged.
#     4. The active term with the lowest coefficient is removed.
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#     n_estimators : int, default 10
#         Number of estimators for the LS regression.
#     n_samples : {int, float}, default 0.2
#         Number of sample to draw. If integer, then this is the number of samples,
#         if float in [0, 1] then it is the portion of the total number of rows.
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#     from scipy.optimize import least_squares
#
#     Nf = X.shape[1]
#     active_terms = np.ones(Nf, dtype=bool)
#     models_list = []
#
#     if type(n_samples) is float:
#         n_samples = int(n_samples*y.size)
#
#     for i in range(Nf):
#         coefs = np.zeros(Nf)*np.nan
#         coefs_table = np.zeros((n_estimators, Nf))*np.nan
#         for k in range(n_estimators):
#
#
#             idx = np.random.randint(y.size, size=n_samples)
#             def res(xi):
#                 return y[idx]-np.sum(X[idx][:, active_terms]*xi, axis=1)
#
#             xi0 = np.linalg.lstsq(X[idx][:, active_terms], y[idx], rcond=_rcond)[0]
#             coefs_table[k, active_terms] = least_squares(res, xi0, loss='soft_l1').x
#
#         coefs[active_terms] = np.average(coefs_table[:, active_terms], axis=0)
#         error = np.average(np.square(y-np.sum(X[:,active_terms]*coefs[active_terms], axis=1)))
#         # Record model
#         models_list.append(model(coefficients=coefs, error=error))
#
#         # Find and remove smallest coef
#         j = np.nanargmin(np.square(coefs))
#         # print('Removing term {}.'.format(j))
#         active_terms[j] = False
#     return models_list
#
#
# def outlier_bagging_SSR(X, y, n_estimators=10, n_samples=0.2):
#     """Stepwise Sparse Regressor. See SSR. At each step of the SSR,
#     for n_estimators:
#     1. Rows are sampled with replacement using n_samples.
#     2. A least squares fitting is done on each estimators.
#     3. Resulting coefficients are averaged.
#     4. The active term with the lowest coefficient is removed.
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#     n_estimators : int, default 10
#         Number of estimators for the LS regression.
#     n_samples : {int, float}, default 0.2
#         Number of sample to draw. If integer, then this is the number of samples,
#         if float in [0, 1] then it is the portion of the total number of rows.
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#     Nf = X.shape[1]
#     active_terms = np.ones(Nf, dtype=bool)
#     models_list = []
#
#     if type(n_samples) is float:
#         n_samples = int(n_samples*y.size)
#
#     for i in range(Nf):
#         coefs = np.zeros(Nf)*np.nan
#         coefs_table = np.zeros((n_estimators, Nf))*np.nan
#         for k in range(n_estimators):
#             idx = np.random.randint(y.size, size=n_samples)
#             coefs_table[k, active_terms] = np.linalg.lstsq(X[idx][:,active_terms], y[idx], rcond=_rcond)[0]
#
#         coefs[active_terms] = np.average(coefs_table[:, active_terms], axis=0)
#         error = np.average(np.square(y-np.sum(X[:,active_terms]*coefs[active_terms], axis=1)))
#         # Record model
#         models_list.append(model(coefficients=coefs, error=error))
#
#         # Find and remove smallest coef
#         j = np.nanargmin(np.square(coefs))
#         # print('Removing term {}.'.format(j))
#         active_terms[j] = False
#     return models_list







# def clamp_SSR(X, y):
#     """A version of Stepwise Sparse Regressor with elternative strategy.
#     At each step, chose whether the smallest or the largest coef should be
#     removed, given the subsequent error. This is in the spirit of a ridge
#     regression.
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#     Nf = X.shape[1]
#     active_terms = np.ones(Nf, dtype=bool)
#     models_list = []
#
#     # Start from ordianry least squares
#     coefs = np.linalg.lstsq(X, y, rcond=_rcond)[0]
#     error = np.average(np.square(y-np.sum(X*coefs, axis=1)))
#     for i in range(Nf-1):
#         # Record model
#         models_list.append(model(coefficients=coefs, error=error))
#
#         j_min = np.nanargmin(np.abs(coefs))
#         active_terms[j_min] = False
#         # Remove smallest and try
#         coefs_remin = np.zeros(Nf)*np.nan
#         coefs_remin[active_terms] = np.linalg.lstsq(X[:, active_terms],
#                                                     y, rcond=_rcond)[0]
#         error_remin = np.average(np.square(y-np.sum(X[:, active_terms] *
#                                                     coefs[active_terms],
#                                                     axis=1)))
#         active_terms[j_min] = True
#
#         # Remove largest and try
#         j_max = np.nanargmax(np.abs(coefs))
#         active_terms[j_max] = False
#         # Remove smallest and try
#         coefs_remax = np.zeros(Nf)*np.nan
#         coefs_remax[active_terms] = np.linalg.lstsq(X[:, active_terms],
#                                                     y, rcond=_rcond)[0]
#         error_remax = np.average(np.square(y-np.sum(X[:, active_terms] *
#                                                     coefs[active_terms],
#                                                     axis=1)))
#         active_terms[j_max] = True
#
#         # Chose the best strategy
#         if error_remin < error_remax:
#             coefs = coefs_remin
#             error = error_remin
#             active_terms[j_min] = False
#             print('Removing small coef.')
#         else:
#             coefs = coefs_remax
#             error = error_remax
#             active_terms[j_max] = False
#             print('Removing large coef.')
#     return models_list
