import numpy as np
from .model import model
from tqdm import trange

_rcond = None


def ridge(X, y, alpha=0):
    """Implementation of ridge regression based in explicit formula and
    numpy.linalg.pinv routine (which means it will also be robust for ordinary
    least squares). It should be scalable for features of O(100).
    See https://en.wikipedia.org/wiki/Ridge_regression.
    Note that for OLS, using the explicit formula
    >>> beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    is actually faster (at the time of writing) than calling:
    >>> beta = np.linalg.lstsq(X, y, rcond=None)[0]

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    alpha : float
        Ridge regularisation parameter.

    Returns
    -------
    beta : ndarray
        Array of coefficients.
    """
    d = X.shape[1]
    if alpha:
        M = X.T @ X + alpha*np.eye(d)
    else:
        M = X.T @ X
    invM = np.linalg.pinv(M)
    return invM @ X.T @ y


def relaxed_ridge(X, y, alpha, Xc, yc):
    """Implementation of ridge regression based in explicit formula and
    numpy.linalg.pinv routine (which means it will also be robust for ordinary
    least squares). It should be scalable for features of O(100).
    See https://en.wikipedia.org/wiki/Ridge_regression

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    alpha : float
        Ridge regularisation parameter.

    Returns
    -------
    beta : ndarray
        Array of coefficients.
    """
    n, d = X.shape
    X_full = np.vstack((X, Xc))
    y_full = np.hstack((y, yc))
    M = X_full.T @ X_full + alpha*np.eye(d)
    invM = np.linalg.pinv(M)
    beta_full = invM @ X_full.T @ y_full
    return beta_full[0:n]


def fit_from_indices(X, y, indices, alpha=0.0):
    """Fit using only features given by index list. Unbiased least squares is
    used by default, but ridge regularisation can be done by providing a non-
    zero alpha parameter.

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    indices : list of int
        List of column indices to use for the fit.
    alpha : float, default 0.0
        Ridge regularisation parameter.

    Returns
    -------
    model object
        Model fit.
    """
    coefs = np.zeros(X.shape[1])*np.nan
    coefs[indices] = ridge(X[:, indices], y, alpha)
    error = np.average(np.square(y-np.sum(X[:, indices] *
                                          coefs[indices], axis=1)))
    return model(coefficients=coefs, error=error)


def SSR(X, y, rescale=False, alpha=0.0):
    """Stepwise Sparse Regressor. See Boninsegna, Nüske, and Clementi J. Chem.
    Phys. 148, 241723 (2018).
    Ridge regularisation can be applied.

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    rescale : bool, default False
        If True, will rescale features by their maximum absolute value
    alpha : float, default 0.0
        Ridge regularisation parameter.

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    print('Running SSR...')
    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    if rescale:
        # Rescale features
        s = np.max(np.abs(X), axis=0)
        X = X/s

    for i in trange(Nf):
        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = ridge(X[:, active_terms], y, alpha)
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        if rescale:
            models_list.append(model(coefficients=coefs/s, error=error))
        else:
            models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef
        j = np.nanargmin(np.abs(coefs))
        # print('Removing term {}.'.format(j))
        active_terms[j] = False

    if rescale:
        X = X*s
    return models_list


def SSRD(X, y, alpha=0.0):
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
    alpha : float, default 0.0
        Ridge regularisation parameter.

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    print('Running SSRD...')
    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    # Correlation matrix without diagonal
    XtX = X.T @ X
    XtX -= np.diag(np.diag(XtX))

    # Starting with full library
    coefs = ridge(X, y, alpha)
    error = np.average(np.square(y-np.sum(X*coefs, axis=1)))

    for i in trange(Nf-3):
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find highest correlated pair
        beta2 = np.outer(coefs, coefs)
        ii = np.nanargmax(np.abs(XtX*beta2))
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
            coefs1[active_terms] = ridge(X[:, active_terms], y, alpha)
            error1 = np.average(np.square(y-np.sum(X[:, active_terms] *
                                                   coefs1[active_terms],
                                                   axis=1)))
            if error1 < error:
                select_index = index
                error = error1
                coefs = coefs1
            active_terms[index] = True

        active_terms[select_index] = False
    models_list.append(model(coefficients=coefs, error=error))

    return models_list


def STLSQ(X, y, threshold=0.01, alpha=0.0):
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
    alpha : float, default 0.0
        Ridge regularisation parameter.

    Returns
    -------
    model
        Model selected and fit.
    """
    Nf = X.shape[1]

    # Initial guess: least squares on full library
    coefs = ridge(X, y, alpha)
    error = np.average(np.square(y-np.sum(X*coefs, axis=1)))

    active_terms = np.ones(Nf, dtype=bool)
    while any(np.abs(coefs) < threshold):
        # kill active terms below `threshold`
        active_terms[np.abs(coefs) < threshold] = False

        # Re-fit
        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = ridge(X[:, active_terms], y, alpha)
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))

    return model(coefficients=coefs, error=error)


def SBE(X, y, n_test=2, alpha=0.0):
    """Stepwise backward elimination. At each step, select randomly two or more
    features and remove the one leading to smallest error increase.

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    n_test : int, default 2
        Number of features to select for testing.
    alpha : float, default 0.0
        Ridge regularisation parameter.

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """

    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []
    features_id = np.arange(Nf)

    # Start from ordianry least squares
    coefs = ridge(X, y, alpha)
    error = np.average(np.square(y-np.sum(X*coefs, axis=1)))
    for i in trange(Nf-1):
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Try and remove one active term
        error = 1e12
        for k in np.random.choice(features_id[active_terms],
                                  size=min(Nf-1-i, n_test), replace=False):
            active_terms[k] = False
            coefs1 = np.zeros(Nf)*np.nan
            coefs1[active_terms] = ridge(X[:, active_terms], y, alpha)
            error1 = np.average(np.square(y-np.sum(X[:, active_terms] *
                                                   coefs1[active_terms],
                                                   axis=1)))
            if error1 < error:
                select_index = k
                error = error1
                coefs = coefs1
            active_terms[k] = True

        active_terms[select_index] = False
    models_list.append(model(coefficients=coefs, error=error))
    return models_list


# def SR3(X, y, alpha=0.0, kappa=0.0):
#     """Sparse Relaxed Regularized Regression.
#     See https://doi.org/10.1109/ACCESS.2018.2886528
#
#     Parameters
#     ----------
#     X : array_like
#         The matrix of features.
#     y : array_like
#         Vector to fit.
#     alpha : float, default 0.0
#         Ridge hyperparameter.
#     kappa : float, default 0.0001
#         Relaxation parameter (partial Ridge).
#
#     Returns
#     -------
#     list of model
#         List of models with decreasing number of active terms.
#     """
#     from sklearn.linear_model import Lasso
#     reg = Lasso(alpha=alpha, fit_intercept=False)
#
#     d = X.shape[1]
#     X_block = np.vstack((X, np.sqrt(kappa)*np.eye(d)))
#     u, s, vT = np.linalg.svd(X_block, full_matrices=False)
#     w = np.zeros(d)
#
#     for i in range(100):
#         # Assemble right-hand side
#         y_block = np.hstack((y, np.sqrt(kappa)*w))
#
#         # Least squares using pre-computed SVD
#         z = (u.T @ y_block)/s
#         xi = np.sum(vT*z[:, None], axis=0)
#
#         # Sparse regression of w
#         w1 = reg.fit(np.eye(d), xi).coef_
#
#         # Change from previous iter
#         dw = np.average(np.square(w1-w))
#         w = w1
#         if dw < 1e-6:
#             break
#
#     error = np.average(np.square(y-np.sum(X*w, axis=1)))
#
#     w[np.abs(w) < 1e-12] = np.nan
#     w[~np.isnan(w)] = np.linalg.lstsq(X[:, ~np.isnan(w)], y, rcond=None)[0]
#     error = np.average(np.square(y-np.sum(X[:, ~np.isnan(w)] *
#                                           w[~np.isnan(w)], axis=1)))
#
#     return model(coefficients=w, error=error)


def SSR_var(X, y, alpha=0.0, Xc=None, yc=None):
    """Sparse Relaxed Regularized Regression.
    See https://doi.org/10.1109/ACCESS.2018.2886528
    Remove the term with highest uncertainty.

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    alpha : float, default 0.0
        Ridge regularisation parameter.
    Xc : ndarray
        Constraint matrix
    yc : ndarray
        Constraint target

    Returns
    -------
    list of model
        List of models with decreasing number of active terms.
    """
    print('Running SSR_var...')
    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    for i in trange(Nf):
        d = np.sum(active_terms)
        M = X[:, active_terms].T @ X[:, active_terms] + alpha*np.eye(d)
        invM = np.linalg.pinv(M)
        beta = invM @ X[:, active_terms].T @ y

        if Xc is not None:
            if Xc.ndim == 1:
                Xc = Xc.reshape((1, Xc.size))
            C = Xc[:, active_terms]
            if yc is None:
                yc = np.zeros(len(Xc))
            res = C @ beta - yc
            beta += - invM @ C.T @ np.linalg.pinv(C @ invM @ C.T) @ res

            # beta = relaxed_ridge(X[:, active_terms], y, alpha, C, yc)

        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = beta

        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef based on (biased) variance of beta
        betavar = np.zeros(Nf)*np.nan
        invXtX = np.linalg.pinv(X[:, active_terms].T @ X[:, active_terms])
        betavar[active_terms] = np.diag(invXtX)/np.square(beta+1e-15)

        j = np.nanargmax(betavar)

        if Xc is not None:
            j1 = j
            for k in range(np.sum(active_terms)-1):
                # Try if constraint is still satitsfied:
                active_terms[j1] = False
                C2 = Xc[:, active_terms]
                dof = np.sum(np.abs(C2) > 0, axis=1) + (np.abs(yc) > 0)
                if np.all(dof > 1):
                    # Exit loop
                    j = j1
                    break
                else:
                    active_terms[j1] = True
                    # Exclude from search
                    betavar[j1] = np.nan
                    j1 = np.nanargmax(betavar)
            if k == np.sum(active_terms)-2:
                print("Warning: could not satisfy constraint.")

        active_terms[j] = False

    return models_list


def SSR_varD(X, y, alpha=0.0001, Xc=None, yc=None):
    """Sparse Relaxed Regularized Regression.
    See https://doi.org/10.1109/ACCESS.2018.2886528
    Remove the term with highest uncertainty.
    Remove correlated items

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
    print('Running SSR_var...')
    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    for i in trange(Nf):
        d = np.sum(active_terms)
        M = X[:, active_terms].T @ X[:, active_terms] + alpha*np.eye(d)
        invXtX = np.linalg.pinv(M)
        beta = invXtX @ X[:, active_terms].T @ y

        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = beta
        if Xc is not None:
            if Xc.ndim == 1:
                Xc = Xc.reshape((1, Xc.size))
            C = Xc[:, active_terms]
            if yc is None:
                yc = np.zeros(len(Xc))
            res = C @ beta - yc
            coefs[active_terms] += - invXtX @ C.T @ np.linalg.pinv(C @ invXtX @
                                                                   C.T) @ res
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef based on (biased) variance of beta
        betavar = np.zeros((Nf, Nf))*np.nan
        dd = invXtX/np.outer(beta, beta)
        dd = -dd + np.diag(np.diag(dd))
        betavar[np.outer(active_terms, active_terms)] = dd.flatten()
        ii = np.nanargmax(betavar)
        (j, k) = np.unravel_index(ii, (Nf, Nf))
        if j == k:
            # Remove coefficient of high variance
            active_terms[j] = False
            print(f'Removing term {j}.')
        else:
            # Remove high correlation
            if betavar[j, j] > betavar.reshape((Nf, Nf))[k, k]:
                active_terms[j] = False
                print(f'Removing correlated term {j}.')
            else:
                active_terms[k] = False
                print(f'Removing correlated term {k}.')

    return models_list


def SSR_var_softmax(X, y, alpha=0.0001, Xc=None, yc=None):
    """Sparse Relaxed Regularized Regression.
    See https://doi.org/10.1109/ACCESS.2018.2886528
    Remove the term with highest uncertainty.

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
    print('Running SSR var softmax...')
    Nf = X.shape[1]
    active_terms = np.ones(Nf, dtype=bool)
    models_list = []

    for i in trange(Nf):
        d = np.sum(active_terms)
        M = X[:, active_terms].T @ X[:, active_terms] + alpha*np.eye(d)
        invXtX = np.linalg.pinv(M)
        beta = invXtX @ X[:, active_terms].T @ y

        if Xc is not None:
            if Xc.ndim == 1:
                Xc = Xc.reshape((1, Xc.size))
            C = Xc[:, active_terms]
            if yc is None:
                yc = np.zeros(len(Xc))
            res = C @ beta - yc
            beta += - invXtX @ C.T @ np.linalg.pinv(C @ invXtX @ C.T) @ res
        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = beta
        error = np.average(np.square(y-np.sum(X[:, active_terms] *
                                              coefs[active_terms], axis=1)))
        # Record model
        models_list.append(model(coefficients=coefs, error=error))

        # Find and remove smallest coef based on (biased) variance of beta
        score = np.square(beta)/(error*np.diag(invXtX))
        feature_index = np.arange(Nf)
        # Proba density

        p = np.exp(-np.sqrt(np.abs(score)))
        #p = 1./(np.sqrt(np.abs(score)) + 1e-15)
        p = p/np.sum(p)
        #print(np.min(p), np.max(p))
        if any(np.isnan(p)):
            print(score)

        j = np.random.choice(feature_index[active_terms], p=p)
        # print('Removing term {}.'.format(j))
        active_terms[j] = False

    return models_list


def STLSQ_p(X, y, p_value=0.05, alpha=0.0001):
    """Sequential thresholded least-squares.
    See https://doi.org/10.1073/pnas.1517384113
    At each step, remove feature with high p_value.

    Parameters
    ----------
    X : array_like
        The matrix of features.
    y : array_like
        Vector to fit.
    p_value : float, default 0.05
        Threshold value. Note that no assumption is made on the scale of each
        feature.

    Returns
    -------
    model
        Model selected and fit.
    """
    #from scipy.stats import t
    # Explicit least squares with ridge regularisation
    Nf = X.shape[1]
    M = X.T @ X + alpha*np.eye(Nf)
    invXtX = np.linalg.pinv(M)
    beta = invXtX @ X.T @ y
    error = np.average(np.square(y-np.sum(X*beta, axis=1)))
    # t stat
    t0 = np.abs(beta)/np.sqrt(error*np.diag(invXtX))
    # Degrees of freedom
    #df = len(X)-Nf-1
    #p = 2*t.sf(t0, df)
    p = 1./t0

    models_list = []
    models_list.append(model(coefficients=beta, error=error))

    active_terms = np.ones(Nf, dtype=bool)
    while any(p > p_value*np.sum(active_terms)/Nf):
        # kill active terms below `threshold`
        active_terms[p > p_value*np.sum(active_terms)/Nf] = False

        # Re-fit
        d = np.sum(active_terms)
        M = X[:, active_terms].T @ X[:, active_terms] + alpha*np.eye(d)
        invXtX = np.linalg.pinv(M)
        beta = invXtX @ X[:, active_terms].T @ y

        coefs = np.zeros(Nf)*np.nan
        coefs[active_terms] = beta
        error = np.average(np.square(y-np.sum(X[:, active_terms]*beta,
                                              axis=1)))
        models_list.append(model(coefficients=coefs, error=error))
        # t stat
        t0 = np.abs(beta)/np.sqrt(error*np.diag(invXtX))
        # Degrees of freedom
        #df = len(X)-d-1
        p = np.zeros(Nf)*np.nan
        p[active_terms] = 1./t0
        print(np.nanmax(p))

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
